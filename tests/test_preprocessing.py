import numpy as np
import pandas as pd
import pytest

from churn_prediction import preprocessing as pp
from churn_prediction.config import RAW_COLUMNS, TARGET_COL

import notebook_reference as ref
from conftest import DAY_MS, REGISTRATION, T0, make_events


def _events(rows):
    """Build a raw event log from (userId, day offset, page[, auth]) tuples."""
    records = []
    for i, row in enumerate(rows):
        user, day, page, *auth = row
        records.append(
            (user, T0 + int(day * DAY_MS), auth[0] if auth else "Logged In", page,
             "free", REGISTRATION, 1, i)
        )  # fmt: skip
    return pd.DataFrame(records, columns=RAW_COLUMNS)


def test_add_churn_flag():
    df = _events(
        [
            ("a", 0, "Home"),
            ("a", 1, "Cancellation Confirmation"),
            ("b", 0, "Home", "Cancelled"),
            ("b", 1, "Cancel"),  # visiting Cancel alone is not a churn
        ]
    )
    assert pp.add_churn_flag(df)["churn_flag"].tolist() == [0, 1, 1, 0]


def test_convert_timestamps_parses_and_sorts():
    df = _events([("b", 2, "Home"), ("a", 5, "Home"), ("a", 1, "Home")])
    out = pp.convert_timestamps(df)
    assert pd.api.types.is_datetime64_any_dtype(out["ts"])
    assert out["userId"].tolist() == ["a", "a", "b"]
    assert out["ts"].is_monotonic_increasing is False  # sorted per user, not globally
    assert out["ts"].iloc[0] == pd.Timestamp("2018-10-02")
    # idempotent on already-converted data
    pd.testing.assert_frame_equal(pp.convert_timestamps(out), out)


def test_encode_level():
    df = pd.DataFrame({"level": ["free", "paid", "free"]})
    assert pp.encode_level(df)["level"].tolist() == [0, 1, 0]


def test_add_time_since_registration():
    df = pp.convert_timestamps(_events([("a", 0, "Home")]))
    expected = (pd.Timestamp("2018-10-01") - REGISTRATION).total_seconds()
    assert pp.add_time_since_registration(df)["timeSinceRegistered"].iloc[0] == expected


def test_add_time_in_activity_restarts_per_user():
    df = pp.convert_timestamps(
        _events([("a", 0, "Home"), ("a", 2, "Home"), ("b", 5, "Home")])
    )
    out = pp.add_time_in_activity(df)
    assert out["timeInSession"].tolist() == [0.0, 2 * 86_400.0, 0.0]


def test_add_cumulative_risk_accumulates_per_user():
    df = pp.convert_timestamps(
        _events(
            [
                ("a", 0, "NextSong"),
                ("a", 1, "Downgrade"),
                ("a", 2, "Logout"),  # unweighted page
                ("a", 3, "Cancel"),
                ("b", 0, "Downgrade"),  # new user starts from zero
            ]
        )
    )
    out = pp.add_cumulative_risk(df)
    np.testing.assert_allclose(out["churn_risk_score"], [0, 0.16, 0.16, 7.16, 0.16])
    np.testing.assert_allclose(
        out["not_churn_score"], [-1.925, -1.925, -1.925, -1.925, 0]
    )


def test_forward_looking_churn_labels_and_filters_churner():
    # user "c" cancels on day 20
    df = _events(
        [
            ("c", 0, "Home"),  # 20 days before churn  -> kept, label 0
            ("c", 12, "Home"),  # 8 days before         -> kept, label 1
            ("c", 19.5, "Home"),  # 0.5 day before     -> dropped (buffer)
            ("c", 20, "Cancellation Confirmation", "Cancelled"),  # churn -> dropped
            ("c", 21, "Home"),  # after churn           -> dropped
        ]
    )
    out = pp.forward_looking_churn(pp.convert_timestamps(pp.add_churn_flag(df)))
    assert out["ts"].tolist() == [
        pd.Timestamp("2018-10-01"),
        pd.Timestamp("2018-10-13"),
    ]
    assert out[TARGET_COL].tolist() == [0, 1]
    assert "churn_flag" not in out


def test_forward_looking_churn_drops_last_window_of_non_churners():
    df = _events(
        [("n", 0, "Home"), ("n", 5, "Home"), ("n", 14, "Home"), ("n", 30, "Home")]
    )
    out = pp.forward_looking_churn(pp.convert_timestamps(pp.add_churn_flag(df)))
    # cutoff = day 30 - 10 days = day 20
    assert len(out) == 3
    assert out[TARGET_COL].eq(0).all()


def test_forward_looking_churn_without_buffer_keeps_events_before_churn():
    df = _events(
        [("c", 0, "Home"), ("c", 0.9, "Home"), ("c", 1, "Cancellation Confirmation")]
    )
    prepared = pp.convert_timestamps(pp.add_churn_flag(df))
    assert len(pp.forward_looking_churn(prepared, buffer_days=0)) == 2
    assert len(pp.forward_looking_churn(prepared, buffer_days=1)) == 0


def test_forward_looking_churn_window_is_inclusive():
    df = _events([("c", 0, "Home"), ("c", 10, "Cancellation Confirmation")])
    out = pp.forward_looking_churn(pp.convert_timestamps(pp.add_churn_flag(df)))
    assert out[TARGET_COL].tolist() == [1]  # exactly 10 days before


def test_preprocess_events_test_mode_keeps_everything(events):
    out = pp.preprocess_events(events, is_train=False)
    assert len(out) == len(events)
    assert TARGET_COL not in out and "churn_flag" not in out
    assert {"churn_risk_score", "not_churn_score", "timeSinceRegistered"} <= set(
        out.columns
    )


def test_preprocess_events_does_not_mutate_input(events):
    before = events.copy()
    pp.preprocess_events(events)
    pd.testing.assert_frame_equal(events, before)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("is_train", [True, False])
def test_preprocess_events_matches_notebook(seed, is_train):
    events = make_events(n_users=15, seed=seed)
    expected = ref.preprocess_data(events, is_train=is_train)
    actual = pp.preprocess_events(events, is_train=is_train)
    columns = [c for c in expected.columns if c != "churn_flag"]
    pd.testing.assert_frame_equal(
        actual[columns].reset_index(drop=True),
        expected[columns].reset_index(drop=True),
        check_dtype=False,
    )
