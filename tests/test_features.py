import numpy as np
import pandas as pd
import pytest

from churn_prediction import features as ft
from churn_prediction.build_features import events_to_features
from churn_prediction.config import KEY_PAGES, TARGET_COL, TRAIN_FEATURES_PATH, USER_COL
from churn_prediction.preprocessing import preprocess_events

import notebook_reference as ref
from conftest import make_events


def _prepared(rows):
    """Preprocessed-like events from (user, session, minute, page) tuples."""
    df = pd.DataFrame(rows, columns=[USER_COL, "sessionId", "minute", "page"])
    df["ts"] = pd.Timestamp("2018-10-01") + pd.to_timedelta(
        df.pop("minute"), unit="min"
    )
    df["itemInSession"] = df.groupby([USER_COL, "sessionId"]).cumcount()
    df["churn_risk_score"] = np.arange(len(df), dtype=float)
    df["not_churn_score"] = -np.arange(len(df), dtype=float)
    return df.sort_values([USER_COL, "ts"]).reset_index(drop=True)


@pytest.mark.parametrize(
    ("page", "expected"),
    [("NextSong", "page_nextsong"), ("Thumbs Up", "page_thumbs_up"),
     ("Add to Playlist", "page_add_to_playlist")],
)  # fmt: skip
def test_page_feature_name(page, expected):
    assert ft.page_feature_name(page) == expected


def test_build_session_table():
    df = _prepared([("a", 1, 0, "Home"), ("a", 1, 30, "Home"), ("a", 2, 120, "Home")])
    sessions = ft.build_session_table(df)
    assert sessions["session_length"].tolist() == [1, 0]
    assert sessions["session_duration"].tolist() == [30.0, 0.0]
    assert sessions["session_churn_risk"].tolist() == [1.0, 2.0]  # last value


def test_create_session_features_gaps_and_last3():
    rows = [("a", s, 60 * 24 * (s - 1), "Home") for s in range(1, 6)]  # daily sessions
    rows += [("a", 3, 60 * 24 * 2 + 10, "Home")]  # session 3 lasts 10 minutes
    rows += [("b", 9, 0, "Home")]  # single-session user
    out = ft.create_session_features(_prepared(rows)).set_index(USER_COL)

    assert out.loc["a", "sessionId_count"] == 5
    assert out.loc["a", "avg_session_gap_hours"] == pytest.approx(24)
    assert out.loc["a", "std_session_gap_hours"] == pytest.approx(0)
    assert out.loc["a", "last3_avg_duration"] == pytest.approx(10 / 3)  # sessions 3-5
    assert out.loc["a", "last3_avg_length"] == pytest.approx(1 / 3)
    # std / gaps are undefined with a single session
    assert out.loc["b", ["session_length_std", "avg_session_gap_hours"]].isna().all()
    assert out.loc["b", "last3_avg_length"] == 0


def test_create_page_features_counts_ratios_and_missing_pages():
    df = _prepared(
        [("a", 1, 0, "NextSong"), ("a", 1, 1, "NextSong"), ("a", 1, 2, "Logout"),
         ("b", 2, 0, "Thumbs Up")]
    )  # fmt: skip
    out = ft.create_page_features(df).set_index(USER_COL)
    expected_cols = [ft.page_feature_name(p) for p in KEY_PAGES]
    assert list(out.columns) == [
        *expected_cols,
        "total_events",
        *[f"{c}_ratio" for c in expected_cols],
    ]
    assert out.loc["a", "page_nextsong"] == 2
    assert out.loc["a", "total_events"] == 3
    assert out.loc["a", "page_nextsong_ratio"] == pytest.approx(2 / 4)
    assert out.loc["a", "page_downgrade"] == 0  # page never visited by anyone
    assert out.loc["b", "page_thumbs_up_ratio"] == pytest.approx(1 / 2)


def test_create_derived_features():
    table = pd.DataFrame(
        {
            "churn_risk_score": [3.0],
            "sessionId_count": [2],
            "session_length_mean": [10.0],
            "last3_avg_length": [4.0],
            "session_length_std": [5.5],
        }
    )
    out = ft.create_derived_features(table).iloc[0]
    assert out["risk_per_session"] == pytest.approx(1.0)
    assert out["engagement_decline"] == pytest.approx(6.0)
    assert out["session_instability"] == pytest.approx(0.5)


def test_make_features_one_row_per_user(events):
    table = events_to_features(events, is_train=False)
    assert table[USER_COL].is_unique
    assert set(table[USER_COL]) == set(events[USER_COL])
    assert TARGET_COL not in table


def test_make_features_train_has_binary_target(events):
    table = events_to_features(events, is_train=True)
    assert set(table[TARGET_COL].unique()) <= {0, 1}
    assert table[TARGET_COL].sum() > 0


def test_make_features_matches_committed_schema(events):
    """The pipeline must produce exactly the columns the models were trained on."""
    committed = pd.read_parquet(TRAIN_FEATURES_PATH).columns.tolist()
    assert events_to_features(events, is_train=True).columns.tolist() == committed


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("is_train", [True, False])
def test_make_features_matches_notebook(seed, is_train):
    events = make_events(n_users=20, seed=seed)
    expected = ref.make_features(ref.preprocess_data(events, is_train), is_train)
    actual = ft.make_features(preprocess_events(events, is_train), is_train)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
