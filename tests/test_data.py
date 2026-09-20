import numpy as np
import pandas as pd
import pytest

from churn_prediction import data
from churn_prediction.config import RAW_COLUMNS, SUBMISSION_FILES, TARGET_COL, USER_COL
from churn_prediction.data import DataValidationError

# --- require_columns -------------------------------------------------------------


def test_require_columns_passes_when_present():
    data.require_columns(pd.DataFrame(columns=["a", "b"]), ["a"], "Table")


def test_require_columns_lists_missing_columns():
    with pytest.raises(
        DataValidationError, match=r"Table is missing required columns: \['c'\]"
    ):
        data.require_columns(pd.DataFrame(columns=["a"]), ["a", "c"], "Table")


# --- raw events --------------------------------------------------------------------


def test_clean_raw_events_keeps_only_required_columns(events):
    raw = events.assign(firstName="Ann", location="Paris")
    cleaned = data.clean_raw_events(raw)
    assert list(cleaned.columns) == RAW_COLUMNS
    assert len(cleaned) == len(events)


def test_clean_raw_events_drops_rows_without_user(events):
    raw = events.copy().astype({USER_COL: object})
    raw.loc[0, USER_COL] = None
    raw.loc[1, USER_COL] = ""
    cleaned = data.clean_raw_events(raw)
    assert len(cleaned) == len(events) - 2
    assert cleaned.index.equals(pd.RangeIndex(len(cleaned)))


def test_clean_raw_events_casts_user_ids_to_str(events):
    raw = events.assign(**{USER_COL: events[USER_COL].astype(int)})
    assert data.clean_raw_events(raw)[USER_COL].map(type).eq(str).all()


def test_clean_raw_events_parses_registration_strings(events):
    raw = events.assign(registration=events["registration"].astype(str))
    cleaned = data.clean_raw_events(raw)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["registration"])


def test_clean_raw_events_rejects_unknown_level(events):
    raw = events.copy()
    raw.loc[0, "level"] = "premium"
    with pytest.raises(DataValidationError, match="premium"):
        data.clean_raw_events(raw)


def test_clean_raw_events_rejects_missing_registration(events):
    raw = events.copy()
    raw.loc[0, "registration"] = pd.NaT
    with pytest.raises(DataValidationError, match="registration"):
        data.clean_raw_events(raw)


def test_clean_raw_events_rejects_non_numeric_ts(events):
    raw = events.assign(ts=events["ts"].astype(str))
    with pytest.raises(DataValidationError, match="'ts'"):
        data.clean_raw_events(raw)


def test_clean_raw_events_rejects_missing_column(events):
    with pytest.raises(DataValidationError, match="page"):
        data.clean_raw_events(events.drop(columns="page"))


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_load_raw_events_round_trip(tmp_path, events, suffix):
    path = tmp_path / f"events{suffix}"
    extra = events.assign(song="x")
    if suffix == ".parquet":
        extra.to_parquet(path)
    else:
        extra.to_csv(path, index=False)
    loaded = data.load_raw_events(path)
    assert list(loaded.columns) == RAW_COLUMNS
    assert len(loaded) == len(events)
    assert loaded["ts"].tolist() == events["ts"].tolist()


def test_load_raw_events_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="README"):
        data.load_raw_events(tmp_path / "nope.parquet")


def test_load_raw_events_unsupported_suffix(tmp_path):
    path = tmp_path / "events.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported"):
        data.load_raw_events(path)


def test_filter_events_by_user_accepts_non_string_ids(events):
    out = data.filter_events(events, user_ids=[1_000_000, "1000001"])
    assert set(out[USER_COL]) == {"1000000", "1000001"}


def test_filter_events_by_page(events):
    out = data.filter_events(events, pages=["Home", "Help"])
    assert set(out["page"]) <= {"Home", "Help"}
    assert len(out) == events["page"].isin(["Home", "Help"]).sum()


def test_filter_events_time_window_is_inclusive(events):
    ts = np.sort(events["ts"].unique())
    start, end = pd.to_datetime(ts[2], unit="ms"), pd.to_datetime(ts[5], unit="ms")
    out = data.filter_events(events, start=start, end=end)
    assert out["ts"].min() == ts[2]
    assert out["ts"].max() == ts[5]


def test_filter_events_without_filters_returns_everything(events):
    assert data.filter_events(events).equals(events)


def test_sample_users_is_deterministic(events):
    a = data.sample_users(events, 3, random_state=1)
    b = data.sample_users(events, 3, random_state=1)
    assert a.equals(b)
    assert a[USER_COL].nunique() == 3
    # all events of the chosen users are kept
    chosen = set(a[USER_COL])
    assert len(a) == events[USER_COL].isin(chosen).sum()


def test_sample_users_rejects_too_many(events):
    with pytest.raises(ValueError, match="only"):
        data.sample_users(events, 1000)


# --- feature tables ------------------------------------------------------------------


def test_load_features_casts_user_id(tmp_path, feature_table):
    path = tmp_path / "features.parquet"
    feature_table.assign(**{USER_COL: feature_table[USER_COL].astype(int)}).to_parquet(
        path
    )
    loaded = data.load_features(path)
    assert loaded[USER_COL].map(type).eq(str).all()


def test_load_features_missing_file_mentions_build_step(tmp_path):
    with pytest.raises(FileNotFoundError, match="churn build-features"):
        data.load_features(tmp_path / "missing.parquet")


def test_load_train_features_requires_target(tmp_path, feature_table):
    path = tmp_path / "features.parquet"
    feature_table.drop(columns=TARGET_COL).to_parquet(path)
    assert TARGET_COL not in data.load_test_features(path)
    with pytest.raises(DataValidationError, match=TARGET_COL):
        data.load_train_features(path)


def test_feature_columns_excludes_id_and_target(feature_table):
    cols = data.feature_columns(feature_table)
    assert USER_COL not in cols and TARGET_COL not in cols
    assert len(cols) == feature_table.shape[1] - 2


def test_prepare_features_fills_nans_and_orders_columns(feature_table):
    order = ["session_length_std", "level"]
    X = data.prepare_features(feature_table, order)
    assert list(X.columns) == order
    assert not X.isna().any().any()
    assert feature_table["session_length_std"].isna().any()  # input untouched


def test_prepare_features_missing_column(feature_table):
    with pytest.raises(DataValidationError):
        data.prepare_features(feature_table, ["not_a_feature"])


def test_split_features_target(feature_table):
    X, y = data.split_features_target(feature_table)
    assert len(X) == len(y) == len(feature_table)
    assert set(y.unique()) <= {0, 1}
    assert TARGET_COL not in X


def test_filter_users_by_level(feature_table):
    paid = data.filter_users(feature_table, level="paid")
    assert (paid["level"] == 1).all()
    assert len(paid) + len(data.filter_users(feature_table, level="free")) == len(
        feature_table
    )


def test_filter_users_by_churn(feature_table):
    churned = data.filter_users(feature_table, churned=True)
    assert (churned[TARGET_COL] == 1).all()
    assert len(churned) == feature_table[TARGET_COL].sum()


def test_filter_users_by_inclusive_range(feature_table):
    values = feature_table["total_events"].sort_values(ignore_index=True)
    lo, hi = values[10], values[50]
    out = data.filter_users(feature_table, ranges={"total_events": (lo, hi)})
    assert len(out) == 41  # both bounds are inclusive
    assert out["total_events"].between(lo, hi).all()


def test_filter_users_combines_filters(feature_table):
    out = data.filter_users(
        feature_table, level="free", churned=False, ranges={"level": (0, 0)}
    )
    assert ((out["level"] == 0) & (out[TARGET_COL] == 0)).all()


def test_filter_users_rejects_bad_arguments(feature_table):
    with pytest.raises(ValueError, match="level"):
        data.filter_users(feature_table, level="gold")
    with pytest.raises(ValueError, match="Invalid range"):
        data.filter_users(feature_table, ranges={"level": (1, 0)})
    with pytest.raises(DataValidationError):
        data.filter_users(feature_table, ranges={"unknown": (0, 1)})
    with pytest.raises(DataValidationError):
        data.filter_users(feature_table.drop(columns=TARGET_COL), churned=True)


def test_churn_rate(feature_table):
    assert data.churn_rate(feature_table) == pytest.approx(
        feature_table[TARGET_COL].mean()
    )
    assert np.isnan(data.churn_rate(feature_table.iloc[:0]))


# --- submissions ------------------------------------------------------------------------


def _write_submission(path, ids, targets):
    pd.DataFrame({"id": ids, "target": targets}).to_csv(path, index=False)


def test_load_submission(tmp_path):
    path = tmp_path / "sub.csv"
    _write_submission(path, [1000001, 1000002], [0, 1])
    sub = data.load_submission(path)
    assert list(sub.columns) == [USER_COL, "target"]
    assert sub[USER_COL].tolist() == ["1000001", "1000002"]
    assert sub["target"].tolist() == [0, 1]


@pytest.mark.parametrize(
    ("ids", "targets", "message"),
    [([1, 1], [0, 1], "duplicate"), ([1, 2], [0, 2], "non-binary")],
)
def test_load_submission_rejects_invalid_files(tmp_path, ids, targets, message):
    path = tmp_path / "sub.csv"
    _write_submission(path, ids, targets)
    with pytest.raises(DataValidationError, match=message):
        data.load_submission(path)


def test_load_submission_missing_file_or_columns(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_submission(tmp_path / "missing.csv")
    path = tmp_path / "sub.csv"
    pd.DataFrame({"id": [1], "prediction": [1]}).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="target"):
        data.load_submission(path)


def test_load_submissions_joins_models(tmp_path):
    _write_submission(tmp_path / "a.csv", [1, 2, 3], [0, 1, 1])
    _write_submission(tmp_path / "b.csv", [3, 2, 1], [0, 0, 1])  # other order
    table = data.load_submissions(tmp_path, {"model_a": "a.csv", "model_b": "b.csv"})
    assert list(table.columns) == [USER_COL, "model_a", "model_b"]
    assert table.set_index(USER_COL).loc["1"].tolist() == [0, 1]
    assert table.set_index(USER_COL).loc["3"].tolist() == [1, 0]


def test_load_submissions_requires_same_users(tmp_path):
    _write_submission(tmp_path / "a.csv", [1, 2], [0, 1])
    _write_submission(tmp_path / "b.csv", [1, 3], [0, 1])
    with pytest.raises(DataValidationError, match="same users"):
        data.load_submissions(tmp_path, {"a": "a.csv", "b": "b.csv"})
    with pytest.raises(ValueError, match="No submission"):
        data.load_submissions(tmp_path, {})


def test_committed_submissions_cover_the_test_users():
    table = data.load_submissions()
    assert list(table.columns) == [USER_COL, *SUBMISSION_FILES]
    test_users = data.load_test_features()[USER_COL]
    assert set(table[USER_COL]) == set(test_users)
    assert table[list(SUBMISSION_FILES)].isin([0, 1]).all().all()
