"""Data loading, validation and filtering helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from churn_prediction.config import (
    LEVEL_MAPPING,
    RAW_COLUMNS,
    SUBMISSION_FILES,
    SUBMISSIONS_DIR,
    TARGET_COL,
    TEST_FEATURES_PATH,
    TRAIN_FEATURES_PATH,
    USER_COL,
)


class DataValidationError(ValueError):
    """Raised when a table does not have the expected structure."""


def require_columns(df: pd.DataFrame, columns: Iterable[str], what: str) -> None:
    """Raise ``DataValidationError`` if any of ``columns`` is missing from ``df``."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"{what} is missing required columns: {missing}")


# --- Raw event logs --------------------------------------------------------------


def load_raw_events(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a raw event log (parquet or CSV), keeping only the columns the pipeline needs."""
    path = Path(path)
    columns = columns or RAW_COLUMNS
    if not path.exists():
        raise FileNotFoundError(
            f"Raw event log not found: {path}. See README > Data for how to obtain it."
        )
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=columns)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, usecols=columns)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} (use .parquet or .csv)")
    return clean_raw_events(df)


def clean_raw_events(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a raw event log and coerce its columns to the expected types.

    Rows without a user id (anonymous/guest traffic) cannot be attributed to a
    user and are dropped.
    """
    require_columns(df, RAW_COLUMNS, "Event log")
    df = df.loc[df[USER_COL].notna(), RAW_COLUMNS]
    df = df.assign(**{USER_COL: df[USER_COL].astype(str)})
    df = df[df[USER_COL].str.len() > 0]

    unknown_levels = set(df["level"].dropna().unique()) - set(LEVEL_MAPPING)
    if unknown_levels:
        raise DataValidationError(
            f"Unknown subscription levels {sorted(unknown_levels)}; "
            f"expected {sorted(LEVEL_MAPPING)}"
        )
    if not pd.api.types.is_datetime64_any_dtype(df["registration"]):
        df = df.assign(registration=pd.to_datetime(df["registration"]))
    if df["registration"].isna().any():
        raise DataValidationError("Event log has rows without a registration date")
    for col in ["ts", "sessionId", "itemInSession"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Column '{col}' must be numeric")
    return df.reset_index(drop=True)


def filter_events(
    events: pd.DataFrame,
    user_ids: Iterable[str] | None = None,
    pages: Iterable[str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter an event log by users, pages and/or an inclusive time window.

    ``start``/``end`` are compared against ``ts`` (epoch milliseconds).
    """
    mask = pd.Series(True, index=events.index)
    if user_ids is not None:
        mask &= events[USER_COL].isin([str(u) for u in user_ids])
    if pages is not None:
        mask &= events["page"].isin(list(pages))
    if start is not None:
        mask &= events["ts"] >= _to_epoch_ms(start)
    if end is not None:
        mask &= events["ts"] <= _to_epoch_ms(end)
    return events[mask]


def sample_users(
    events: pd.DataFrame, n_users: int, random_state: int = 42
) -> pd.DataFrame:
    """Return all events of ``n_users`` randomly chosen users."""
    users = events[USER_COL].unique()
    if n_users > len(users):
        raise ValueError(f"Requested {n_users} users but only {len(users)} exist")
    rng = np.random.default_rng(random_state)
    chosen = rng.choice(users, size=n_users, replace=False)
    return filter_events(events, user_ids=chosen).reset_index(drop=True)


def _to_epoch_ms(value: pd.Timestamp | str) -> int:
    return int(pd.Timestamp(value).value // 1_000_000)


# --- User-level feature tables ---------------------------------------------------


def load_features(path: str | Path) -> pd.DataFrame:
    """Load a user-level feature table produced by ``build_features``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {path}. " "Run `churn build-features` first."
        )
    df = pd.read_parquet(path)
    require_columns(df, [USER_COL], "Feature table")
    df[USER_COL] = df[USER_COL].astype(str)
    return df


def load_train_features(path: str | Path = TRAIN_FEATURES_PATH) -> pd.DataFrame:
    df = load_features(path)
    require_columns(df, [TARGET_COL], "Training feature table")
    return df


def load_test_features(path: str | Path = TEST_FEATURES_PATH) -> pd.DataFrame:
    return load_features(path)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: everything except the user id and the target."""
    return [c for c in df.columns if c not in (USER_COL, TARGET_COL)]


def prepare_features(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Select model input columns in a fixed order and fill missing values with 0.

    Missing values come from standard deviations / gaps of users with a single
    session, for which 0 is the natural value.
    """
    columns = columns or feature_columns(df)
    require_columns(df, columns, "Feature table")
    return df[columns].fillna(0)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a training feature table into ``X`` (NaN filled) and ``y``."""
    require_columns(df, [TARGET_COL], "Training feature table")
    return prepare_features(df), df[TARGET_COL].astype(int)


def filter_users(
    df: pd.DataFrame,
    level: str | None = None,
    churned: bool | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Filter a user-level feature table.

    Args:
        df: user-level feature table.
        level: ``"free"``, ``"paid"`` or ``None`` for both.
        churned: keep only churners (``True``), non-churners (``False``) or both.
        ranges: ``{column: (low, high)}`` inclusive bounds.
    """
    mask = pd.Series(True, index=df.index)
    if level is not None:
        if level not in LEVEL_MAPPING:
            raise ValueError(f"level must be one of {sorted(LEVEL_MAPPING)}")
        mask &= df["level"] == LEVEL_MAPPING[level]
    if churned is not None:
        require_columns(df, [TARGET_COL], "Feature table")
        mask &= df[TARGET_COL] == int(churned)
    for column, (low, high) in (ranges or {}).items():
        require_columns(df, [column], "Feature table")
        if low > high:
            raise ValueError(f"Invalid range for {column}: {low} > {high}")
        mask &= df[column].between(low, high)
    return df[mask]


def churn_rate(df: pd.DataFrame) -> float:
    """Share of users labelled as churning (NaN for an empty table)."""
    require_columns(df, [TARGET_COL], "Feature table")
    return float(df[TARGET_COL].mean()) if len(df) else float("nan")


# --- Submissions (test-set predictions) ----------------------------------------------


def load_submission(path: str | Path) -> pd.DataFrame:
    """Load one ``id,target`` submission file as ``userId`` (str) + ``target`` (0/1)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")
    df = pd.read_csv(path)
    require_columns(df, ["id", "target"], f"Submission {path.name}")
    if df["id"].duplicated().any():
        raise DataValidationError(f"Submission {path.name} has duplicate ids")
    if not df["target"].isin([0, 1]).all():
        raise DataValidationError(f"Submission {path.name} has non-binary targets")
    return pd.DataFrame(
        {USER_COL: df["id"].astype(str), "target": df["target"].astype(int)}
    )


def load_submissions(
    directory: str | Path = SUBMISSIONS_DIR,
    files: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """All submissions side by side: ``userId`` plus one 0/1 column per model.

    Raises ``DataValidationError`` if the files do not cover the same users.
    """
    files = SUBMISSION_FILES if files is None else files
    table = None
    for model, filename in files.items():
        sub = load_submission(Path(directory) / filename).rename(
            columns={"target": model}
        )
        if table is None:
            table = sub
            continue
        if set(sub[USER_COL]) != set(table[USER_COL]):
            raise DataValidationError(
                f"{filename} does not cover the same users as the other submissions"
            )
        table = table.merge(sub, on=USER_COL, how="left")
    if table is None:
        raise ValueError("No submission files given")
    return table
