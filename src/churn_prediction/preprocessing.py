"""Event-level preprocessing: churn flags, timestamps, cumulative risk and labels."""

from __future__ import annotations

import pandas as pd

from churn_prediction.config import (
    BUFFER_DAYS,
    CHURN_PAGES,
    LEVEL_MAPPING,
    PREDICTION_WINDOW_DAYS,
    RETAIN_PAGES,
    TARGET_COL,
    USER_COL,
)

SECONDS_PER_DAY = 86_400


def add_churn_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag cancellation events (cancelled auth status or confirmation page)."""
    churned = (df["auth"] == "Cancelled") | (df["page"] == "Cancellation Confirmation")
    return df.assign(churn_flag=churned.astype(int))


def convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert epoch-millisecond ``ts`` to datetime and sort events per user."""
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df = df.assign(ts=pd.to_datetime(df["ts"], unit="ms"))
    return df.sort_values([USER_COL, "ts"]).reset_index(drop=True)


def encode_level(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the subscription level as 0 (free) / 1 (paid)."""
    return df.assign(level=df["level"].map(LEVEL_MAPPING))


def add_time_in_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Seconds elapsed since the user's first logged event."""
    first_seen = df.groupby(USER_COL)["ts"].transform("min")
    return df.assign(timeInSession=(df["ts"] - first_seen).dt.total_seconds())


def add_time_since_registration(df: pd.DataFrame) -> pd.DataFrame:
    """Account age in seconds at the time of each event."""
    age = (df["ts"] - df["registration"]).dt.total_seconds()
    return df.assign(timeSinceRegistered=age)


def add_cumulative_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Running per-user sums of churn / retention page weights.

    ``df`` must already be sorted by user and time (see ``convert_timestamps``).
    """
    by_user = df[USER_COL]
    churn_weight = df["page"].map(CHURN_PAGES).fillna(0.0).astype(float)
    retain_weight = df["page"].map(RETAIN_PAGES).fillna(0.0).astype(float)
    return df.assign(
        churn_risk_score=churn_weight.groupby(by_user).cumsum(),
        not_churn_score=retain_weight.groupby(by_user).cumsum(),
    )


def forward_looking_churn(
    df: pd.DataFrame,
    prediction_window_days: float = PREDICTION_WINDOW_DAYS,
    buffer_days: float = BUFFER_DAYS,
) -> pd.DataFrame:
    """Turn retrospective churn flags into a forward-looking label.

    An event is labelled 1 if the user's first cancellation happens within
    ``prediction_window_days`` after it. To avoid leakage:

    * events less than ``buffer_days`` before the cancellation (and all events
      after it) are dropped;
    * for users who never churn, the last ``prediction_window_days`` of activity
      are dropped, since their outcome in that window is unknown.
    """
    df = df.sort_values([USER_COL, "ts"]).reset_index(drop=True)

    churn_time = (
        df["ts"].where(df["churn_flag"] == 1).groupby(df[USER_COL]).transform("min")
    )
    days_until_churn = (churn_time - df["ts"]).dt.total_seconds() / SECONDS_PER_DAY

    label = days_until_churn.between(0, prediction_window_days).astype(int)

    min_gap = buffer_days if buffer_days > 0 else 0
    keep = days_until_churn.isna() | (days_until_churn > min_gap)

    never_churns = churn_time.isna()
    last_seen = df["ts"].groupby(df[USER_COL]).transform("max")
    cutoff = last_seen - pd.Timedelta(days=prediction_window_days)
    keep &= ~never_churns | (df["ts"] <= cutoff)

    out = df.assign(**{TARGET_COL: label})[keep]
    return out.drop(columns=["churn_flag"]).reset_index(drop=True)


def preprocess_events(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Full event-level preprocessing used before feature engineering.

    Args:
        df: cleaned raw event log (see ``data.clean_raw_events``).
        is_train: when ``True``, build the forward-looking ``will_churn_in_10d``
            label and apply the anti-leakage filtering.
    """
    df = add_churn_flag(df)
    df = convert_timestamps(df)
    df = encode_level(df)
    df = add_time_in_activity(df)
    df = add_time_since_registration(df)
    df = add_cumulative_risk(df)
    if is_train:
        return forward_looking_churn(df)
    return df.drop(columns=["churn_flag"])
