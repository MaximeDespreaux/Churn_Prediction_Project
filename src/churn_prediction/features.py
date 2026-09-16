"""User-level feature engineering from preprocessed event logs."""

from __future__ import annotations

import pandas as pd

from churn_prediction.config import KEY_PAGES, TARGET_COL, USER_COL

# Statistics of each per-session measure that become user-level features
SESSION_AGGREGATIONS = {
    "session_length": ["mean", "std", "max", "min"],
    "session_duration": ["mean", "std", "max", "min"],
    "session_churn_risk": ["mean", "std", "max", "last"],
    "session_retain_score": ["mean", "std", "min", "last"],
}

_SESSION_MEASURES = {
    "session_length": "items played/visited per session",
    "session_duration": "session duration (minutes)",
    "session_churn_risk": "churn-risk score at the end of each session",
    "session_retain_score": "retention score at the end of each session",
}
_STAT_NAMES = {
    "mean": "Mean",
    "std": "Standard deviation of",
    "max": "Maximum",
    "min": "Minimum",
    "last": "Most recent",
}


def page_feature_name(page: str) -> str:
    """``"Thumbs Up"`` -> ``"page_thumbs_up"``."""
    return f"page_{page.lower().replace(' ', '_')}"


def _feature_descriptions() -> dict[str, str]:
    descriptions = {
        "level": "Current subscription tier (0 = free, 1 = paid).",
        "churn_risk_score": "Cumulative weight of visits to churn-indicating pages "
        "(Cancel, Downgrade, Settings, ...). Higher = riskier.",
        "not_churn_score": "Cumulative weight of engagement pages (NextSong, Home, "
        "Thumbs Up, ...). More negative = more engaged.",
        "timeSinceRegistered": "Account age in seconds at the user's last event.",
        "sessionId_count": "Number of sessions.",
        "avg_session_gap_hours": "Average hours between the starts of consecutive "
        "sessions.",
        "std_session_gap_hours": "Variability of the hours between sessions.",
        "last3_avg_length": "Average session length over the last 3 sessions.",
        "last3_avg_duration": "Average session duration (minutes) over the last 3 "
        "sessions.",
        "last3_avg_risk": "Average churn-risk score over the last 3 sessions.",
        "total_events": "Total number of logged events.",
        "risk_per_session": "churn_risk_score / (sessions + 1): concentration of risk "
        "signals.",
        "engagement_decline": "Lifetime mean session length minus the last-3-session "
        "average. Positive = engagement is dropping.",
        "session_instability": "Coefficient of variation of session length "
        "(std / (mean + 1)). High = erratic usage.",
    }
    for measure, stats in SESSION_AGGREGATIONS.items():
        for stat in stats:
            descriptions[f"{measure}_{stat}"] = (
                f"{_STAT_NAMES[stat]} {_SESSION_MEASURES[measure]}."
            )
    for page in KEY_PAGES:
        name = page_feature_name(page)
        descriptions[name] = f"Number of '{page}' events."
        descriptions[f"{name}_ratio"] = (
            f"'{page}' events / (total events + 1): share of activity."
        )
    return descriptions


FEATURE_DESCRIPTIONS = _feature_descriptions()


def build_session_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (user, session) with length, duration and end-of-session scores."""
    sessions = (
        df.groupby([USER_COL, "sessionId"])
        .agg(
            session_length=("itemInSession", "max"),
            session_churn_risk=("churn_risk_score", "last"),
            session_retain_score=("not_churn_score", "last"),
            session_start=("ts", "min"),
            session_end=("ts", "max"),
        )
        .reset_index()
    )
    duration = sessions["session_end"] - sessions["session_start"]
    sessions["session_duration"] = duration.dt.total_seconds() / 60
    return sessions


def create_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate session statistics, inter-session gaps and last-3-session trends."""
    sessions = build_session_table(df)

    features = sessions.groupby(USER_COL).agg(
        {**SESSION_AGGREGATIONS, "sessionId": "count"}
    )
    features.columns = ["_".join(col) for col in features.columns]
    features = features.reset_index()

    # Hours between consecutive session starts (sessions ordered by id, as in
    # the original research notebook the models were tuned on).
    gaps = sessions.groupby(USER_COL)["session_start"].diff().dt.total_seconds() / 3600
    gap_stats = (
        gaps.groupby(sessions[USER_COL])
        .agg(["mean", "std"])
        .rename(
            columns={"mean": "avg_session_gap_hours", "std": "std_session_gap_hours"}
        )
        .reset_index()
    )
    features = features.merge(gap_stats, on=USER_COL, how="left")

    last3 = (
        sessions.sort_values([USER_COL, "session_start"])
        .groupby(USER_COL)
        .tail(3)
        .groupby(USER_COL)
        .agg(
            last3_avg_length=("session_length", "mean"),
            last3_avg_duration=("session_duration", "mean"),
            last3_avg_risk=("session_churn_risk", "mean"),
        )
        .reset_index()
    )
    return features.merge(last3, on=USER_COL, how="left")


def create_page_features(df: pd.DataFrame) -> pd.DataFrame:
    """Counts and activity-normalised ratios of visits to the key pages."""
    counts = pd.crosstab(df[USER_COL], df["page"])
    counts = counts.reindex(columns=KEY_PAGES, fill_value=0)
    counts.columns = [page_feature_name(p) for p in KEY_PAGES]
    counts.columns.name = None

    page_cols = list(counts.columns)
    counts["total_events"] = df.groupby(USER_COL).size()
    ratios = counts[page_cols].div(counts["total_events"] + 1, axis=0)
    ratios.columns = [f"{c}_ratio" for c in page_cols]
    return pd.concat([counts, ratios], axis=1).rename_axis(USER_COL).reset_index()


def create_derived_features(features: pd.DataFrame) -> pd.DataFrame:
    """Composite features combining session and risk aggregates."""
    return features.assign(
        risk_per_session=features["churn_risk_score"]
        / (features["sessionId_count"] + 1),
        engagement_decline=features["session_length_mean"]
        - features["last3_avg_length"],
        session_instability=features["session_length_std"]
        / (features["session_length_mean"] + 1),
    )


def make_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Build the user-level feature table from a preprocessed event log.

    Returns one row per user with ``userId``, the model features and, when
    ``is_train``, the ``will_churn_in_10d`` target.
    """
    agg = {
        "level": "last",
        "churn_risk_score": "max",
        "not_churn_score": "min",
        "timeSinceRegistered": "max",
    }
    if is_train:
        agg[TARGET_COL] = "max"
    basic = df.groupby(USER_COL).agg(agg).reset_index()

    features = basic.merge(create_session_features(df), on=USER_COL).merge(
        create_page_features(df), on=USER_COL
    )
    return create_derived_features(features)
