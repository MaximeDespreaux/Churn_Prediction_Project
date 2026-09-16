"""Verbatim (loop-based) implementation from notebooks/churn_research.ipynb.

Used only as an oracle: the refactored, vectorised package must reproduce it.
"""

import pandas as pd

from churn_prediction.config import CHURN_PAGES, KEY_PAGES, RETAIN_PAGES


def compute_cumulative_risk(user_df):
    churn_score = 0
    not_churn_score = 0
    churn_list = []
    not_churn_list = []
    for page in user_df["page"]:
        churn_score += CHURN_PAGES.get(page, 0)
        not_churn_score += RETAIN_PAGES.get(page, 0)
        churn_list.append(churn_score)
        not_churn_list.append(not_churn_score)
    user_df["churn_risk_score"] = churn_list
    user_df["not_churn_score"] = not_churn_list
    return user_df


def forward_looking_churn(df, prediction_window_days=10, buffer_days=0):
    df = df.sort_values(["userId", "ts"]).reset_index(drop=True)
    churn_events = df[df["churn_flag"] == 1].groupby("userId")["ts"].min().reset_index()
    churn_events.columns = ["userId", "churn_time"]
    df = df.merge(churn_events, on="userId", how="left")
    df["days_until_churn"] = (df["churn_time"] - df["ts"]).dt.total_seconds() / 86400
    df["will_churn_in_10d"] = (
        (df["days_until_churn"] >= 0)
        & (df["days_until_churn"] <= prediction_window_days)
    ).astype(int)
    df["will_churn_in_10d"] = df["will_churn_in_10d"].fillna(0).astype(int)
    if buffer_days > 0:
        mask_keep = (df["days_until_churn"].isna()) | (
            df["days_until_churn"] > buffer_days
        )
    else:
        mask_keep = (df["days_until_churn"].isna()) | (df["days_until_churn"] > 0)
    df_clean = df[mask_keep].copy()

    def filter_non_churner_tail(user_df):
        if user_df["churn_time"].isna().all():
            max_time = user_df["ts"].max()
            cutoff = max_time - pd.Timedelta(days=prediction_window_days)
            return user_df[user_df["ts"] <= cutoff]
        return user_df

    df_clean = df_clean.groupby("userId", group_keys=False)[df_clean.columns].apply(
        filter_non_churner_tail
    )
    df_clean = df_clean.drop(columns=["churn_time", "days_until_churn", "churn_flag"])
    return df_clean.reset_index(drop=True)


def preprocess_data(df, is_train=True):
    df = df.copy()
    df["churn_flag"] = (
        (df["auth"] == "Cancelled") | (df["page"] == "Cancellation Confirmation")
    ).astype(int)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values(["userId", "ts"]).reset_index(drop=True)
    df["churn_risk_score"] = 0.0
    df["not_churn_score"] = 0.0
    df["level"] = df["level"].map({"free": 0, "paid": 1})
    df["timeInSession"] = df.groupby("userId")["ts"].transform(
        lambda x: (x - x.min()).dt.total_seconds()
    )
    df["timeSinceRegistered"] = (df["ts"] - df["registration"]).dt.total_seconds()
    df = df.groupby("userId", group_keys=False)[df.columns].apply(
        compute_cumulative_risk
    )
    if is_train:
        df = forward_looking_churn(df, prediction_window_days=10, buffer_days=1)
    return df


def create_session_features(df):
    session_stats = (
        df.groupby(["userId", "sessionId"])
        .agg(
            {
                "itemInSession": "max",
                "churn_risk_score": "last",
                "not_churn_score": "last",
                "ts": ["min", "max"],
            }
        )
        .reset_index()
    )
    session_stats.columns = [
        "userId",
        "sessionId",
        "session_length",
        "session_churn_risk",
        "session_retain_score",
        "session_start",
        "session_end",
    ]
    session_stats["session_duration"] = (
        session_stats["session_end"] - session_stats["session_start"]
    ).dt.total_seconds() / 60
    user_session_features = (
        session_stats.groupby("userId")
        .agg(
            {
                "session_length": ["mean", "std", "max", "min"],
                "session_duration": ["mean", "std", "max", "min"],
                "session_churn_risk": ["mean", "std", "max", "last"],
                "session_retain_score": ["mean", "std", "min", "last"],
                "sessionId": "count",
            }
        )
        .reset_index()
    )
    user_session_features.columns = ["userId"] + [
        "_".join(col).strip("_") for col in user_session_features.columns.values[1:]
    ]
    session_gaps = (
        session_stats.groupby("userId")["session_start"]
        .apply(lambda x: x.diff().dt.total_seconds() / 3600)
        .reset_index()
    )
    session_gap_stats = (
        session_gaps.groupby("userId")["session_start"]
        .agg(["mean", "std"])
        .reset_index()
    )
    session_gap_stats.columns = [
        "userId",
        "avg_session_gap_hours",
        "std_session_gap_hours",
    ]
    user_session_features = user_session_features.merge(
        session_gap_stats, on="userId", how="left"
    )
    last_sessions = (
        session_stats.sort_values(["userId", "session_start"]).groupby("userId").tail(3)
    )
    last_session_agg = (
        last_sessions.groupby("userId")
        .agg(
            {
                "session_length": "mean",
                "session_duration": "mean",
                "session_churn_risk": "mean",
            }
        )
        .reset_index()
    )
    last_session_agg.columns = [
        "userId",
        "last3_avg_length",
        "last3_avg_duration",
        "last3_avg_risk",
    ]
    return user_session_features.merge(last_session_agg, on="userId", how="left")


def create_page_features(df):
    page_counts = (
        df.groupby(["userId", "page"]).size().unstack(fill_value=0).reset_index()
    )
    for page in KEY_PAGES:
        if page not in page_counts.columns:
            page_counts[page] = 0
    page_features = page_counts[
        ["userId"] + [p for p in KEY_PAGES if p in page_counts.columns]
    ]
    page_features.columns = ["userId"] + [
        f'page_{col.lower().replace(" ", "_")}' for col in page_features.columns[1:]
    ]
    total_events = df.groupby("userId").size().reset_index(name="total_events")
    page_features = page_features.merge(total_events, on="userId")
    for col in page_features.columns:
        if col.startswith("page_"):
            page_features[f"{col}_ratio"] = page_features[col] / (
                page_features["total_events"] + 1
            )
    return page_features


def make_features(df, is_train=True):
    session_features = create_session_features(df)
    page_features = create_page_features(df)
    agg_dict = {
        "level": "last",
        "churn_risk_score": "max",
        "not_churn_score": "min",
        "timeSinceRegistered": "max",
    }
    if is_train:
        agg_dict["will_churn_in_10d"] = "max"
    basic_agg = df.groupby("userId").agg(agg_dict).reset_index()
    full = basic_agg.merge(session_features, on="userId").merge(
        page_features, on="userId"
    )
    full["risk_per_session"] = full["churn_risk_score"] / (full["sessionId_count"] + 1)
    full["engagement_decline"] = full["session_length_mean"] - full["last3_avg_length"]
    full["session_instability"] = full["session_length_std"] / (
        full["session_length_mean"] + 1
    )
    return full
