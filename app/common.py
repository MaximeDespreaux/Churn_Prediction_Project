"""Cached data access and small UI helpers shared by the app pages.

The app only reads committed files (feature tables, evaluation reports and
submission files); it never fits or loads models.
"""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from churn_prediction import config
from churn_prediction.analysis import add_votes
from churn_prediction.config import MODEL_NAMES, TARGET_COL, USER_COL
from churn_prediction.data import (
    load_submissions,
    load_test_features,
    load_train_features,
)
from churn_prediction.modeling import load_model_params

MODEL_LABELS = config.MODEL_LABELS
ALL_MODELS = ["ensemble", *MODEL_NAMES]
LEVEL_NAMES = {0: "Free", 1: "Paid"}


# --- Data -------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading training users…")
def train_features() -> pd.DataFrame:
    return load_train_features(config.TRAIN_FEATURES_PATH)


@st.cache_data(show_spinner="Loading test users…")
def test_features() -> pd.DataFrame:
    return load_test_features(config.TEST_FEATURES_PATH)


@st.cache_data
def metrics_report() -> dict | None:
    path = config.METRICS_PATH
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data
def validation_predictions() -> pd.DataFrame | None:
    """Hold-out probabilities per model, with each user's subscription level."""
    path = config.VALIDATION_PREDICTIONS_PATH
    if not path.exists():
        return None
    preds = pd.read_parquet(path)
    preds[USER_COL] = preds[USER_COL].astype(str)
    levels = train_features()[[USER_COL, "level"]]
    preds = preds.merge(levels, on=USER_COL, how="left")
    preds["subscription"] = preds["level"].map(LEVEL_NAMES)
    return preds


@st.cache_data
def notebook_importance() -> pd.DataFrame | None:
    path = config.NOTEBOOK_IMPORTANCE_PATH
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner="Loading submissions…")
def test_predictions() -> pd.DataFrame | None:
    """Submission decisions of every model joined with the test users' features."""
    try:
        subs = load_submissions(config.SUBMISSIONS_DIR)
    except FileNotFoundError:
        return None
    subs = add_votes(subs, MODEL_NAMES)
    table = subs.merge(test_features(), on=USER_COL, how="left")
    table["subscription"] = table["level"].map(LEVEL_NAMES)
    return table


@st.cache_data
def training_churn_rates() -> dict[str, float]:
    """Actual churn rate of training users, overall and per subscription level."""
    train = train_features()
    rates = train.groupby(train["level"].map(LEVEL_NAMES))[TARGET_COL].mean()
    return {"All users": float(train[TARGET_COL].mean()), **rates.to_dict()}


@st.cache_data
def model_params() -> dict[str, dict]:
    return {name: load_model_params(name) for name in MODEL_NAMES}


# --- UI helpers ---------------------------------------------------------------------


def show_chart(chart: alt.TopLevelMixin, data: pd.DataFrame, key: str) -> None:
    """Render a chart followed by a collapsible table with the same data."""
    st.altair_chart(chart, theme=None, width="stretch")
    with st.expander("Show data", expanded=False):
        st.dataframe(data, hide_index=True, width="stretch", key=f"{key}_table")


def pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def delta_style(change: float, tolerance: float = 5e-4) -> dict:
    """``st.metric`` options that hide the arrow and colour for a zero change."""
    if abs(change) < tolerance:
        return {"delta_color": "off", "delta_arrow": "off"}
    return {}


# Features examined in the research notebook's correlation analysis
KEY_FEATURES = [
    "timeSinceRegistered",
    "avg_session_gap_hours",
    "page_thumbs_up_ratio",
    "page_home_ratio",
    "page_thumbs_down_ratio",
    "page_roll_advert_ratio",
    "page_nextsong_ratio",
    "sessionId_count",
    "page_add_to_playlist_ratio",
    "std_session_gap_hours",
    "page_settings_ratio",
    "session_instability",
    "session_duration_min",
    "page_roll_advert",
]
