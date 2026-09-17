"""Project-wide constants: paths, page weights, column names and model settings."""

from __future__ import annotations

import os
from pathlib import Path

# Repository root; override with CHURN_PROJECT_ROOT (e.g. inside a container).
PROJECT_ROOT = Path(
    os.environ.get("CHURN_PROJECT_ROOT", Path(__file__).resolve().parents[2])
)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
PARAMS_DIR = MODELS_DIR / "params"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUBMISSIONS_DIR = REPORTS_DIR / "submissions"

RAW_TRAIN_PATH = RAW_DIR / "train.parquet"
RAW_TEST_PATH = RAW_DIR / "test.parquet"
TRAIN_FEATURES_PATH = PROCESSED_DIR / "train_features.parquet"
TEST_FEATURES_PATH = PROCESSED_DIR / "test_features.parquet"
SAMPLE_EVENTS_PATH = SAMPLE_DIR / "sample_events.parquet"
METRICS_PATH = REPORTS_DIR / "metrics.json"
VALIDATION_PREDICTIONS_PATH = REPORTS_DIR / "validation_predictions.parquet"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
NOTEBOOK_IMPORTANCE_PATH = REPORTS_DIR / "notebook_feature_importance.csv"

# --- Raw event log schema ----------------------------------------------------
USER_COL = "userId"
TARGET_COL = "will_churn_in_10d"

# Only these raw columns are needed; the others (names, location, ...) are PII
# or unused, so they are never loaded.
RAW_COLUMNS = [
    "userId",
    "ts",
    "auth",
    "page",
    "level",
    "registration",
    "sessionId",
    "itemInSession",
]

LEVEL_MAPPING = {"free": 0, "paid": 1}

# --- Label construction ------------------------------------------------------
PREDICTION_WINDOW_DAYS = 10
BUFFER_DAYS = 1

# --- Cumulative page-visit risk weights ----------------------------------------
# Logistic-regression coefficients of "user visited page" on churn (see notebook).
CHURN_PAGES = {
    "Cancel": 7,
    "Cancellation Confirmation": 7,
    "Downgrade": 0.16,
    "Submit Upgrade": 0.14,
    "Upgrade": 0.03,
    "Settings": 0.02,
    "Submit Downgrade": 0.01,
}

RETAIN_PAGES = {
    "Save Settings": -0.0007,
    "Thumbs Down": -0.016,
    "About": -0.054,
    "Help": -0.058,
    "Add Friend": -0.158,
    "Add to Playlist": -0.383,
    "Roll Advert": -0.422,
    "Thumbs Up": -0.716,
    "Home": -1.423,
    "NextSong": -1.925,
}

KEY_PAGES = [
    "NextSong",
    "Thumbs Up",
    "Thumbs Down",
    "Add to Playlist",
    "Home",
    "Settings",
    "Downgrade",
    "Upgrade",
    "Roll Advert",
]

# --- Modelling -----------------------------------------------------------------
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
F1_WEIGHT = 0.6  # combined score = 0.6 * macro-F1 + 0.4 * ROC-AUC

MODEL_NAMES = ["xgboost", "lightgbm", "catboost", "logreg"]

# Test-set predictions produced by the research notebook (competition format)
SUBMISSION_FILES = {
    "ensemble": "submission_ensemble.csv",
    "xgboost": "submission_xgb.csv",
    "lightgbm": "submission_lgb.csv",
    "catboost": "submission_cat.csv",
    "logreg": "submission_logreg.csv",
}

MODEL_LABELS = {
    "ensemble": "Ensemble",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "logreg": "Logistic Regression",
}
