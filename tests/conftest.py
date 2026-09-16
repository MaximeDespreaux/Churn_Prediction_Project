"""Shared fixtures: synthetic event logs, feature tables and fast model params."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_prediction.config import PARAMS_DIR, RAW_COLUMNS

DAY_MS = 86_400_000
T0 = 1_538_352_000_000  # 2018-10-01 00:00:00 UTC in ms
REGISTRATION = pd.Timestamp("2018-09-01")

PAGES = [
    "NextSong", "NextSong", "NextSong", "NextSong", "Home", "Thumbs Up",
    "Thumbs Down", "Add to Playlist", "Roll Advert", "Settings", "Help",
    "About", "Add Friend", "Downgrade", "Upgrade", "Save Settings", "Logout",
]  # fmt: skip

# Small parameter sets so model tests fit in well under a second.
FAST_PARAMS = {
    "xgboost": {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3},
    "lightgbm": {"n_estimators": 5, "num_leaves": 4, "min_child_samples": 2},
    "catboost": {"iterations": 5, "depth": 2},
    "logreg": {"C": 1.0, "max_iter": 200},
}


def make_events(n_users: int = 12, churn_every: int = 3, seed: int = 0) -> pd.DataFrame:
    """Random but realistic raw event log.

    Every ``churn_every``-th user cancels at the end of their activity.
    """
    rng = np.random.default_rng(seed)
    rows = []
    session_id = 1
    for u in range(n_users):
        user_id = str(1_000_000 + u)
        level = "paid" if u % 2 else "free"
        t = T0 + int(rng.integers(0, 2 * DAY_MS))
        n_sessions = int(rng.integers(1, 7))
        for _ in range(n_sessions):
            for item in range(int(rng.integers(1, 15))):
                rows.append(
                    (user_id, t, "Logged In", str(rng.choice(PAGES)), level,
                     REGISTRATION, session_id, item)
                )  # fmt: skip
                t += int(rng.integers(30_000, 400_000))
            session_id += 1
            t += int(rng.integers(DAY_MS // 2, 4 * DAY_MS))
        if u % churn_every == 0:
            rows.append(
                (
                    user_id,
                    t,
                    "Logged In",
                    "Cancel",
                    level,
                    REGISTRATION,
                    session_id - 1,
                    99,
                )
            )
            rows.append((user_id, t + 1000, "Cancelled", "Cancellation Confirmation",
                         level, REGISTRATION, session_id - 1, 100))  # fmt: skip
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def make_feature_table(n_users: int = 120, seed: int = 0) -> pd.DataFrame:
    """Synthetic user-level table with the real schema and a learnable signal."""
    from churn_prediction.build_features import events_to_features

    template = events_to_features(make_events(n_users=6, seed=seed), is_train=True)
    rng = np.random.default_rng(seed)
    columns = [c for c in template.columns if c not in ("userId", "will_churn_in_10d")]
    X = pd.DataFrame(rng.normal(size=(n_users, len(columns))), columns=columns)
    X["level"] = rng.integers(0, 2, n_users)
    X.loc[rng.random(n_users) < 0.1, "session_length_std"] = np.nan
    y = (X["churn_risk_score"] + 0.5 * rng.normal(size=n_users) > 0.8).astype(int)
    return pd.concat(
        [
            pd.DataFrame({"userId": [str(2_000_000 + i) for i in range(n_users)]}),
            X[["level", "churn_risk_score", "not_churn_score", "timeSinceRegistered"]],
            pd.DataFrame({"will_churn_in_10d": y}),
            X.drop(
                columns=[
                    "level",
                    "churn_risk_score",
                    "not_churn_score",
                    "timeSinceRegistered",
                ]
            ),
        ],
        axis=1,
    )


@pytest.fixture
def events() -> pd.DataFrame:
    return make_events()


@pytest.fixture
def feature_table() -> pd.DataFrame:
    return make_feature_table()


def write_fast_params(params_dir: Path) -> Path:
    """A params directory with tiny models but the real ensemble weights/thresholds."""
    params_dir.mkdir(parents=True, exist_ok=True)
    for name, params in FAST_PARAMS.items():
        (params_dir / f"{name}.json").write_text(json.dumps(params))
    for name in ["ensemble.json", "thresholds.json"]:
        shutil.copy(PARAMS_DIR / name, params_dir / name)
    return params_dir


@pytest.fixture
def fast_params_dir(tmp_path: Path) -> Path:
    return write_fast_params(tmp_path / "params")
