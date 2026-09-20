"""Model construction, ensembling, evaluation metrics and persistence."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from churn_prediction.config import (
    F1_WEIGHT,
    MODEL_NAMES,
    MODELS_DIR,
    PARAMS_DIR,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)

MODEL_BUNDLE_PATH = MODELS_DIR / "churn_models.joblib"


# --- Parameters ----------------------------------------------------------------


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model_params(name: str, params_dir: str | Path = PARAMS_DIR) -> dict:
    """Tuned hyper-parameters of one model (``models/params/<name>.json``)."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{name}'; expected one of {MODEL_NAMES}")
    return load_json(Path(params_dir) / f"{name}.json")


def load_ensemble_weights(params_dir: str | Path = PARAMS_DIR) -> dict[str, float]:
    """Ensemble weights per model; validated to cover all models and sum to 1."""
    weights = load_json(Path(params_dir) / "ensemble.json")["weights"]
    if set(weights) != set(MODEL_NAMES):
        raise ValueError(f"Ensemble weights must cover exactly {MODEL_NAMES}")
    if any(w < 0 for w in weights.values()):
        raise ValueError("Ensemble weights must be non-negative")
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"Ensemble weights must sum to 1, got {sum(weights.values())}")
    return {name: float(weights[name]) for name in MODEL_NAMES}


def load_thresholds(params_dir: str | Path = PARAMS_DIR) -> dict[str, float]:
    """Decision thresholds for the ensemble and each individual model."""
    thresholds = {
        k: float(v) for k, v in load_json(Path(params_dir) / "thresholds.json").items()
    }
    for name, value in thresholds.items():
        if not 0 < value < 1:
            raise ValueError(f"Threshold for {name} must be in (0, 1), got {value}")
    return thresholds


# --- Model construction ----------------------------------------------------------


def build_model(name: str, params: Mapping) -> Pipeline:
    """``StandardScaler`` + classifier configured with the tuned ``params``."""
    params = dict(params)
    if name == "xgboost":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            **params, random_state=RANDOM_STATE, eval_metric="logloss"
        )
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(**params, random_state=RANDOM_STATE, verbose=-1)
    elif name == "catboost":
        from catboost import CatBoostClassifier

        estimator = CatBoostClassifier(
            **params,
            random_state=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        )
    elif name == "logreg":
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression(**params, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model '{name}'; expected one of {MODEL_NAMES}")
    # pandas output keeps feature names all the way to the estimator
    return make_pipeline(StandardScaler().set_output(transform="pandas"), estimator)


def build_models(params_dir: str | Path = PARAMS_DIR) -> dict[str, Pipeline]:
    """All models of the ensemble, built from their saved hyper-parameters."""
    return {
        name: build_model(name, load_model_params(name, params_dir))
        for name in MODEL_NAMES
    }


def fit_models(
    models: Mapping[str, Pipeline], X: pd.DataFrame, y: pd.Series
) -> dict[str, Pipeline]:
    fitted = {}
    for name, model in models.items():
        logger.info("Fitting %s on %d rows", name, len(X))
        fitted[name] = model.fit(X, y)
    return fitted


# --- Prediction ------------------------------------------------------------------


def predict_probas(models: Mapping[str, Pipeline], X: pd.DataFrame) -> pd.DataFrame:
    """Churn probability of every model, one column per model."""
    return pd.DataFrame(
        {name: model.predict_proba(X)[:, 1] for name, model in models.items()},
        index=X.index,
    )


def ensemble_proba(probas: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Weighted average of the individual model probabilities."""
    missing = set(weights) - set(probas.columns)
    if missing:
        raise ValueError(f"Missing probabilities for models: {sorted(missing)}")
    total = sum(weights.values())
    blended = sum(probas[name] * w for name, w in weights.items()) / total
    return blended.rename("ensemble")


def risk_tier(proba: pd.Series | np.ndarray, threshold: float) -> pd.Series:
    """Bucket probabilities: High (>= threshold), Medium (>= threshold/2), Low."""
    proba = pd.Series(proba)
    tiers = np.select(
        [proba >= threshold, proba >= threshold / 2], ["High", "Medium"], default="Low"
    )
    return pd.Series(
        pd.Categorical(tiers, categories=["Low", "Medium", "High"], ordered=True),
        index=proba.index,
        name="risk_tier",
    )


@dataclass
class ModelBundle:
    """Fitted models plus everything needed to score new users."""

    models: dict[str, Pipeline]
    feature_names: list[str]
    weights: dict[str, float]
    thresholds: dict[str, float]
    versions: dict[str, str] = field(default_factory=dict)

    @property
    def threshold(self) -> float:
        return self.thresholds["ensemble"]

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Per-model and ensemble probabilities, the churn decision and a risk tier."""
        missing = [c for c in self.feature_names if c not in features.columns]
        if missing:
            raise ValueError(f"Features are missing columns: {missing}")
        X = features[self.feature_names].fillna(0)
        out = predict_probas(self.models, X)
        out["ensemble"] = ensemble_proba(out, self.weights)
        out["will_churn"] = (out["ensemble"] >= self.threshold).astype(int)
        out["risk_tier"] = risk_tier(out["ensemble"], self.threshold)
        return out


def library_versions() -> dict[str, str]:
    versions = {}
    for package in ["scikit-learn", "xgboost", "xgboost-cpu", "lightgbm", "catboost"]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def save_bundle(bundle: ModelBundle, path: str | Path = MODEL_BUNDLE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: str | Path = MODEL_BUNDLE_PATH) -> ModelBundle:
    """Load fitted models. Only load files you created yourself (joblib = pickle)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained models at {path}. Run `churn train`.")
    bundle = joblib.load(path)
    current = library_versions()
    for package, trained_with in bundle.versions.items():
        if current.get(package) not in (None, trained_with):
            logger.warning(
                "%s was %s at training time but is %s now; retrain the models.",
                package,
                trained_with,
                current[package],
            )
    return bundle


# --- Metrics ---------------------------------------------------------------------


def combined_score(
    y_true, proba, threshold: float = 0.5, f1_weight: float = F1_WEIGHT
) -> float:
    """Project metric: ``f1_weight`` x macro-F1 + (1 - ``f1_weight``) x ROC-AUC."""
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, proba)
    return f1_weight * f1 + (1 - f1_weight) * auc


def evaluate_predictions(y_true, proba, threshold: float) -> dict[str, float]:
    """Classification metrics of ``proba`` at ``threshold`` (macro averages)."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "churn_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "churn_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "combined": float(combined_score(y_true, proba, threshold)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_sweep(y_true, proba, thresholds=None) -> pd.DataFrame:
    """Macro F1 / precision / recall and predicted churn rate for each threshold."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    rows = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "f1_macro": f1_score(y_true, y_pred, average="macro"),
                "precision_macro": precision_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(y_true, y_pred, average="macro"),
                "predicted_churn_rate": y_pred.mean(),
            }
        )
    return pd.DataFrame(rows)


def roc_points(y_true, proba) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(y_true, proba)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def feature_importance(
    models: Mapping[str, Pipeline],
    feature_names: list[str],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Per-model importances normalised to sum to 1, plus their weighted blend.

    Raw importances are on different scales (split counts for LightGBM, gain
    shares for XGBoost, percentages for CatBoost, coefficients for logistic
    regression), so each model is normalised before blending. Logistic
    regression uses absolute coefficients.
    """
    table = pd.DataFrame(index=feature_names)
    for name, model in models.items():
        estimator = model[-1]
        if hasattr(estimator, "feature_importances_"):
            raw = np.asarray(estimator.feature_importances_, dtype=float)
        else:
            raw = np.abs(np.asarray(estimator.coef_, dtype=float)[0])
        table[name] = raw / raw.sum() if raw.sum() else raw
    table["ensemble"] = sum(table[name] * weights[name] for name in models)
    return (
        table.rename_axis("feature")
        .reset_index()
        .sort_values("ensemble", ascending=False, ignore_index=True)
    )
