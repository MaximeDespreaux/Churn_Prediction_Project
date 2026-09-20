"""Hyper-parameter search with Optuna (expensive; results are already committed).

Ported from the research notebook. Requires the ``tune`` extra (``optuna``).

Usage:
    churn tune --n-trials 50 --write
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from churn_prediction.config import (
    F1_WEIGHT,
    MODEL_NAMES,
    PARAMS_DIR,
    RANDOM_STATE,
    TRAIN_FEATURES_PATH,
    VALIDATION_SIZE,
)
from churn_prediction.data import load_train_features, split_features_target
from churn_prediction.modeling import (
    build_model,
    fit_models,
    predict_probas,
)

logger = logging.getLogger(__name__)


def _xgboost_space(trial, spw: float) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 700, 1500, step=100),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", spw, spw * 1.5),
        "subsample": trial.suggest_float("subsample", 0.6, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 3),
    }


def _lightgbm_space(trial, spw: float) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 700, 1500, step=100),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", spw, spw * 1.5),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
    }


def _catboost_space(trial, spw: float) -> dict:
    return {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
        "iterations": trial.suggest_int("iterations", 700, 1500, step=100),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", spw, spw * 1.5),
    }


def _logreg_space(trial, spw: float) -> dict:
    penalty = trial.suggest_categorical("penalty", ["l2", "l1"])
    return {
        "penalty": penalty,
        "solver": "liblinear" if penalty == "l1" else "lbfgs",
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 800),
        "class_weight": "balanced",
    }


SEARCH_SPACES: dict[str, Callable] = {
    "xgboost": _xgboost_space,
    "lightgbm": _lightgbm_space,
    "catboost": _catboost_space,
    "logreg": _logreg_space,
}


def cv_scores(model, X, y, cv: int = 5) -> dict[str, float]:
    """Cross-validated macro-F1, ROC-AUC and their weighted combination."""
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model, X, y, cv=folds, scoring={"f1": "f1_macro", "auc": "roc_auc"}
    )
    f1, auc = scores["test_f1"].mean(), scores["test_auc"].mean()
    return {"f1": f1, "auc": auc, "combined": F1_WEIGHT * f1 + (1 - F1_WEIGHT) * auc}


def tune_model(name: str, X, y, n_trials: int, cv: int = 5):
    """Run an Optuna TPE search for one model and return the study."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spw = float((y == 0).sum() / (y == 1).sum())

    def objective(trial):
        params = SEARCH_SPACES[name](trial, spw)
        scores = cv_scores(build_model(name, params), X, y, cv=cv)
        trial.set_user_attr("params", params)
        trial.set_user_attr("f1_mean", scores["f1"])
        trial.set_user_attr("auc_mean", scores["auc"])
        return scores["combined"]

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def search_ensemble(
    y_true,
    probas: pd.DataFrame,
    step: float = 0.05,
    min_catboost: float = 0.3,
    thresholds=None,
) -> tuple[dict[str, float], float, float]:
    """Grid-search ensemble weights and decision threshold on validation data.

    Unlike the original notebook search, the logistic-regression probabilities
    are included in the blend they are weighted for.

    Returns ``(weights, threshold, combined_score)``.
    """
    grid = np.round(np.arange(0.1, 0.7, step), 4)
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.8, 0.005), 4)
    best = (None, None, -np.inf)
    for w_xgb, w_lgb, w_lr in product(grid, grid, grid):
        w_cat = round(1 - w_xgb - w_lgb - w_lr, 4)
        if w_cat < min_catboost:
            continue
        weights = {
            "xgboost": w_xgb,
            "lightgbm": w_lgb,
            "catboost": w_cat,
            "logreg": w_lr,
        }
        blended = sum(probas[m] * w for m, w in weights.items()).to_numpy()
        auc = roc_auc_score(y_true, blended)
        for t in thresholds:
            f1 = f1_score(y_true, (blended >= t).astype(int), average="macro")
            score = F1_WEIGHT * f1 + (1 - F1_WEIGHT) * auc
            if score > best[2]:
                best = ({k: float(v) for k, v in weights.items()}, float(t), score)
    return best


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=TRAIN_FEATURES_PATH)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--params-dir", type=Path, default=PARAMS_DIR)
    parser.add_argument(
        "--write", action="store_true", help="Overwrite models/params/*.json"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    X, y = split_features_target(load_train_features(args.features))
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    best_params = {}
    for name in args.models:
        study = tune_model(name, X_train, y_train, n_trials=args.n_trials)
        best_params[name] = study.best_trial.user_attrs["params"]
        logger.info("%s: combined=%.4f %s", name, study.best_value, best_params[name])
        if args.write:
            path = args.params_dir / f"{name}.json"
            path.write_text(json.dumps(best_params[name], indent=2) + "\n")

    if set(best_params) == set(MODEL_NAMES):
        models = {n: build_model(n, p) for n, p in best_params.items()}
        probas = predict_probas(fit_models(models, X_train, y_train), X_val)
        weights, threshold, score = search_ensemble(y_val, probas)
        logger.info(
            "Ensemble: %s threshold=%.3f combined=%.4f", weights, threshold, score
        )
        if args.write:
            (args.params_dir / "ensemble.json").write_text(
                json.dumps({"weights": weights}, indent=2) + "\n"
            )
            thresholds_path = args.params_dir / "thresholds.json"
            thresholds = json.loads(thresholds_path.read_text())
            thresholds["ensemble"] = threshold
            thresholds_path.write_text(json.dumps(thresholds, indent=2) + "\n")


if __name__ == "__main__":
    main()
