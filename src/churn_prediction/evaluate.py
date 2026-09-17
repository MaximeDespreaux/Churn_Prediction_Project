"""Hold-out evaluation of the saved hyper-parameters (no tuning).

Reproduces the notebook's stratified 80/20 split, fits every model with its
saved parameters on the 80 %, and scores the 20 %.

Usage:
    churn evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_prediction.config import (
    METRICS_PATH,
    PARAMS_DIR,
    RANDOM_STATE,
    TARGET_COL,
    TRAIN_FEATURES_PATH,
    USER_COL,
    VALIDATION_PREDICTIONS_PATH,
    VALIDATION_SIZE,
)
from churn_prediction.data import load_train_features, split_features_target
from churn_prediction.modeling import (
    build_models,
    ensemble_proba,
    evaluate_predictions,
    fit_models,
    library_versions,
    load_ensemble_weights,
    load_thresholds,
    predict_probas,
)

logger = logging.getLogger(__name__)


def evaluate(
    features: pd.DataFrame,
    params_dir: Path = PARAMS_DIR,
    validation_size: float = VALIDATION_SIZE,
    models: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Fit on a stratified train split and score the validation split.

    Returns the metrics report and a per-user table of validation probabilities.
    """
    X, y = split_features_target(features)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_size, random_state=RANDOM_STATE, stratify=y
    )
    weights = load_ensemble_weights(params_dir)
    thresholds = load_thresholds(params_dir)
    models = fit_models(models or build_models(params_dir), X_train, y_train)

    probas = predict_probas(models, X_val)
    probas["ensemble"] = ensemble_proba(probas, weights)

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": {
            "validation_size": validation_size,
            "random_state": RANDOM_STATE,
            "n_train": len(X_train),
            "n_validation": len(X_val),
            "n_features": X.shape[1],
            "churn_rate": float(y.mean()),
        },
        "weights": weights,
        "library_versions": library_versions(),
        "models": {
            name: evaluate_predictions(y_val, probas[name], thresholds[name])
            for name in ["ensemble", *models]
        },
    }
    predictions = probas.assign(
        **{USER_COL: features.loc[X_val.index, USER_COL], TARGET_COL: y_val}
    )
    predictions = predictions[[USER_COL, TARGET_COL, "ensemble", *models]]
    return report, predictions.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=TRAIN_FEATURES_PATH)
    parser.add_argument("--params-dir", type=Path, default=PARAMS_DIR)
    parser.add_argument("--metrics-out", type=Path, default=METRICS_PATH)
    parser.add_argument(
        "--predictions-out", type=Path, default=VALIDATION_PREDICTIONS_PATH
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    report, predictions = evaluate(load_train_features(args.features), args.params_dir)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    predictions.to_parquet(args.predictions_out, index=False)

    table = pd.DataFrame(report["models"]).T
    print(table[["threshold", "f1_macro", "roc_auc", "combined"]].round(4).to_string())
    logger.info("Wrote %s and %s", args.metrics_out, args.predictions_out)


if __name__ == "__main__":
    main()
