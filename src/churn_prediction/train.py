"""Fit the final models on the full training set with the saved hyper-parameters.

Usage:
    churn train
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from churn_prediction.config import (
    FEATURE_IMPORTANCE_PATH,
    PARAMS_DIR,
    TRAIN_FEATURES_PATH,
)
from churn_prediction.data import load_train_features, split_features_target
from churn_prediction.modeling import (
    MODEL_BUNDLE_PATH,
    ModelBundle,
    build_models,
    feature_importance,
    fit_models,
    library_versions,
    load_ensemble_weights,
    load_thresholds,
    save_bundle,
)

logger = logging.getLogger(__name__)


def train(features: pd.DataFrame, params_dir: Path = PARAMS_DIR) -> ModelBundle:
    X, y = split_features_target(features)
    models = fit_models(build_models(params_dir), X, y)
    return ModelBundle(
        models=models,
        feature_names=list(X.columns),
        weights=load_ensemble_weights(params_dir),
        thresholds=load_thresholds(params_dir),
        versions=library_versions(),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=TRAIN_FEATURES_PATH)
    parser.add_argument("--params-dir", type=Path, default=PARAMS_DIR)
    parser.add_argument("--out", type=Path, default=MODEL_BUNDLE_PATH)
    parser.add_argument("--importance-out", type=Path, default=FEATURE_IMPORTANCE_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    bundle = train(load_train_features(args.features), args.params_dir)
    logger.info("Saved models to %s", save_bundle(bundle, args.out))

    importance = feature_importance(bundle.models, bundle.feature_names, bundle.weights)
    args.importance_out.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.importance_out, index=False, float_format="%.6f")
    logger.info("Wrote feature importances to %s", args.importance_out)


if __name__ == "__main__":
    main()
