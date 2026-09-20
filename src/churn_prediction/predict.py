"""Score a user-level feature table with the trained models.

Usage:
    churn predict                       # test set
    churn predict --submission reports/submissions/submission_ensemble.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from churn_prediction.config import REPORTS_DIR, TEST_FEATURES_PATH, USER_COL
from churn_prediction.data import load_features
from churn_prediction.modeling import MODEL_BUNDLE_PATH, ModelBundle, load_bundle

logger = logging.getLogger(__name__)


def score_users(bundle: ModelBundle, features: pd.DataFrame) -> pd.DataFrame:
    """One row per user: id, ensemble probability, decision, tier, model probabilities."""
    scores = bundle.predict(features)
    scores.insert(0, USER_COL, features[USER_COL].to_numpy())
    return scores.rename(columns={"ensemble": "churn_probability"})


def to_submission(scores: pd.DataFrame) -> pd.DataFrame:
    """Competition format: ``id`` (user id) and binary ``target``."""
    return pd.DataFrame(
        {"id": scores[USER_COL].astype("int64"), "target": scores["will_churn"]}
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=TEST_FEATURES_PATH)
    parser.add_argument("--models", type=Path, default=MODEL_BUNDLE_PATH)
    parser.add_argument(
        "--out", type=Path, default=REPORTS_DIR / "test_predictions.csv"
    )
    parser.add_argument("--submission", type=Path, help="Also write id,target CSV")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    scores = score_users(load_bundle(args.models), load_features(args.features))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out, index=False, float_format="%.6f")
    logger.info("Wrote %d predictions to %s", len(scores), args.out)
    if args.submission:
        to_submission(scores).to_csv(args.submission, index=False)
        logger.info("Wrote submission to %s", args.submission)


if __name__ == "__main__":
    main()
