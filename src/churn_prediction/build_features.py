"""Build user-level feature tables from the raw event logs.

Usage:
    churn build-features                    # train + test
    churn build-features --sample-users 25
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from churn_prediction.config import (
    RAW_TEST_PATH,
    RAW_TRAIN_PATH,
    SAMPLE_EVENTS_PATH,
    TEST_FEATURES_PATH,
    TRAIN_FEATURES_PATH,
)
from churn_prediction.data import load_raw_events, sample_users
from churn_prediction.features import make_features
from churn_prediction.preprocessing import preprocess_events

logger = logging.getLogger(__name__)


def events_to_features(events: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """Raw (cleaned) event log -> user-level feature table."""
    return make_features(
        preprocess_events(events, is_train=is_train), is_train=is_train
    )


def build(raw_path: Path, out_path: Path, is_train: bool) -> pd.DataFrame:
    logger.info("Loading %s", raw_path)
    events = load_raw_events(raw_path)
    logger.info("Building features from %d events", len(events))
    features = events_to_features(events, is_train=is_train)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    logger.info("Wrote %d users x %d columns to %s", *features.shape, out_path)
    return features


def write_sample(raw_path: Path, out_path: Path, n_users: int) -> pd.DataFrame:
    """Save the events of a few users (required columns only) for demos and tests."""
    sample = sample_users(load_raw_events(raw_path), n_users)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out_path, index=False)
    logger.info("Wrote %d events of %d users to %s", len(sample), n_users, out_path)
    return sample


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-train", type=Path, default=RAW_TRAIN_PATH)
    parser.add_argument("--raw-test", type=Path, default=RAW_TEST_PATH)
    parser.add_argument("--train-out", type=Path, default=TRAIN_FEATURES_PATH)
    parser.add_argument("--test-out", type=Path, default=TEST_FEATURES_PATH)
    parser.add_argument(
        "--only", choices=["train", "test"], help="Build a single table only"
    )
    parser.add_argument(
        "--sample-users",
        type=int,
        help="Instead of building features, save a sample of N test users' events",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.sample_users:
        write_sample(args.raw_test, SAMPLE_EVENTS_PATH, args.sample_users)
        return
    if args.only in (None, "train"):
        build(args.raw_train, args.train_out, is_train=True)
    if args.only in (None, "test"):
        build(args.raw_test, args.test_out, is_train=False)


if __name__ == "__main__":
    main()
