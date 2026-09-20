"""End-to-end tests of the command-line pipeline on small synthetic data."""

import json

import numpy as np
import pandas as pd
import pytest

from churn_prediction import build_features, evaluate, predict, train, tune
from churn_prediction.config import (
    MODEL_NAMES,
    SAMPLE_EVENTS_PATH,
    TARGET_COL,
    TEST_FEATURES_PATH,
    USER_COL,
)
from churn_prediction.data import load_raw_events
from churn_prediction.modeling import build_model, build_models, load_bundle

from conftest import make_events


def test_build_features_cli(tmp_path):
    raw_train, raw_test = tmp_path / "train.parquet", tmp_path / "test.parquet"
    make_events(n_users=10, seed=1).to_parquet(raw_train)
    make_events(n_users=5, seed=2).to_csv(raw_test.with_suffix(".csv"), index=False)
    out_train, out_test = (
        tmp_path / "out" / "train.parquet",
        tmp_path / "out" / "test.parquet",
    )

    build_features.main(
        ["--raw-train", str(raw_train), "--raw-test", str(raw_test.with_suffix(".csv")),
         "--train-out", str(out_train), "--test-out", str(out_test)]
    )  # fmt: skip

    train_table, test_table = pd.read_parquet(out_train), pd.read_parquet(out_test)
    assert TARGET_COL in train_table and TARGET_COL not in test_table
    assert len(test_table) == 5
    assert list(test_table.columns) == [
        c for c in train_table.columns if c != TARGET_COL
    ]


def test_build_features_only_flag(tmp_path):
    raw = tmp_path / "test.parquet"
    make_events(n_users=4).to_parquet(raw)
    out_train, out_test = tmp_path / "train_out.parquet", tmp_path / "test_out.parquet"
    build_features.main(
        ["--only", "test", "--raw-test", str(raw), "--raw-train", str(tmp_path / "missing"),
         "--test-out", str(out_test), "--train-out", str(out_train)]
    )  # fmt: skip
    assert out_test.exists() and not out_train.exists()


def test_build_features_only_train(tmp_path):
    raw = tmp_path / "train.parquet"
    make_events(n_users=6).to_parquet(raw)
    out_train, out_test = tmp_path / "train_out.parquet", tmp_path / "test_out.parquet"
    build_features.main(
        ["--only", "train", "--raw-train", str(raw), "--raw-test", str(tmp_path / "missing"),
         "--test-out", str(out_test), "--train-out", str(out_train)]
    )  # fmt: skip
    assert out_train.exists() and not out_test.exists()
    assert TARGET_COL in pd.read_parquet(out_train)


def test_write_sample(tmp_path):
    raw = tmp_path / "events.parquet"
    make_events(n_users=8).to_parquet(raw)
    sample = build_features.write_sample(raw, tmp_path / "sample.parquet", n_users=3)
    assert sample[USER_COL].nunique() == 3
    assert pd.read_parquet(tmp_path / "sample.parquet").equals(sample)


@pytest.mark.model
def test_evaluate_reports_all_models(feature_table, fast_params_dir):
    report, preds = evaluate.evaluate(feature_table, fast_params_dir)
    assert list(report["models"]) == ["ensemble", *MODEL_NAMES]
    assert report["split"]["n_validation"] == len(preds) == 24
    assert report["split"]["n_features"] == feature_table.shape[1] - 2
    for metrics in report["models"].values():
        assert 0 <= metrics["roc_auc"] <= 1
        assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == 24
    assert list(preds.columns) == [USER_COL, TARGET_COL, "ensemble", *MODEL_NAMES]
    # validation users keep their ids and labels
    merged = preds.merge(
        feature_table[[USER_COL, TARGET_COL]], on=USER_COL, suffixes=("", "_orig")
    )
    assert (merged[TARGET_COL] == merged[f"{TARGET_COL}_orig"]).all()
    json.dumps(report)  # serialisable


@pytest.mark.model
def test_evaluate_split_is_stratified_and_reproducible(feature_table, fast_params_dir):
    _, a = evaluate.evaluate(feature_table, fast_params_dir)
    _, b = evaluate.evaluate(feature_table, fast_params_dir)
    pd.testing.assert_frame_equal(a, b)
    assert a[TARGET_COL].mean() == pytest.approx(
        feature_table[TARGET_COL].mean(), abs=0.05
    )


@pytest.mark.model
def test_evaluate_cli_writes_outputs(tmp_path, feature_table, fast_params_dir):
    features = tmp_path / "features.parquet"
    feature_table.to_parquet(features)
    metrics, preds = tmp_path / "r" / "metrics.json", tmp_path / "r" / "preds.parquet"
    evaluate.main(
        ["--features", str(features), "--params-dir", str(fast_params_dir),
         "--metrics-out", str(metrics), "--predictions-out", str(preds)]
    )  # fmt: skip
    assert json.loads(metrics.read_text())["models"]["ensemble"]["threshold"] == 0.44
    assert len(pd.read_parquet(preds)) == 24


@pytest.mark.model
def test_train_and_predict_cli(tmp_path, feature_table, fast_params_dir):
    features = tmp_path / "features.parquet"
    feature_table.to_parquet(features)
    bundle_path = tmp_path / "models" / "bundle.joblib"
    importance = tmp_path / "importance.csv"
    train.main(
        ["--features", str(features), "--params-dir", str(fast_params_dir),
         "--out", str(bundle_path), "--importance-out", str(importance)]
    )  # fmt: skip
    bundle = load_bundle(bundle_path)
    assert bundle.feature_names == [
        c for c in feature_table.columns if c not in (USER_COL, TARGET_COL)
    ]
    assert set(pd.read_csv(importance)["feature"]) == set(bundle.feature_names)

    test_features = tmp_path / "test.parquet"
    feature_table.drop(columns=TARGET_COL).to_parquet(test_features)
    out, submission = tmp_path / "preds.csv", tmp_path / "submission.csv"
    predict.main(
        ["--features", str(test_features), "--models", str(bundle_path),
         "--out", str(out), "--submission", str(submission)]
    )  # fmt: skip
    scores = pd.read_csv(out, dtype={USER_COL: str})
    assert scores[USER_COL].tolist() == feature_table[USER_COL].tolist()
    assert scores["churn_probability"].between(0, 1).all()
    sub = pd.read_csv(submission)
    assert list(sub.columns) == ["id", "target"]
    assert sub["target"].isin([0, 1]).all()
    assert (sub["target"] == scores["will_churn"]).all()

    only_scores = tmp_path / "only_scores.csv"
    predict.main(["--features", str(test_features), "--models", str(bundle_path),
                  "--out", str(only_scores)])  # fmt: skip
    pd.testing.assert_frame_equal(
        pd.read_csv(only_scores, dtype={USER_COL: str}), scores
    )
    assert sorted(p.name for p in tmp_path.glob("*.csv")) == [
        "importance.csv",
        "only_scores.csv",
        "preds.csv",
        "submission.csv",
    ]


def test_sample_events_rebuild_committed_features():
    """Real raw events of 25 test users -> exactly the committed feature rows."""
    if not SAMPLE_EVENTS_PATH.exists():
        pytest.skip("sample events not generated")
    features = build_features.events_to_features(
        load_raw_events(SAMPLE_EVENTS_PATH), is_train=False
    )
    assert features[USER_COL].nunique() == len(features) == 25
    committed = pd.read_parquet(TEST_FEATURES_PATH).set_index(USER_COL)
    rebuilt = features.set_index(USER_COL)
    pd.testing.assert_frame_equal(
        rebuilt, committed.loc[rebuilt.index], check_dtype=False
    )


def test_search_ensemble_finds_best_combination():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    noise = pd.DataFrame(rng.random((200, 4)), columns=MODEL_NAMES)
    probas = noise.assign(catboost=np.clip(y * 0.6 + 0.3 * rng.random(200), 0, 1))
    weights, threshold, score = tune.search_ensemble(
        y, probas, step=0.1, thresholds=np.linspace(0.1, 0.8, 8)
    )
    assert sum(weights.values()) == pytest.approx(1)
    assert weights["catboost"] == max(weights.values())  # the only informative model
    assert 0.1 <= threshold <= 0.8
    assert 0 < score <= 1


@pytest.mark.model
def test_tune_model_smoke(feature_table):
    pytest.importorskip("optuna")
    X, y = (
        feature_table.drop(columns=[USER_COL, TARGET_COL]).fillna(0),
        feature_table[TARGET_COL],
    )
    study = tune.tune_model("logreg", X, y, n_trials=2, cv=3)
    assert len(study.trials) == 2
    params = study.best_trial.user_attrs["params"]
    assert params["solver"] in {"lbfgs", "liblinear"}
    assert 0 <= study.best_value <= 1


def test_search_spaces_cover_all_models():
    assert set(tune.SEARCH_SPACES) == set(MODEL_NAMES)
    assert set(build_models()) == set(MODEL_NAMES)


def test_build_features_cli_writes_a_sample(tmp_path, monkeypatch):
    raw = tmp_path / "test.parquet"
    make_events(n_users=6).to_parquet(raw)
    sample_path = tmp_path / "sample" / "events.parquet"
    monkeypatch.setattr(build_features, "SAMPLE_EVENTS_PATH", sample_path)
    build_features.main(["--raw-test", str(raw), "--sample-users", "2"])
    assert pd.read_parquet(sample_path)[USER_COL].nunique() == 2


@pytest.mark.parametrize(
    ("name", "fixed"),
    [
        ("xgboost", {"max_depth": 3, "learning_rate": 0.01, "n_estimators": 700,
                     "scale_pos_weight": 4.0, "subsample": 0.7,
                     "colsample_bytree": 0.7, "min_child_weight": 1}),
        ("lightgbm", {"max_depth": 3, "learning_rate": 0.01, "n_estimators": 700,
                      "scale_pos_weight": 4.0, "num_leaves": 31}),
        ("catboost", {"depth": 4, "learning_rate": 0.01, "iterations": 700,
                      "scale_pos_weight": 4.0}),
        ("logreg", {"penalty": "l1", "C": 1.0, "max_iter": 100}),
    ],
)  # fmt: skip
def test_search_spaces_build_valid_models(name, fixed):
    optuna = pytest.importorskip("optuna")
    params = tune.SEARCH_SPACES[name](optuna.trial.FixedTrial(fixed), 4.0)
    model = build_model(name, params)
    estimator_params = model[-1].get_params()
    for key, value in params.items():
        assert estimator_params[key] == value
    if name == "logreg":
        assert params["solver"] == "liblinear"  # the only lbfgs-free l1 solver


def test_search_ensemble_default_threshold_grid():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 100)
    probas = pd.DataFrame(rng.random((100, 4)), columns=MODEL_NAMES)
    weights, threshold, _ = tune.search_ensemble(y, probas, step=0.3)
    assert sum(weights.values()) == pytest.approx(1)
    assert weights["catboost"] >= 0.3
    default_grid = np.round(np.arange(0.05, 0.8, 0.005), 4)
    assert threshold in default_grid


class _FakeStudy:
    def __init__(self, params):
        self.best_value = 0.5
        self.best_trial = type("Trial", (), {"user_attrs": {"params": params}})()


@pytest.mark.model
@pytest.mark.parametrize("write", [True, False])
def test_tune_cli_all_models(
    tmp_path, monkeypatch, feature_table, fast_params_dir, write
):
    """main() tunes every model, fits them, searches the ensemble and saves the results.

    The Optuna search (test_tune_model_smoke) and the ensemble grid search
    (test_search_ensemble_*) are tested for real above; here they are replaced so
    that only main()'s own work runs: the split, fitting the tuned models and
    writing the JSON files.
    """
    from conftest import FAST_PARAMS

    features = tmp_path / "features.parquet"
    feature_table.to_parquet(features)
    searched = []

    def fake_tune(name, X, y, n_trials, cv=5):
        searched.append((name, n_trials))
        return _FakeStudy(FAST_PARAMS[name])

    weights = {"xgboost": 0.1, "lightgbm": 0.2, "catboost": 0.6, "logreg": 0.1}
    monkeypatch.setattr(tune, "tune_model", fake_tune)
    seen = {}

    def fake_search(y, probas):
        seen["models"] = list(probas.columns)
        seen["rows"] = len(probas)
        return weights, 0.33, 0.7

    monkeypatch.setattr(tune, "search_ensemble", fake_search)
    before = {p.name: p.read_text() for p in fast_params_dir.glob("*.json")}

    args = ["--features", str(features), "--params-dir", str(fast_params_dir),
            "--n-trials", "3"]  # fmt: skip
    tune.main([*args, "--write"] if write else args)

    assert searched == [(name, 3) for name in MODEL_NAMES]
    # the ensemble search gets real validation probabilities of the fitted models
    assert seen == {"models": MODEL_NAMES, "rows": 24}
    if not write:
        after = {p.name: p.read_text() for p in fast_params_dir.glob("*.json")}
        assert after == before
        return
    for name in MODEL_NAMES:
        saved = json.loads((fast_params_dir / f"{name}.json").read_text())
        assert saved == FAST_PARAMS[name]
    ensemble = json.loads((fast_params_dir / "ensemble.json").read_text())
    assert ensemble == {"weights": weights}
    thresholds = json.loads((fast_params_dir / "thresholds.json").read_text())
    assert thresholds["ensemble"] == 0.33
    assert thresholds["xgboost"] == 0.5  # other thresholds are kept


def test_tune_cli_single_model_skips_the_ensemble(
    tmp_path, monkeypatch, feature_table, fast_params_dir
):
    """With a subset of models there is no ensemble to search (search replaced)."""
    features = tmp_path / "features.parquet"
    feature_table.to_parquet(features)
    before = (fast_params_dir / "logreg.json").read_text()
    monkeypatch.setattr(
        tune, "tune_model", lambda *a, **k: _FakeStudy({"C": 0.5, "max_iter": 50})
    )
    monkeypatch.setattr(
        tune, "search_ensemble", lambda *a: pytest.fail("ensemble search must not run")
    )
    tune.main(["--features", str(features), "--params-dir", str(fast_params_dir),
               "--models", "logreg"])  # fmt: skip
    # without --write nothing is saved
    assert (fast_params_dir / "logreg.json").read_text() == before
