import json
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn_prediction import modeling as md
from churn_prediction.config import MODEL_NAMES, PARAMS_DIR
from churn_prediction.data import split_features_target

# --- committed parameters ----------------------------------------------------------


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_committed_params_build_valid_models(name):
    params = md.load_model_params(name)
    model = md.build_model(name, params)
    assert isinstance(model, Pipeline)
    estimator_params = model[-1].get_params()
    for key, value in params.items():
        assert estimator_params[key] == value


def test_committed_ensemble_config_is_valid():
    weights = md.load_ensemble_weights()
    assert list(weights) == MODEL_NAMES
    assert sum(weights.values()) == pytest.approx(1)
    thresholds = md.load_thresholds()
    assert set(thresholds) == {"ensemble", *MODEL_NAMES}


def test_unknown_model_name():
    with pytest.raises(ValueError, match="Unknown model"):
        md.load_model_params("svm")
    with pytest.raises(ValueError, match="Unknown model"):
        md.build_model("svm", {})


@pytest.fixture
def params_copy(tmp_path):
    target = tmp_path / "params"
    shutil.copytree(PARAMS_DIR, target)
    return target


def _write_weights(params_dir, weights):
    (params_dir / "ensemble.json").write_text(json.dumps({"weights": weights}))


def test_ensemble_weights_must_sum_to_one(params_copy):
    _write_weights(
        params_copy, {"xgboost": 0.5, "lightgbm": 0.5, "catboost": 0.5, "logreg": 0}
    )
    with pytest.raises(ValueError, match="sum to 1"):
        md.load_ensemble_weights(params_copy)


def test_ensemble_weights_must_cover_all_models(params_copy):
    _write_weights(params_copy, {"xgboost": 1.0})
    with pytest.raises(ValueError, match="cover"):
        md.load_ensemble_weights(params_copy)


def test_ensemble_weights_must_be_non_negative(params_copy):
    _write_weights(
        params_copy, {"xgboost": 1.5, "lightgbm": -0.5, "catboost": 0, "logreg": 0}
    )
    with pytest.raises(ValueError, match="non-negative"):
        md.load_ensemble_weights(params_copy)


def test_thresholds_must_be_probabilities(params_copy):
    (params_copy / "thresholds.json").write_text(json.dumps({"ensemble": 1.2}))
    with pytest.raises(ValueError, match="ensemble"):
        md.load_thresholds(params_copy)


# --- ensembling -------------------------------------------------------------------


def test_ensemble_proba_is_weighted_average():
    probas = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 1.0]})
    out = md.ensemble_proba(probas, {"a": 0.25, "b": 0.75})
    np.testing.assert_allclose(out, [0.75, 1.0])
    assert out.name == "ensemble"


def test_ensemble_proba_normalises_weights():
    probas = pd.DataFrame({"a": [0.2], "b": [0.6]})
    assert md.ensemble_proba(probas, {"a": 1, "b": 1}).iloc[0] == pytest.approx(0.4)


def test_ensemble_proba_missing_model():
    with pytest.raises(ValueError, match="Missing"):
        md.ensemble_proba(pd.DataFrame({"a": [0.1]}), {"a": 0.5, "b": 0.5})


def test_risk_tier_boundaries():
    tiers = md.risk_tier(np.array([0.0, 0.19, 0.2, 0.39, 0.4, 1.0]), threshold=0.4)
    assert tiers.tolist() == ["Low", "Low", "Medium", "Medium", "High", "High"]
    assert tiers.cat.ordered


# --- metrics ------------------------------------------------------------------------


def test_combined_score_perfect_and_weighted():
    y = [0, 0, 1, 1]
    assert md.combined_score(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    # perfect ranking but everything predicted positive at 0.5:
    # macro F1 = (0 + 2/3) / 2, AUC = 1
    assert md.combined_score(y, [0.6, 0.7, 0.8, 0.9]) == pytest.approx(0.6 / 3 + 0.4)


def test_evaluate_predictions_confusion_counts():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.2, 0.7, 0.3])
    m = md.evaluate_predictions(y, proba, threshold=0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 1, 1)
    assert m["churn_recall"] == pytest.approx(0.5)
    assert m["churn_precision"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["threshold"] == 0.5
    assert 0 <= m["combined"] <= 1


def test_evaluate_predictions_with_no_positive_predictions():
    m = md.evaluate_predictions([0, 1], [0.1, 0.2], threshold=0.9)
    assert m["churn_precision"] == 0.0
    assert m["tp"] == 0


def test_threshold_sweep():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    proba = np.clip(y * 0.4 + rng.random(200) * 0.6, 0, 1)
    sweep = md.threshold_sweep(y, proba)
    assert sweep["threshold"].is_monotonic_increasing
    assert sweep["predicted_churn_rate"].is_monotonic_decreasing
    assert (
        sweep[["f1_macro", "precision_macro", "recall_macro"]]
        .stack()
        .between(0, 1)
        .all()
    )
    custom = md.threshold_sweep(y, proba, thresholds=[0.3, 0.6])
    assert custom["threshold"].tolist() == [0.3, 0.6]


def test_roc_points():
    roc = md.roc_points([0, 1], [0.2, 0.8])
    assert roc.iloc[0].tolist() == [0.0, 0.0]
    assert roc.iloc[-1].tolist() == [1.0, 1.0]


# --- fitting, importances and persistence ------------------------------------------------


@pytest.fixture
def fitted(feature_table, fast_params_dir):
    X, y = split_features_target(feature_table)
    models = md.fit_models(md.build_models(fast_params_dir), X, y)
    return models, X, y


@pytest.mark.model
def test_fit_and_predict_probas(fitted):
    models, X, _ = fitted
    probas = md.predict_probas(models, X)
    assert list(probas.columns) == MODEL_NAMES
    assert probas.index.equals(X.index)
    assert probas.stack().between(0, 1).all()


@pytest.mark.model
def test_models_learn_the_signal(fitted):
    models, X, y = fitted
    probas = md.predict_probas(models, X)
    for name in MODEL_NAMES:
        assert md.evaluate_predictions(y, probas[name], 0.5)["roc_auc"] > 0.8, name


@pytest.mark.model
def test_feature_importance_is_normalised(fitted):
    models, X, _ = fitted
    weights = md.load_ensemble_weights()
    importance = md.feature_importance(models, list(X.columns), weights)
    assert len(importance) == X.shape[1]
    for name in [*MODEL_NAMES, "ensemble"]:
        assert importance[name].sum() == pytest.approx(1), name
        assert (importance[name] >= 0).all()
    assert importance["ensemble"].is_monotonic_decreasing
    assert importance["feature"].iloc[0] == "churn_risk_score"  # the planted signal


@pytest.mark.model
def test_bundle_predict_and_round_trip(tmp_path, fitted, fast_params_dir):
    models, X, _ = fitted
    bundle = md.ModelBundle(
        models=models,
        feature_names=list(X.columns),
        weights=md.load_ensemble_weights(fast_params_dir),
        thresholds=md.load_thresholds(fast_params_dir),
        versions=md.library_versions(),
    )
    table = X.iloc[:10][list(reversed(X.columns))]  # column order must not matter
    out = bundle.predict(table)
    assert list(out.columns) == [*MODEL_NAMES, "ensemble", "will_churn", "risk_tier"]
    assert (out["will_churn"] == (out["ensemble"] >= bundle.threshold)).all()

    path = md.save_bundle(bundle, tmp_path / "nested" / "bundle.joblib")
    loaded = md.load_bundle(path)
    pd.testing.assert_frame_equal(loaded.predict(table), out)

    with pytest.raises(ValueError, match="missing columns"):
        bundle.predict(table.drop(columns="level"))


@pytest.mark.model
def test_load_bundle_warns_on_version_mismatch(tmp_path, fitted, caplog):
    models, X, _ = fitted
    bundle = md.ModelBundle(models, list(X.columns), {}, {"ensemble": 0.5},
                            versions={"scikit-learn": "0.0.1"})  # fmt: skip
    path = md.save_bundle(bundle, tmp_path / "bundle.joblib")
    md.load_bundle(path)
    assert "retrain" in caplog.text


def test_load_bundle_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="churn train"):
        md.load_bundle(tmp_path / "missing.joblib")


def test_library_versions_reports_installed_packages():
    versions = md.library_versions()
    assert "scikit-learn" in versions
    assert "lightgbm" in versions
