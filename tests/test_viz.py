"""Chart builders must produce valid Vega-Lite specs with the fixed palette."""

import json

import numpy as np
import pandas as pd
import pytest

from churn_prediction import analysis, viz
from churn_prediction.config import MODEL_NAMES
from churn_prediction.modeling import roc_points, threshold_sweep


def _spec(chart) -> str:
    return json.dumps(chart.to_dict())  # to_dict validates against the schema


@pytest.fixture
def val():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    return pd.DataFrame({"y": y, "p": np.clip(0.3 * y + 0.7 * rng.random(300), 0, 1)})


def test_distribution_and_churn_rate_charts(feature_table):
    dist = analysis.distribution_by_label(feature_table, "churn_risk_score")
    spec = _spec(viz.distribution_chart(dist, "churn_risk_score"))
    assert viz.BLUE in spec and viz.ORANGE in spec

    rates = analysis.churn_rate_by_quantile(feature_table, "churn_risk_score")
    assert "Overall" in _spec(viz.churn_rate_bars(rates, 0.3))


def test_correlation_charts(feature_table):
    corr = analysis.target_correlations(feature_table)
    spec = _spec(viz.correlation_bars(corr))
    assert viz.RED in spec and viz.BLUE in spec
    assert "Negative" in spec
    long = analysis.correlation_long(
        feature_table, ["level", "churn_risk_score", "total_events"]
    )
    assert '"domain": [-1, 0, 1]' in _spec(viz.correlation_heatmap(long))


def test_model_charts(val):
    metrics = pd.DataFrame(
        {"model": list(viz.MODEL_COLORS), "roc_auc": [0.9, 0.8, 0.7, 0.6, 0.5]}
    )
    spec = _spec(viz.metric_bars(metrics, "roc_auc", "ROC-AUC"))
    for color in viz.MODEL_COLORS.values():
        assert color in spec

    _spec(viz.threshold_chart(threshold_sweep(val["y"], val["p"]), 0.4))
    _spec(viz.confusion_chart(10, 2, 3, 5))

    roc = pd.concat(
        [
            roc_points(val["y"], val["p"]).assign(model=name, auc=0.8)
            for name in viz.MODEL_COLORS
        ]
    )
    assert "AUC 0.800" in _spec(viz.roc_chart(roc))


def test_probability_histogram(val):
    data = pd.DataFrame(
        {"probability": val["p"], "label": val["y"].map({0: "Retained", 1: "Churned"})}
    )
    spec = _spec(viz.probability_histogram(data, 0.44))
    assert "Threshold 0.44" in spec
    assert viz.BLUE in spec and viz.ORANGE in spec


def test_importance_and_percentile_charts():
    importance = pd.DataFrame(
        {"feature": list("abcdef"), "ensemble": np.linspace(300, 50, 6)}
    )
    spec = _spec(viz.importance_bars(importance, "ensemble", "Importance", top_n=3))
    assert '"f"' not in spec  # only the top 3 are shown
    assert "Importance" in spec

    pct = pd.DataFrame(
        {"feature": ["a", "b"], "user_value": [1.0, 2.0], "population_median": [1.5, 1.0],
         "percentile": [30.0, 80.0]}
    )  # fmt: skip
    _spec(viz.percentile_chart(pct))


@pytest.fixture
def submissions():
    rng = np.random.default_rng(1)
    flags = pd.DataFrame(
        rng.integers(0, 2, (50, 5)), columns=["ensemble", *MODEL_NAMES]
    ).assign(subscription=["Free", "Paid"] * 25)
    return analysis.add_votes(flags, MODEL_NAMES)


def test_rate_comparison_chart(submissions):
    rates = analysis.predicted_vs_actual(
        submissions, ["ensemble", *MODEL_NAMES], actual=0.205
    )
    spec = _spec(viz.rate_comparison_chart(rates, "Actual (training)"))
    assert "Actual (training) 20.5%" in spec
    for color in viz.MODEL_COLORS.values():
        assert color in spec


def test_vote_and_agreement_charts(submissions):
    votes = analysis.vote_distribution(submissions, len(MODEL_NAMES))
    spec = _spec(viz.vote_bars(votes, len(MODEL_NAMES)))
    assert "4 of 4" in spec
    assert all(color in spec for color in viz.VOTE_COLORS)

    agreement = analysis.agreement_matrix(submissions, ["ensemble", *MODEL_NAMES])
    assert "CatBoost" in _spec(viz.agreement_heatmap(agreement))


def test_flag_distribution_chart(submissions):
    submissions["x"] = np.arange(len(submissions), dtype=float)
    dist = analysis.distribution_by_label(
        submissions, "x", label_col="ensemble", names=analysis.FLAG_NAMES, clip=False
    )
    spec = _spec(viz.distribution_chart(dist, "x", viz.FLAG_COLORS))
    assert "Not flagged" in spec and viz.ORANGE in spec
