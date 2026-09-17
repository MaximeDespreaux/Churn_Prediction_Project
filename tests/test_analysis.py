import numpy as np
import pandas as pd
import pytest

from churn_prediction import analysis
from churn_prediction.config import TARGET_COL, TRAIN_FEATURES_PATH
from churn_prediction.data import DataValidationError
from churn_prediction.features import FEATURE_DESCRIPTIONS


@pytest.fixture
def table():
    return pd.DataFrame(
        {
            "userId": [str(i) for i in range(10)],
            "level": [0, 1] * 5,
            "sessionId_count": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "total_events": [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            TARGET_COL: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )


def test_summarize_users(table):
    s = analysis.summarize_users(table)
    assert s == {
        "users": 10,
        "median_sessions": 5.5,
        "median_events": 55.0,
        "paid_share": 0.5,
        "churn_rate": 0.5,
    }


def test_summarize_users_empty(table):
    s = analysis.summarize_users(table.iloc[:0])
    assert s["users"] == 0
    assert np.isnan(s["churn_rate"])


def test_summarize_users_without_labels(table):
    s = analysis.summarize_users(table.drop(columns=TARGET_COL))
    assert "churn_rate" not in s
    assert s["users"] == 10


def test_clip_to_quantiles_removes_extremes(table):
    clipped = analysis.clip_to_quantiles(table["total_events"], 0.0, 0.9)
    assert 1000 not in clipped.values
    assert len(clipped) == 9
    assert analysis.clip_to_quantiles(pd.Series(dtype=float)).empty


def test_distribution_by_label_shares_sum_to_one(table):
    dist = analysis.distribution_by_label(table, "x", bins=5, clip=False)
    assert set(dist["label"]) == {"Retained", "Churned"}
    assert dist.groupby("label")["share"].sum().tolist() == pytest.approx([1, 1])
    assert dist["users"].sum() == len(table)
    assert (dist["bin_end"] > dist["bin_start"]).all()


def test_distribution_by_label_handles_missing_values(table):
    table.loc[0, "x"] = np.nan
    dist = analysis.distribution_by_label(table, "x", bins=3, clip=False)
    assert dist["users"].sum() == 9


def test_distribution_by_label_empty(table):
    out = analysis.distribution_by_label(table.iloc[:0], "x")
    assert out.empty and "share" in out


def test_churn_rate_by_quantile(table):
    rates = analysis.churn_rate_by_quantile(table, "x", q=2)
    assert rates["users"].tolist() == [5, 5]
    assert rates["churn_rate"].tolist() == [0.0, 1.0]
    assert rates["bin"].iloc[0].startswith("1. ")
    assert (rates["low"] < rates["high"]).all()


def test_churn_rate_by_quantile_merges_ties(table):
    table["x"] = [1.0] * 9 + [2.0]
    rates = analysis.churn_rate_by_quantile(table, "x", q=10)
    assert rates["users"].sum() == 10
    assert len(rates) < 10


def test_target_correlations_sorted_and_signed(table):
    corr = analysis.target_correlations(table, ["x", "y", "level"])
    assert corr.index[0] == "x" and corr.index[-1] == "y"
    assert corr["x"] == pytest.approx(-corr["y"])
    assert corr.is_monotonic_decreasing


def test_target_correlations_drops_constant_features(table):
    table["constant"] = 1.0
    assert "constant" not in analysis.target_correlations(table, ["x", "constant"])


def test_correlation_long_and_top_pairs(table):
    long = analysis.correlation_long(table, ["x", "y", "level"])
    assert len(long) == 9
    assert list(long.columns) == ["feature_a", "feature_b", "correlation"]
    assert long["correlation"].max() == pytest.approx(1)
    pairs = analysis.top_correlated_pairs(table, ["x", "y", "level"], n=2)
    assert len(pairs) == 2
    assert pairs.loc[0, "correlation"] == pytest.approx(-1)
    assert {pairs.loc[0, "feature_a"], pairs.loc[0, "feature_b"]} == {"x", "y"}


def test_user_percentiles(table):
    user = table.iloc[4]  # x = 5
    out = analysis.user_percentiles(user, table, ["x", "total_events"])
    assert out.set_index("feature").loc["x", "percentile"] == pytest.approx(40)
    assert out.set_index("feature").loc["x", "population_median"] == 5.5
    user = user.copy()
    user["x"] = np.nan
    assert np.isnan(analysis.user_percentiles(user, table, ["x"])["percentile"].iloc[0])


def test_every_feature_has_exactly_one_description():
    columns = pd.read_parquet(TRAIN_FEATURES_PATH).columns
    features = {c for c in columns if c not in ("userId", TARGET_COL)}
    assert set(FEATURE_DESCRIPTIONS) == features


def test_churn_rate_by_quantile_empty(table):
    out = analysis.churn_rate_by_quantile(table.iloc[:0], "x")
    assert out.empty
    assert list(out.columns) == ["bin", "low", "high", "users", "churn_rate"]


def test_distribution_by_custom_label(table):
    table["flag"] = [1, 0] * 5
    dist = analysis.distribution_by_label(
        table, "x", bins=2, clip=False, label_col="flag", names=analysis.FLAG_NAMES
    )
    assert set(dist["label"]) == {"Flagged", "Not flagged"}
    assert dist.groupby("label")["users"].sum().to_dict() == {
        "Flagged": 5,
        "Not flagged": 5,
    }


def test_target_correlations_with_custom_target(table):
    table["flag"] = 1 - table[TARGET_COL]
    corr = analysis.target_correlations(table, ["x", "y"], target="flag")
    assert corr["y"] == pytest.approx(-corr["x"])
    assert corr.index[0] == "y"


# --- predictions -------------------------------------------------------------------------


@pytest.fixture
def flags():
    return pd.DataFrame(
        {
            "ensemble": [1, 1, 0, 0, 1, 0],
            "a": [1, 0, 0, 0, 1, 1],
            "b": [1, 1, 0, 0, 1, 0],
            "truth": [1, 0, 0, 0, 1, 1],
            "subscription": ["Free", "Free", "Free", "Paid", "Paid", "Paid"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def test_threshold_flags():
    probas = pd.DataFrame({"a": [0.1, 0.5, 0.9], "b": [0.3, 0.3, 0.2]})
    out = analysis.threshold_flags(probas, {"a": 0.5, "b": 0.25})
    assert out["a"].tolist() == [0, 1, 1]  # threshold is inclusive
    assert out["b"].tolist() == [1, 1, 0]
    with pytest.raises(DataValidationError):
        analysis.threshold_flags(probas, {"c": 0.5})


def test_predicted_vs_actual_with_label_column(flags):
    out = analysis.predicted_vs_actual(flags, ["ensemble", "a"], actual="truth")
    assert out["group"].unique().tolist() == ["All users"]
    ens = out.set_index("model").loc["ensemble"]
    assert ens["predicted_rate"] == pytest.approx(0.5)
    assert ens["actual_rate"] == pytest.approx(0.5)
    assert ens["difference"] == pytest.approx(0)
    assert ens["predicted_churners"] == 3
    assert ens["users"] == 6


def test_predicted_vs_actual_by_group_with_external_truth(flags):
    out = analysis.predicted_vs_actual(
        flags, ["b"], actual={"Free": 0.1, "Paid": 0.4}, group_col="subscription"
    ).set_index("group")
    assert out.loc["Free", "predicted_rate"] == pytest.approx(2 / 3)
    assert out.loc["Free", "actual_rate"] == pytest.approx(0.1)
    assert out.loc["Paid", "difference"] == pytest.approx(1 / 3 - 0.4)


def test_predicted_vs_actual_with_constant_truth(flags):
    out = analysis.predicted_vs_actual(flags, ["a", "b"], actual=0.25)
    assert out["actual_rate"].eq(0.25).all()
    assert out["model"].tolist() == ["a", "b"]


def test_add_votes_and_vote_distribution(flags):
    voted = analysis.add_votes(flags, ["a", "b"])
    assert voted["votes"].tolist() == [2, 1, 0, 0, 2, 1]
    dist = analysis.vote_distribution(voted, n_models=2)
    assert dist["votes"].tolist() == [0, 1, 2]
    assert dist["users"].tolist() == [2, 2, 2]
    assert dist["share"].sum() == pytest.approx(1)
    # vote counts nobody reached are still listed
    wider = analysis.vote_distribution(voted, n_models=4)
    assert wider["users"].tolist() == [2, 2, 2, 0, 0]


def test_agreement_matrix(flags):
    out = analysis.agreement_matrix(flags, ["a", "b"]).set_index(["model_a", "model_b"])
    assert out.loc[("a", "a"), "agreement"] == 1
    assert out.loc[("a", "b"), "agreement"] == pytest.approx(4 / 6)
    assert out.loc[("a", "b"), "agreement"] == out.loc[("b", "a"), "agreement"]
    assert out.loc[("a", "b"), "both_flagged"] == 2


def test_compare_groups(flags):
    out = analysis.compare_groups(flags, "ensemble", ["x"]).iloc[0]
    assert out["flagged_median"] == pytest.approx(2.0)  # x = 1, 2, 5
    assert out["not_flagged_median"] == pytest.approx(4.0)  # x = 3, 4, 6
