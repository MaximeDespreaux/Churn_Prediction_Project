"""Pure-pandas summaries used by the Streamlit app (kept here so they are testable)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from churn_prediction.config import TARGET_COL
from churn_prediction.data import feature_columns, require_columns

LABEL_NAMES = {0: "Retained", 1: "Churned"}


def summarize_users(df: pd.DataFrame) -> dict[str, float]:
    """Headline numbers for a (filtered) user-level feature table."""
    summary = {
        "users": len(df),
        "median_sessions": float(df["sessionId_count"].median()) if len(df) else np.nan,
        "median_events": float(df["total_events"].median()) if len(df) else np.nan,
        "paid_share": float(df["level"].mean()) if len(df) else np.nan,
    }
    if TARGET_COL in df:
        summary["churn_rate"] = float(df[TARGET_COL].mean()) if len(df) else np.nan
    return summary


def clip_to_quantiles(
    values: pd.Series, lower: float = 0.01, upper: float = 0.99
) -> pd.Series:
    """Drop values outside the given quantiles (for readable distributions)."""
    if values.empty:
        return values
    lo, hi = values.quantile([lower, upper])
    return values[values.between(lo, hi)]


def distribution_by_label(
    df: pd.DataFrame,
    feature: str,
    bins: int = 30,
    clip: bool = True,
    label_col: str = TARGET_COL,
    names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Share of each group's users falling in each histogram bin of ``feature``.

    Groups are the values of ``label_col`` (the churn label by default), named
    with ``names``. Shares (not counts) keep unequal groups comparable.
    """
    names = names or LABEL_NAMES
    require_columns(df, [feature, label_col], "Feature table")
    data = df[[feature, label_col]].dropna()
    if clip:
        data = data.loc[clip_to_quantiles(data[feature]).index]
    if data.empty:
        return pd.DataFrame(columns=["bin_start", "bin_end", "label", "share", "users"])
    edges = np.histogram_bin_edges(data[feature], bins=bins)
    rows = []
    for label, group in data.groupby(label_col):
        counts, _ = np.histogram(group[feature], bins=edges)
        rows.append(
            pd.DataFrame(
                {
                    "bin_start": edges[:-1],
                    "bin_end": edges[1:],
                    "label": names.get(int(label), str(label)),
                    "share": counts / counts.sum() if counts.sum() else counts,
                    "users": counts,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def churn_rate_by_quantile(df: pd.DataFrame, feature: str, q: int = 10) -> pd.DataFrame:
    """Churn rate within quantile bins of ``feature`` (ties merged)."""
    require_columns(df, [feature, TARGET_COL], "Feature table")
    data = df[[feature, TARGET_COL]].dropna()
    if data.empty:
        return pd.DataFrame(columns=["bin", "low", "high", "users", "churn_rate"])
    bins = pd.qcut(data[feature], q=q, duplicates="drop")
    out = (
        data.groupby(bins, observed=True)[TARGET_COL]
        .agg(users="size", churn_rate="mean")
        .reset_index(names="interval")
    )
    out["low"] = out["interval"].map(lambda i: i.left).astype(float)
    out["high"] = out["interval"].map(lambda i: i.right).astype(float)
    out["bin"] = [
        f"{i + 1}. {_fmt(lo)} – {_fmt(hi)}"
        for i, (lo, hi) in enumerate(zip(out["low"], out["high"], strict=True))
    ]
    return out[["bin", "low", "high", "users", "churn_rate"]]


def _fmt(value: float) -> str:
    return f"{value:,.3g}" if abs(value) < 1000 else f"{value:,.0f}"


def target_correlations(
    df: pd.DataFrame, features: list[str] | None = None, target: str = TARGET_COL
) -> pd.Series:
    """Pearson correlation of each feature with ``target`` (the churn label by
    default), sorted descending."""
    require_columns(df, [target], "Feature table")
    features = features or feature_columns(df)
    with np.errstate(divide="ignore", invalid="ignore"):  # constant columns -> NaN
        corr = df[features].corrwith(df[target].astype(float))
    return corr.dropna().sort_values(ascending=False).rename("correlation")


def correlation_long(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Correlation matrix in long format (feature_a, feature_b, correlation)."""
    corr = df[features].corr()
    corr.index, corr.columns = list(features), list(features)
    long = corr.stack().rename("correlation").reset_index()
    long.columns = ["feature_a", "feature_b", "correlation"]
    return long


def top_correlated_pairs(
    df: pd.DataFrame, features: list[str], n: int = 5
) -> pd.DataFrame:
    """The ``n`` most strongly (absolutely) correlated distinct feature pairs."""
    corr = df[features].corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("correlation").reset_index()
    pairs.columns = ["feature_a", "feature_b", "correlation"]
    order = pairs["correlation"].abs().sort_values(ascending=False).index
    return pairs.loc[order].head(n).reset_index(drop=True)


def user_percentiles(
    user: pd.Series, population: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Where one user sits in the population for each feature (0–100 percentile)."""
    rows = []
    for feature in features:
        values = population[feature].dropna()
        value = user[feature]
        pct = float((values < value).mean() * 100) if pd.notna(value) else np.nan
        rows.append(
            {
                "feature": feature,
                "user_value": value,
                "population_median": float(values.median()),
                "percentile": pct,
            }
        )
    return pd.DataFrame(rows)


# --- Test-set predictions (submission files) -------------------------------------------

FLAG_NAMES = {0: "Not flagged", 1: "Flagged"}


def add_votes(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Add ``votes``: how many of ``models`` flag each user as a churner."""
    require_columns(preds, models, "Predictions")
    return preds.assign(votes=preds[models].sum(axis=1).astype(int))


def threshold_flags(
    probas: pd.DataFrame, thresholds: Mapping[str, float]
) -> pd.DataFrame:
    """Turn per-model probabilities into 0/1 churn flags at each model's threshold."""
    require_columns(probas, list(thresholds), "Predictions")
    return pd.DataFrame(
        {m: (probas[m] >= t).astype(int) for m, t in thresholds.items()},
        index=probas.index,
    )


def predicted_vs_actual(
    flags: pd.DataFrame,
    models: list[str],
    actual: str | float | Mapping,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Predicted churn rate of each model next to the actual churn rate.

    Args:
        flags: one 0/1 column per model (plus ``group_col`` / label column).
        models: model columns to compare.
        actual: name of a 0/1 label column in ``flags``, a single rate, or a
            ``{group: rate}`` mapping when the truth comes from another table.
        group_col: optional column to break the comparison down by.
    """
    require_columns(flags, models, "Predictions")
    groups = (
        flags.groupby(group_col, sort=True) if group_col else [("All users", flags)]
    )
    rows = []
    for group, part in groups:
        if isinstance(actual, str):
            actual_rate = float(part[actual].mean())
        elif isinstance(actual, Mapping):
            actual_rate = float(actual[group])
        else:
            actual_rate = float(actual)
        for model in models:
            rows.append(
                {
                    "group": group,
                    "model": model,
                    "users": len(part),
                    "predicted_churners": int(part[model].sum()),
                    "predicted_rate": float(part[model].mean()),
                    "actual_rate": actual_rate,
                }
            )
    out = pd.DataFrame(rows)
    out["difference"] = out["predicted_rate"] - out["actual_rate"]
    return out


def vote_distribution(preds: pd.DataFrame, n_models: int) -> pd.DataFrame:
    """Users by number of models flagging them (0..``n_models``), including zeros."""
    require_columns(preds, ["votes"], "Predictions")
    counts = preds["votes"].value_counts().reindex(range(n_models + 1), fill_value=0)
    total = counts.sum()
    return pd.DataFrame(
        {
            "votes": counts.index.astype(int),
            "users": counts.to_numpy(),
            "share": counts.to_numpy() / total if total else np.nan,
        }
    )


def agreement_matrix(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Pairwise share of users on which two models make the same call (long format)."""
    require_columns(preds, models, "Predictions")
    rows = []
    for a in models:
        for b in models:
            same = preds[a] == preds[b]
            both = (preds[a] == 1) & (preds[b] == 1)
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "agreement": float(same.mean()) if len(preds) else np.nan,
                    "both_flagged": int(both.sum()),
                }
            )
    return pd.DataFrame(rows)


def compare_groups(
    df: pd.DataFrame, flag_col: str, features: list[str]
) -> pd.DataFrame:
    """Median of each feature for flagged vs not-flagged users."""
    require_columns(df, [flag_col, *features], "Predictions")
    medians = df.groupby(flag_col)[features].median().T
    medians = medians.reindex(columns=[1, 0])
    return pd.DataFrame(
        {
            "feature": features,
            "flagged_median": medians[1].to_numpy(),
            "not_flagged_median": medians[0].to_numpy(),
        }
    )
