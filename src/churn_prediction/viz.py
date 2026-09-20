"""Altair chart builders for the Streamlit app.

Colours follow a validated categorical palette (fixed order, colour follows the
entity), a blue sequential ramp and a blue <-> red diverging scale.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from churn_prediction.config import MODEL_LABELS

# Validation-set charts (ROC curves, probability histograms) exceed Altair's
# default 5,000-row embedding guard; the data stays small enough to embed.
alt.data_transformers.disable_max_rows()

# --- Palette -------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
)
RED = "#e34948"
NEUTRAL = "#f0efec"
INK, INK_SECONDARY = "#0b0b0b", "#52514e"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SEQUENTIAL = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]

LABEL_COLORS = {"Retained": BLUE, "Churned": ORANGE}
FLAG_COLORS = {"Not flagged": BLUE, "Flagged": ORANGE}
# Ordinal blue ramp for "number of models flagging a user" (0 = lightest)
VOTE_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
MODEL_COLORS = {
    MODEL_LABELS["ensemble"]: BLUE,
    MODEL_LABELS["xgboost"]: ORANGE,
    MODEL_LABELS["lightgbm"]: AQUA,
    MODEL_LABELS["catboost"]: YELLOW,
    MODEL_LABELS["logreg"]: MAGENTA,
}
METRIC_COLORS = {"Macro F1": BLUE, "Macro precision": ORANGE, "Macro recall": AQUA}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _scale(mapping: dict[str, str]) -> alt.Scale:
    return alt.Scale(domain=list(mapping), range=list(mapping.values()))


def _style(chart: alt.TopLevelMixin, height: int = 320) -> alt.TopLevelMixin:
    return (
        chart.properties(height=height, width="container")
        .configure(font=FONT, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor=GRID,
            domainColor=BASELINE,
            tickColor=BASELINE,
            labelColor=INK_SECONDARY,
            titleColor=INK_SECONDARY,
            titleFontWeight="normal",
            labelFontSize=12,
            titleFontSize=12,
        )
        .configure_axisX(grid=False)
        .configure_legend(
            orient="top",
            title=None,
            labelColor=INK_SECONDARY,
            labelFontSize=12,
            symbolStrokeWidth=2,
        )
        .configure_title(color=INK, fontSize=14, anchor="start", fontWeight=600)
    )


def distribution_chart(
    dist: pd.DataFrame, feature: str, colors: dict[str, str] | None = None
) -> alt.TopLevelMixin:
    """Step lines of the share of users per bin, one line per group."""
    base = alt.Chart(dist).encode(
        x=alt.X("bin_start:Q", title=feature),
        y=alt.Y("share:Q", title="Share of group's users", axis=alt.Axis(format="%")),
        color=alt.Color("label:N", scale=_scale(colors or LABEL_COLORS)),
    )
    lines = base.mark_line(interpolate="step-after", strokeWidth=2)
    points = base.mark_point(size=80, opacity=0, filled=True).encode(
        tooltip=[
            alt.Tooltip("label:N", title="Group"),
            alt.Tooltip("bin_start:Q", title="From", format=",.3~g"),
            alt.Tooltip("bin_end:Q", title="To", format=",.3~g"),
            alt.Tooltip("share:Q", title="Share", format=".1%"),
            alt.Tooltip("users:Q", title="Users", format=","),
        ]
    )
    return _style(lines + points)


def churn_rate_bars(rates: pd.DataFrame, overall_rate: float) -> alt.TopLevelMixin:
    """Churn rate per quantile bin with the overall rate as a reference line."""
    bars = (
        alt.Chart(rates)
        .mark_bar(color=BLUE, cornerRadiusEnd=4, size=22)
        .encode(
            x=alt.X(
                "bin:N",
                sort=None,
                title="Quantile bin (low → high)",
                axis=alt.Axis(labelAngle=-30, labelLimit=140),
            ),
            y=alt.Y("churn_rate:Q", title="Churn rate", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("bin:N", title="Bin"),
                alt.Tooltip("users:Q", title="Users", format=","),
                alt.Tooltip("churn_rate:Q", title="Churn rate", format=".1%"),
            ],
        )
    )
    ref = pd.DataFrame(
        {"rate": [overall_rate], "label": [f"Overall {overall_rate:.1%}"]}
    )
    rule = (
        alt.Chart(ref).mark_rule(color=INK_SECONDARY, strokeWidth=1).encode(y="rate:Q")
    )
    text = (
        alt.Chart(ref)
        .mark_text(align="right", dy=-6, color=INK_SECONDARY, fontSize=11)
        .encode(y="rate:Q", x=alt.value("width"), text="label:N")
    )
    return _style(bars + rule + text)


def correlation_bars(
    corr: pd.Series, title: str = "Correlation with churn"
) -> alt.TopLevelMixin:
    """Diverging bars of each feature's correlation (red positive, blue negative)."""
    data = corr.rename_axis("feature").reset_index(name="correlation")
    data["direction"] = np.where(data["correlation"] >= 0, "Positive", "Negative")
    limit = max(float(data["correlation"].abs().max()), 0.05)
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, size=12)
        .encode(
            y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X(
                "correlation:Q",
                title=title,
                scale=alt.Scale(domain=[-limit, limit]),
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Positive", "Negative"], range=[RED, BLUE]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("correlation:Q", title="Correlation", format="+.3f"),
            ],
        )
    )
    return _style(chart, height=max(20 * len(data), 200))


def correlation_heatmap(long: pd.DataFrame) -> alt.TopLevelMixin:
    order = list(dict.fromkeys(long["feature_a"]))
    base = alt.Chart(long).encode(
        x=alt.X(
            "feature_a:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelAngle=-45, labelLimit=160),
        ),
        y=alt.Y("feature_b:N", sort=order, title=None, axis=alt.Axis(labelLimit=160)),
    )
    cells = base.mark_rect(stroke="#fcfcfb", strokeWidth=2).encode(
        color=alt.Color(
            "correlation:Q",
            scale=alt.Scale(domain=[-1, 0, 1], range=[BLUE, NEUTRAL, RED]),
            legend=alt.Legend(title="Correlation", orient="right", gradientLength=160),
        ),
        tooltip=[
            alt.Tooltip("feature_a:N", title="Feature A"),
            alt.Tooltip("feature_b:N", title="Feature B"),
            alt.Tooltip("correlation:Q", title="Correlation", format="+.2f"),
        ],
    )
    return _style(cells, height=max(26 * len(order), 260))


def metric_bars(metrics: pd.DataFrame, metric: str, title: str) -> alt.TopLevelMixin:
    """One bar per model for a single metric (``metrics`` has a ``model`` column)."""
    chart = (
        alt.Chart(metrics)
        .mark_bar(cornerRadiusEnd=4, size=16)
        .encode(
            y=alt.Y("model:N", sort=list(MODEL_COLORS), title=None),
            x=alt.X(f"{metric}:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model:N", scale=_scale(MODEL_COLORS), legend=None),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip(f"{metric}:Q", title=title, format=".4f"),
            ],
        )
    )
    labels = chart.mark_text(
        align="left", dx=4, color=INK_SECONDARY, fontSize=11
    ).encode(text=alt.Text(f"{metric}:Q", format=".3f"), color=alt.value(INK_SECONDARY))
    return _style((chart + labels).properties(title=title), height=190)


def threshold_chart(sweep: pd.DataFrame, threshold: float) -> alt.TopLevelMixin:
    """Macro F1 / precision / recall against the decision threshold."""
    long = sweep.rename(
        columns={
            "f1_macro": "Macro F1",
            "precision_macro": "Macro precision",
            "recall_macro": "Macro recall",
        }
    ).melt(id_vars="threshold", value_vars=list(METRIC_COLORS), var_name="metric")
    hover = alt.selection_point(
        fields=["threshold"],
        nearest=True,
        on="pointerover",
        empty=False,
        clear="pointerout",
    )
    base = alt.Chart(long).encode(
        x=alt.X("threshold:Q", title="Decision threshold"),
        y=alt.Y("value:Q", title="Score", scale=alt.Scale(zero=False)),
        color=alt.Color("metric:N", scale=_scale(METRIC_COLORS)),
    )
    lines = base.mark_line(strokeWidth=2)
    points = (
        base.mark_point(filled=True, size=70, stroke="#fcfcfb", strokeWidth=2)
        .encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("threshold:Q", title="Threshold", format=".2f"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Score", format=".3f"),
            ],
        )
        .add_params(hover)
    )
    crosshair = (
        alt.Chart(long)
        .mark_rule(color=BASELINE, strokeWidth=1)
        .encode(x="threshold:Q")
        .transform_filter(hover)
    )
    selected = (
        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
        .mark_rule(color=INK_SECONDARY, strokeWidth=1.5)
        .encode(x="threshold:Q")
    )
    return _style(lines + crosshair + selected + points)


def confusion_chart(tn: int, fp: int, fn: int, tp: int) -> alt.TopLevelMixin:
    data = pd.DataFrame(
        {
            "actual": ["Retained", "Retained", "Churned", "Churned"],
            "predicted": ["Retained", "Churned", "Retained", "Churned"],
            "users": [tn, fp, fn, tp],
        }
    )
    data["share"] = data["users"] / data["users"].sum()
    base = alt.Chart(data).encode(
        x=alt.X(
            "predicted:N",
            sort=["Retained", "Churned"],
            title="Predicted",
            axis=alt.Axis(labelAngle=0, orient="top"),
        ),
        y=alt.Y("actual:N", sort=["Retained", "Churned"], title="Actual"),
    )
    cells = base.mark_rect(stroke="#fcfcfb", strokeWidth=2, cornerRadius=4).encode(
        color=alt.Color("users:Q", scale=alt.Scale(range=SEQUENTIAL[:5]), legend=None),
        tooltip=[
            alt.Tooltip("actual:N", title="Actual"),
            alt.Tooltip("predicted:N", title="Predicted"),
            alt.Tooltip("users:Q", title="Users", format=","),
            alt.Tooltip("share:Q", title="Share", format=".1%"),
        ],
    )
    midpoint = data["users"].max() / 2
    text = base.mark_text(fontSize=16, fontWeight=600).encode(
        text=alt.Text("users:Q", format=","),
        color=alt.condition(
            f"datum.users > {midpoint}", alt.value("#ffffff"), alt.value(INK)
        ),
    )
    return _style(cells + text, height=240)


def roc_chart(roc: pd.DataFrame) -> alt.TopLevelMixin:
    """ROC curves; ``roc`` has columns model, fpr, tpr, auc."""
    roc = roc.assign(
        legend=roc["model"] + " (AUC " + roc["auc"].map("{:.3f}".format) + ")"
    )
    legend_order = list(
        dict.fromkeys(roc.sort_values("model", key=_model_order)["legend"])
    )
    colors = {
        row.legend: MODEL_COLORS[row.model]
        for row in roc.drop_duplicates("model").itertuples()
    }
    diagonal = (
        alt.Chart(pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]}))
        .mark_line(color=BASELINE, strokeWidth=1)
        .encode(x="fpr:Q", y="tpr:Q")
    )
    lines = (
        alt.Chart(roc)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("fpr:Q", title="False positive rate"),
            y=alt.Y("tpr:Q", title="True positive rate"),
            color=alt.Color(
                "legend:N",
                scale=alt.Scale(
                    domain=legend_order, range=[colors[k] for k in legend_order]
                ),
                legend=alt.Legend(orient="bottom", columns=3),
            ),
            order="fpr:Q",
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("fpr:Q", title="FPR", format=".3f"),
                alt.Tooltip("tpr:Q", title="TPR", format=".3f"),
                alt.Tooltip("auc:Q", title="AUC", format=".4f"),
            ],
        )
    )
    return _style(diagonal + lines, height=380)


def _model_order(labels: pd.Series) -> pd.Series:
    order = {label: i for i, label in enumerate(MODEL_COLORS)}
    return labels.map(order)


def probability_histogram(data: pd.DataFrame, threshold: float) -> alt.TopLevelMixin:
    """Share of each group's users by predicted probability, with the threshold.

    ``data`` has a ``probability`` and a ``label`` (Retained / Churned) column.
    """
    chart = (
        alt.Chart(data)
        .transform_bin("bin", "probability", bin=alt.Bin(extent=[0, 1], step=0.025))
        .transform_aggregate(users="count()", groupby=["bin", "label"])
        .transform_joinaggregate(total="sum(users)", groupby=["label"])
        .transform_calculate(share="datum.users / datum.total")
        .mark_line(interpolate="step-after", strokeWidth=2)
        .encode(
            x=alt.X(
                "bin:Q",
                title="Predicted churn probability",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "share:Q", title="Share of group's users", axis=alt.Axis(format="%")
            ),
            color=alt.Color("label:N", scale=_scale(LABEL_COLORS)),
            tooltip=[
                alt.Tooltip("label:N", title="Actual"),
                alt.Tooltip("bin:Q", title="From", format=".3f"),
                alt.Tooltip("users:Q", title="Users", format=","),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
    )
    rule = (
        alt.Chart(
            pd.DataFrame(
                {"threshold": [threshold], "label": [f"Threshold {threshold:.2f}"]}
            )
        )
        .mark_rule(color=INK_SECONDARY, strokeWidth=1.5)
        .encode(x="threshold:Q", tooltip=alt.Tooltip("label:N", title="Decision"))
    )
    return _style(chart + rule, height=280)


def importance_bars(
    importance: pd.DataFrame, column: str, title: str, top_n: int = 15
) -> alt.TopLevelMixin:
    """Horizontal bars of the ``top_n`` largest values of ``column``."""
    data = importance.nlargest(top_n, column)[["feature", column]]
    chart = (
        alt.Chart(data)
        .mark_bar(color=BLUE, cornerRadiusEnd=4, size=14)
        .encode(
            y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X(f"{column}:Q", title=title),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip(f"{column}:Q", title=title, format=",.1f"),
            ],
        )
    )
    return _style(chart, height=max(22 * len(data), 160))


def rate_comparison_chart(
    rates: pd.DataFrame, actual_label: str = "Actual"
) -> alt.TopLevelMixin:
    """Predicted churn rate per model (bars) against the actual rate (rule).

    ``rates`` is one group of ``analysis.predicted_vs_actual``.
    """
    data = rates.assign(
        model=rates["model"].map(MODEL_LABELS),
        actual_label=[f"{actual_label} {r:.1%}" for r in rates["actual_rate"]],
    )
    order = [m for m in MODEL_COLORS if m in set(data["model"])]
    limit = max(float(data["predicted_rate"].max()), float(data["actual_rate"].max()))
    x_max = min(1.0, limit * 1.18)
    x_scale = alt.Scale(domain=[0, x_max], nice=False)
    tooltip = [
        alt.Tooltip("model:N", title="Model"),
        alt.Tooltip("predicted_rate:Q", title="Predicted churn rate", format=".1%"),
        alt.Tooltip("actual_rate:Q", title=f"{actual_label} churn rate", format=".1%"),
        alt.Tooltip("difference:Q", title="Difference", format="+.1%"),
        alt.Tooltip("predicted_churners:Q", title="Users flagged", format=","),
        alt.Tooltip("users:Q", title="Users", format=","),
    ]
    base = alt.Chart(data).encode(y=alt.Y("model:N", sort=order, title=None))
    bars = base.mark_bar(cornerRadiusEnd=4, size=18).encode(
        x=alt.X(
            "predicted_rate:Q",
            title="Churn rate",
            scale=x_scale,
            axis=alt.Axis(format=".0%", tickCount=6),
        ),
        color=alt.Color("model:N", scale=_scale(MODEL_COLORS), legend=None),
        tooltip=tooltip,
    )
    values = base.mark_text(
        align="right", fontSize=11, fontWeight=600, color=INK_SECONDARY
    ).encode(
        x=alt.datum(x_max, scale=x_scale),
        text=alt.Text("predicted_rate:Q", format=".1%"),
    )
    rule = (
        alt.Chart(data)
        .mark_rule(color=INK, strokeWidth=2)
        .encode(x=alt.X("actual_rate:Q", scale=x_scale), tooltip=tooltip[2:3])
    )
    rule_label = (
        alt.Chart(data.head(1))
        .mark_text(align="left", dx=4, dy=-6, fontSize=11, fontWeight=600, color=INK)
        .encode(x="actual_rate:Q", y=alt.value(0), text="actual_label:N")
    )
    chart = (bars + values + rule + rule_label).properties(
        padding={"top": 18, "left": 4, "right": 12, "bottom": 4}
    )
    return _style(chart, height=40 * len(order) + 20)


def vote_bars(votes: pd.DataFrame, n_models: int) -> alt.TopLevelMixin:
    """Users by how many models flag them (ordinal blue ramp)."""
    data = votes.assign(label=[f"{v} of {n_models}" for v in votes["votes"]])
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, size=36)
        .encode(
            x=alt.X(
                "label:N",
                sort=list(data["label"]),
                title="Models flagging the user",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("users:Q", title="Users"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=list(data["label"]), range=VOTE_COLORS[: len(data)]
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Flagged by"),
                alt.Tooltip("users:Q", title="Users", format=","),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
    )
    labels = chart.mark_text(dy=-8, fontSize=11).encode(
        text=alt.Text("share:Q", format=".0%"), color=alt.value(INK_SECONDARY)
    )
    return _style(chart + labels, height=260)


def agreement_heatmap(agreement: pd.DataFrame) -> alt.TopLevelMixin:
    """Share of users on which each pair of models makes the same call."""
    data = agreement.assign(
        model_a=agreement["model_a"].map(MODEL_LABELS),
        model_b=agreement["model_b"].map(MODEL_LABELS),
    )
    order = [m for m in MODEL_COLORS if m in set(data["model_a"])]
    pairs = data[data["model_a"] != data["model_b"]]["agreement"]
    low = float(np.floor((pairs.min() if len(pairs) else 0.5) * 20) / 20)
    middle = (low + 1) / 2
    base = alt.Chart(data).encode(
        x=alt.X("model_a:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("model_b:N", sort=order, title=None),
    )
    cells = base.mark_rect(stroke="#fcfcfb", strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color(
            "agreement:Q",
            scale=alt.Scale(domain=[low, 1], range=[SEQUENTIAL[0], SEQUENTIAL[5]]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("model_a:N", title="Model"),
            alt.Tooltip("model_b:N", title="vs"),
            alt.Tooltip("agreement:Q", title="Same decision", format=".1%"),
            alt.Tooltip("both_flagged:Q", title="Flagged by both", format=","),
        ],
    )
    text = base.mark_text(fontSize=12).encode(
        text=alt.Text("agreement:Q", format=".0%"),
        color=alt.condition(
            f"datum.agreement > {middle}", alt.value("#ffffff"), alt.value(INK)
        ),
    )
    return _style(cells + text, height=260)


def percentile_chart(percentiles: pd.DataFrame) -> alt.TopLevelMixin:
    """Dot plot of a user's percentile for each feature (50 = population median)."""
    base = alt.Chart(percentiles).encode(
        y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
    )
    track = (
        base.mark_rule(color=GRID, strokeWidth=2)
        .encode(
            x=alt.X(
                "zero:Q",
                title="Percentile among training users",
                scale=alt.Scale(domain=[0, 100]),
            ),
            x2="hundred:Q",
        )
        .transform_calculate(zero="0", hundred="100")
    )
    median = (
        alt.Chart(pd.DataFrame({"p": [50]}))
        .mark_rule(color=BASELINE, strokeWidth=1)
        .encode(x="p:Q")
    )
    dots = base.mark_circle(
        size=110, color=BLUE, opacity=1, stroke="#fcfcfb", strokeWidth=2
    ).encode(
        x="percentile:Q",
        tooltip=[
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("user_value:Q", title="This user", format=",.4~g"),
            alt.Tooltip("population_median:Q", title="Median user", format=",.4~g"),
            alt.Tooltip("percentile:Q", title="Percentile", format=".0f"),
        ],
    )
    return _style(track + median + dots, height=max(24 * len(percentiles), 160))
