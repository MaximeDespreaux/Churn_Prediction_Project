"""Hold-out performance: predicted vs actual churn, metrics, thresholds, importance."""

import pandas as pd
import streamlit as st

from churn_prediction import analysis, viz
from churn_prediction.config import TARGET_COL
from churn_prediction.modeling import evaluate_predictions, roc_points, threshold_sweep

from common import (
    MODEL_LABELS,
    delta_style,
    metrics_report,
    notebook_importance,
    pct,
    show_chart,
    validation_predictions,
)

st.title("Model performance")

report = metrics_report()
preds = validation_predictions()
if report is None or preds is None:
    st.info(
        "No evaluation results found. Run `uv run churn evaluate` "
        "to create `reports/metrics.json` and `reports/validation_predictions.parquet`."
    )
    st.stop()

split = report["split"]
models = list(report["models"])
thresholds = {name: m["threshold"] for name, m in report["models"].items()}
st.markdown(
    f"Hold-out evaluation: each model was fit with its saved hyper-parameters on "
    f"**{split['n_train']:,}** training users and scored on "
    f"**{split['n_validation']:,}** held-out users whose real outcome is known "
    f"(stratified {1 - split['validation_size']:.0%}/{split['validation_size']:.0%} "
    f"split, seed {split['random_state']})."
)

# --- Predicted vs actual churn rate --------------------------------------------------------
st.subheader("Predicted vs actual churn rate")
st.caption(
    "Bars: share of held-out users each model flags as churners at its decision "
    "threshold. Black line: share of those users who actually churned."
)
flags = analysis.threshold_flags(preds, thresholds).assign(
    **{TARGET_COL: preds[TARGET_COL], "subscription": preds["subscription"]}
)
breakdown = st.segmented_control(
    "Break down by",
    ["All users", "Subscription"],
    default="All users",
    key="perf_breakdown",
)
if breakdown == "Subscription":
    rates = analysis.predicted_vs_actual(
        flags, models, actual=TARGET_COL, group_col="subscription"
    )
    columns = st.columns(rates["group"].nunique())
    for column, (group, part) in zip(columns, rates.groupby("group"), strict=True):
        with column:
            st.markdown(f"**{group} users** · {part['users'].iloc[0]:,} users")
            show_chart(
                viz.rate_comparison_chart(part, "Actual"), part, key=f"rates_{group}"
            )
else:
    rates = analysis.predicted_vs_actual(flags, models, actual=TARGET_COL)
    show_chart(viz.rate_comparison_chart(rates, "Actual"), rates, key="rates_all")

# --- Metrics -------------------------------------------------------------------------------
st.subheader("Scores")
table = pd.DataFrame(
    [
        {
            "model": MODEL_LABELS[name],
            "threshold": m["threshold"],
            "roc_auc": m["roc_auc"],
            "f1_macro": m["f1_macro"],
            "combined": m["combined"],
            "churn_recall": m["churn_recall"],
            "churn_precision": m["churn_precision"],
            "accuracy": m["accuracy"],
        }
        for name, m in report["models"].items()
    ]
)
number = st.column_config.NumberColumn
st.dataframe(
    table,
    hide_index=True,
    column_config={
        "model": "Model",
        "threshold": number("Threshold", format="%.3f"),
        "roc_auc": number("ROC-AUC", format="%.4f"),
        "f1_macro": number("Macro F1", format="%.4f"),
        "combined": number(
            "Combined", format="%.4f", help="0.6 × macro F1 + 0.4 × ROC-AUC"
        ),
        "churn_recall": number(
            "Churner recall", format="%.3f", help="Share of real churners flagged"
        ),
        "churn_precision": number(
            "Churner precision",
            format="%.3f",
            help="Share of flagged users who churned",
        ),
        "accuracy": number("Accuracy", format="%.3f"),
    },
)
left, right = st.columns(2)
with left:
    st.altair_chart(
        viz.metric_bars(table, "roc_auc", "ROC-AUC"), theme=None, width="stretch"
    )
with right:
    st.altair_chart(
        viz.metric_bars(table, "f1_macro", "Macro F1 at each model's threshold"),
        theme=None,
        width="stretch",
    )

# --- Threshold explorer -------------------------------------------------------------------
st.subheader("Threshold explorer")
st.markdown(
    "Lowering the threshold catches more churners but also flags more users who "
    "would have stayed. Choose a model and a threshold to see the trade-off."
)
c1, c2 = st.columns([1, 2])
model = c1.selectbox("Model", models, format_func=MODEL_LABELS.get)
saved = thresholds[model]
threshold = c2.slider(
    "Decision threshold",
    0.05,
    0.95,
    float(saved),
    0.005,
    format="%.3f",
    help=f"Saved threshold for this model: {saved:.3f}",
)

y_true = preds[TARGET_COL]
proba = preds[model]
current = evaluate_predictions(y_true, proba, threshold)
at_saved = report["models"][model]
flagged = (current["tp"] + current["fp"]) / len(preds)
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Predicted churn rate",
    pct(flagged),
    f"actual {pct(y_true.mean())}",
    delta_color="off",
    delta_arrow="off",
)
k2.metric(
    "Macro F1",
    f"{current['f1_macro']:.3f}",
    f"{current['f1_macro'] - at_saved['f1_macro']:+.3f} vs saved",
    **delta_style(current["f1_macro"] - at_saved["f1_macro"]),
)
k3.metric(
    "Churners caught",
    pct(current["churn_recall"]),
    f"{(current['churn_recall'] - at_saved['churn_recall']) * 100:+.1f} pp",
    **delta_style(current["churn_recall"] - at_saved["churn_recall"]),
)
k4.metric(
    "Flagged users who churn",
    pct(current["churn_precision"]),
    f"{(current['churn_precision'] - at_saved['churn_precision']) * 100:+.1f} pp",
    **delta_style(current["churn_precision"] - at_saved["churn_precision"]),
)


@st.cache_data
def sweep_for(model_name: str) -> pd.DataFrame:
    return threshold_sweep(preds[TARGET_COL], preds[model_name])


left, right = st.columns([3, 2])
with left:
    st.markdown("**Scores by threshold**")
    sweep = sweep_for(model)
    show_chart(viz.threshold_chart(sweep, threshold), sweep, key="sweep")
with right:
    st.markdown("**Confusion matrix**")
    show_chart(
        viz.confusion_chart(current["tn"], current["fp"], current["fn"], current["tp"]),
        pd.DataFrame(
            {
                "actual": ["Retained", "Retained", "Churned", "Churned"],
                "predicted": ["Retained", "Churned", "Retained", "Churned"],
                "users": [current["tn"], current["fp"], current["fn"], current["tp"]],
            }
        ),
        key="confusion",
    )

st.markdown("**Predicted probability by actual outcome**")
hist_data = pd.DataFrame(
    {"probability": proba, "label": y_true.map({0: "Retained", 1: "Churned"})}
)
show_chart(
    viz.probability_histogram(hist_data, threshold),
    hist_data.groupby("label")["probability"].describe().reset_index(),
    key="proba_hist",
)

# --- ROC -------------------------------------------------------------------------------------
st.subheader("ROC curves")
roc = pd.concat(
    [
        roc_points(y_true, preds[name]).assign(
            model=MODEL_LABELS[name], auc=report["models"][name]["roc_auc"]
        )
        for name in models
    ],
    ignore_index=True,
)
show_chart(viz.roc_chart(roc), roc, key="roc")

# --- Feature importance ---------------------------------------------------------------------
st.subheader("Feature importance")
importance = notebook_importance()
if importance is None:
    st.info("`reports/notebook_feature_importance.csv` is missing.")
else:
    st.caption(
        "Top features by ensemble-weighted importance, as reported by the research "
        "notebook (`notebooks/churn_research.ipynb`)."
    )
    top_n = st.slider("Features shown", 5, len(importance), 14)
    show_chart(
        viz.importance_bars(
            importance, "ensemble_importance", "Ensemble-weighted importance", top_n
        ),
        importance.head(top_n),
        key="importance",
    )
