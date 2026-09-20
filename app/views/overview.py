"""Landing page: what the project does and headline numbers."""

import pandas as pd
import streamlit as st

from churn_prediction.config import BUFFER_DAYS, PREDICTION_WINDOW_DAYS
from churn_prediction.data import churn_rate, feature_columns

from common import MODEL_LABELS, metrics_report, model_params, pct, train_features

st.title("Music-streaming churn prediction")
st.markdown(
    f"Predict which listeners of a music-streaming service will **cancel within the "
    f"next {PREDICTION_WINDOW_DAYS} days**, from their behaviour in the app "
    "(sessions, songs, thumbs up/down, visits to the settings or downgrade pages, …). "
    "A weighted ensemble of four models turns 48 behavioural features into a churn "
    "probability for each user."
)

train = train_features()
report = metrics_report()

cols = st.columns(4)
cols[0].metric("Training users", f"{len(train):,}")
cols[1].metric(f"{PREDICTION_WINDOW_DAYS}-day churn rate", pct(churn_rate(train)))
cols[2].metric("Behavioural features", len(feature_columns(train)))
if report:
    ens = report["models"]["ensemble"]
    cols[3].metric(
        "Ensemble ROC-AUC",
        f"{ens['roc_auc']:.3f}",
        help="Measured on a 20 % hold-out set of training users.",
    )

st.subheader("How it works")
st.graphviz_chart(
    """
    digraph {
      rankdir=LR; bgcolor="transparent";
      node [shape=box style="rounded,filled" fillcolor="#f0efec" color="#c3c2b7"
            fontname="Helvetica" fontsize=11 fontcolor="#0b0b0b"];
      edge [color="#898781" arrowsize=0.7];
      raw [label="Raw event log\\n(one row per click)"];
      prep [label="Preprocessing\\nchurn flags, risk scores,\\nforward-looking label"];
      feat [label="User features\\n48 per user"];
      xgb [label="XGBoost"]; lgb [label="LightGBM"];
      cat [label="CatBoost"]; lr [label="Logistic\\nregression"];
      ens [label="Weighted\\nensemble" fillcolor="#cde2fb" color="#2a78d6"];
      out [label="Churn probability\\n+ decision"];
      raw -> prep -> feat;
      feat -> {xgb lgb cat lr};
      {xgb lgb cat lr} -> ens -> out;
    }
    """,
    width="stretch",
)

left, middle, right = st.columns(3)
with left:
    st.markdown("##### 1 · Label")
    st.markdown(
        f"A user is positive if they cancel within **{PREDICTION_WINDOW_DAYS} days** "
        f"after an observed event. Activity in the last **{BUFFER_DAYS} day** before "
        "cancelling is removed, and so are the last "
        f"{PREDICTION_WINDOW_DAYS} days of users who never cancel, so the model can't "
        "see the answer ahead of time."
    )
with middle:
    st.markdown("##### 2 · Features")
    st.markdown(
        "Each page visit adds to a **churn-risk** or **retention** score, using "
        "weights from a logistic regression. The events are then summarised per user: "
        "session statistics, time between sessions, the last three sessions, "
        "how often key pages were visited, and a few combined features."
    )
with right:
    st.markdown("##### 3 · Models")
    st.markdown(
        "XGBoost, LightGBM, CatBoost and logistic regression were tuned with Optuna "
        "(target: 0.6 × macro-F1 + 0.4 × ROC-AUC). Their probabilities are "
        "averaged with tuned weights, and a tuned threshold turns the average "
        "into a yes/no churn decision."
    )

if report:
    st.subheader("Ensemble configuration")
    weights = pd.DataFrame(
        {
            "Model": [MODEL_LABELS[m] for m in report["weights"]],
            "Weight": list(report["weights"].values()),
            "Hold-out ROC-AUC": [
                report["models"][m]["roc_auc"] for m in report["weights"]
            ],
            "Hold-out macro F1": [
                report["models"][m]["f1_macro"] for m in report["weights"]
            ],
        }
    )
    st.dataframe(
        weights,
        hide_index=True,
        column_config={
            "Weight": st.column_config.ProgressColumn(
                format="%.2f", min_value=0.0, max_value=1.0
            ),
            "Hold-out ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
            "Hold-out macro F1": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        f"Ensemble decision threshold: **{ens['threshold']:.2f}**, "
        f"giving macro F1 {ens['f1_macro']:.3f} on the hold-out set."
    )

with st.expander("Tuned hyper-parameters"):
    st.json(model_params())

st.subheader("Where to next")
c1, c2, c3 = st.columns(3)
with c1:
    st.page_link(
        "views/explore.py", label="Explore users", icon=":material/query_stats:"
    )
    st.caption("Filter users and see how each feature relates to churn.")
with c2:
    st.page_link(
        "views/performance.py", label="Model performance", icon=":material/speed:"
    )
    st.caption("Predicted vs actual churn rate, scores and decision thresholds.")
with c3:
    st.page_link(
        "views/predictions.py",
        label="Test-set predictions",
        icon=":material/online_prediction:",
    )
    st.caption("Compare each model's predicted churn rate and see who gets flagged.")
