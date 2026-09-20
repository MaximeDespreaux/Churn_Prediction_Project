"""Test-set predictions, read from the submission files in reports/submissions/."""

import pandas as pd
import streamlit as st

from churn_prediction import analysis, viz
from churn_prediction.config import MODEL_NAMES, USER_COL
from churn_prediction.features import FEATURE_DESCRIPTIONS

from common import (
    ALL_MODELS,
    KEY_FEATURES,
    MODEL_LABELS,
    pct,
    show_chart,
    test_predictions,
    train_features,
    training_churn_rates,
)

SECONDS_PER_DAY = 86_400

st.title("Test-set predictions")

table = test_predictions()
if table is None:
    st.info("No submission files found in `reports/submissions/`.")
    st.stop()

st.markdown(
    f"Churn decisions for the **{len(table):,} test users**, taken from the submission "
    "files of the ensemble and of each model (`reports/submissions/`). The real "
    "outcome of test users is unknown, so each prediction is compared with the "
    "churn rate actually observed among training users."
)

# --- Filter row ------------------------------------------------------------------------
subscription = st.selectbox("Subscription", ["All", "Free", "Paid"], key="pred_level")
view = table if subscription == "All" else table[table["subscription"] == subscription]
truth = training_churn_rates()
truth_group = "All users" if subscription == "All" else subscription

k1, k2, k3, k4 = st.columns(4)
k1.metric("Test users", f"{len(view):,}")
k2.metric(
    "Flagged by the ensemble",
    f"{view['ensemble'].sum():,}",
    f"{pct(view['ensemble'].mean())} vs {pct(truth[truth_group])} actual (training)",
    delta_color="off",
    delta_arrow="off",
)
k3.metric("Flagged by all 4 models", f"{(view['votes'] == 4).sum():,}")
k4.metric("Flagged by no model", f"{(view['votes'] == 0).sum():,}")

# --- Predicted vs actual churn rate ------------------------------------------------------
st.subheader("Predicted vs actual churn rate")
st.caption(
    "Bars: share of test users each submission flags as churners. Black line: share "
    "of training users with the same subscription who actually churned."
)
breakdown = "All users"
if subscription == "All":
    breakdown = st.segmented_control(
        "Break down by",
        ["All users", "Subscription"],
        default="All users",
        key="pred_breakdown",
    )
if breakdown == "Subscription":
    rates = analysis.predicted_vs_actual(
        view, ALL_MODELS, actual=truth, group_col="subscription"
    )
    columns = st.columns(rates["group"].nunique())
    for column, (group, part) in zip(columns, rates.groupby("group"), strict=True):
        with column:
            st.markdown(f"**{group} users** · {part['users'].iloc[0]:,} users")
            show_chart(
                viz.rate_comparison_chart(part, "Actual (training)"),
                part,
                key=f"test_rates_{group}",
            )
else:
    rates = analysis.predicted_vs_actual(view, ALL_MODELS, actual=truth[truth_group])
    rates["group"] = truth_group
    show_chart(
        viz.rate_comparison_chart(rates, "Actual (training)"), rates, key="test_rates"
    )

# --- Agreement ---------------------------------------------------------------------------
st.subheader("Do the models agree?")
left, right = st.columns(2)
with left:
    st.markdown("**How many models flag each user**")
    st.caption("Out of XGBoost, LightGBM, CatBoost and logistic regression.")
    votes = analysis.vote_distribution(view, len(MODEL_NAMES))
    show_chart(viz.vote_bars(votes, len(MODEL_NAMES)), votes, key="votes")
with right:
    st.markdown("**Same decision, pair by pair**")
    st.caption("Share of users on which two submissions make the same call.")
    agreement = analysis.agreement_matrix(view, ALL_MODELS)
    show_chart(viz.agreement_heatmap(agreement), agreement, key="agreement")

# --- Who gets flagged --------------------------------------------------------------------
st.subheader("Who gets flagged?")
features = [c for c in train_features().columns if c in FEATURE_DESCRIPTIONS]
c1, c2, c3 = st.columns([1, 2, 1])
model = c1.selectbox("Submission", ALL_MODELS, format_func=MODEL_LABELS.get)
feature = c2.selectbox(
    "Feature", features, index=features.index("avg_session_gap_hours")
)
clip = c3.toggle("Hide outliers (1st–99th pct.)", value=True, key="pred_clip")
st.caption(FEATURE_DESCRIPTIONS.get(feature, ""))

if view[model].nunique() < 2:
    st.info("Every user in this selection gets the same decision from this model.")
else:
    left, right = st.columns(2)
    with left:
        st.markdown("**Distribution: flagged vs not flagged**")
        dist = analysis.distribution_by_label(
            view, feature, clip=clip, label_col=model, names=analysis.FLAG_NAMES
        )
        show_chart(
            viz.distribution_chart(dist, feature, viz.FLAG_COLORS), dist, key="pdist"
        )
    with right:
        st.markdown("**Correlation of the key features with being flagged**")
        corr = analysis.target_correlations(view, KEY_FEATURES, target=model)
        show_chart(
            viz.correlation_bars(corr, title=f"Correlation with {MODEL_LABELS[model]}"),
            corr.rename_axis("feature").reset_index(),
            key="pcorr",
        )
    st.markdown("**Typical flagged vs not-flagged user**")
    medians = analysis.compare_groups(view, model, KEY_FEATURES)
    st.dataframe(
        medians,
        hide_index=True,
        column_config={
            "feature": "Feature",
            "flagged_median": st.column_config.NumberColumn(
                "Median (flagged)", format="%.4g"
            ),
            "not_flagged_median": st.column_config.NumberColumn(
                "Median (not flagged)", format="%.4g"
            ),
        },
    )

# --- Users table -------------------------------------------------------------------------
st.subheader("Users")
f1, f2, f3 = st.columns([1, 1, 2])
min_votes = f1.slider("Flagged by at least … models", 0, len(MODEL_NAMES), 0)
ensemble_filter = f2.selectbox("Ensemble decision", ["All", "Flagged", "Not flagged"])
search = f3.text_input("Find user id", placeholder="e.g. 1128274")

users = view[view["votes"] >= min_votes]
if ensemble_filter != "All":
    users = users[users["ensemble"] == int(ensemble_filter == "Flagged")]
if search:
    users = users[users[USER_COL].str.contains(search.strip(), regex=False)]
users = users.sort_values(["votes", "ensemble"], ascending=False).assign(
    account_age_days=lambda d: d["timeSinceRegistered"] / SECONDS_PER_DAY
)

shown = [
    USER_COL,
    "subscription",
    *ALL_MODELS,
    "votes",
    "sessionId_count",
    "total_events",
    "avg_session_gap_hours",
    "account_age_days",
]
flag_column = st.column_config.CheckboxColumn
st.dataframe(
    users[shown],
    hide_index=True,
    height=380,
    column_config={
        USER_COL: st.column_config.TextColumn("User"),
        "subscription": "Subscription",
        **{m: flag_column(MODEL_LABELS[m]) for m in ALL_MODELS},
        "votes": st.column_config.NumberColumn(
            "Models flagging", help="Out of the 4 single models"
        ),
        "sessionId_count": "Sessions",
        "total_events": "Events",
        "avg_session_gap_hours": st.column_config.NumberColumn(
            "Avg hours between sessions", format="%.1f"
        ),
        "account_age_days": st.column_config.NumberColumn(
            "Account age (days)", format="%.0f"
        ),
    },
)
st.caption(f"{len(users):,} users shown")
st.download_button(
    "Download these users (CSV)",
    users[shown].to_csv(index=False).encode(),
    file_name="test_predictions.csv",
    mime="text/csv",
    icon=":material/download:",
)

# --- Single user ----------------------------------------------------------------------------
st.subheader("Single user")
if users.empty:
    st.info("No users match the filters above.")
    st.stop()
ranked = users.set_index(USER_COL)
user_id = st.selectbox(
    "User (from the table above, most flagged first)",
    list(ranked.index),
    format_func=lambda u: f"{u} · flagged by {ranked.loc[u, 'votes']} of 4",
)
row = ranked.loc[user_id]

k1, k2, k3 = st.columns(3)
k1.metric(
    "Ensemble decision", "Likely to churn" if row["ensemble"] else "Likely to stay"
)
k2.metric("Single models flagging", f"{row['votes']} of {len(MODEL_NAMES)}")
k3.metric("Subscription", row["subscription"])

badges = st.columns(len(ALL_MODELS))
for column, name in zip(badges, ALL_MODELS, strict=True):
    with column:
        st.caption(MODEL_LABELS[name])
        if row[name]:
            st.badge("Churn", icon=":material/warning:", color="orange")
        else:
            st.badge("Stay", icon=":material/check:", color="blue")

st.markdown("**How this user compares with training users**")
st.caption("Percentile for each key feature (50 = the median user). Hover for values.")
percentiles = analysis.user_percentiles(row, train_features(), KEY_FEATURES)
show_chart(viz.percentile_chart(percentiles), percentiles, key="user_pct")

with st.expander("All features of this user"):
    st.dataframe(
        pd.DataFrame({"value": row[features].astype(float)})
        .rename_axis("feature")
        .reset_index(),
        hide_index=True,
    )
