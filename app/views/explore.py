"""Interactive exploration of the user-level training features."""

import streamlit as st

from churn_prediction import analysis, viz
from churn_prediction.config import TARGET_COL
from churn_prediction.data import churn_rate, feature_columns, filter_users
from churn_prediction.features import FEATURE_DESCRIPTIONS

from common import KEY_FEATURES, delta_style, pct, show_chart, train_features

SECONDS_PER_DAY = 86_400


st.title("Explore users")
st.markdown(
    "Each row of the training data is one user: 48 behavioural features plus "
    f"`{TARGET_COL}`, which says whether they cancelled in the following 10 days. "
    "Use the filters to pick a group of users. Every chart and table below "
    "shows that group."
)

data = train_features()
features = feature_columns(data)
overall_rate = churn_rate(data)

# --- Filter row (scopes everything below) -------------------------------------------
f1, f2, f3, f4 = st.columns([1, 1, 2, 2])
level = f1.selectbox("Subscription", ["All", "Free", "Paid"])
outcome = f2.selectbox("Outcome", ["All", "Churned", "Retained"])
max_sessions = int(data["sessionId_count"].max())
sessions = f3.slider("Number of sessions", 1, max_sessions, (1, max_sessions))
max_days = int(data["timeSinceRegistered"].max() // SECONDS_PER_DAY) + 1
tenure = f4.slider("Account age (days)", 0, max_days, (0, max_days))

subset = filter_users(
    data,
    level=None if level == "All" else level.lower(),
    churned=None if outcome == "All" else outcome == "Churned",
    ranges={
        "sessionId_count": sessions,
        "timeSinceRegistered": (
            tenure[0] * SECONDS_PER_DAY,
            tenure[1] * SECONDS_PER_DAY,
        ),
    },
)
if subset.empty:
    st.warning("No users match these filters. Widen the ranges to see results.")
    st.stop()

summary = analysis.summarize_users(subset)
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Users",
    f"{summary['users']:,}",
    f"{summary['users'] / len(data):.0%} of all",
    delta_color="off",
    delta_arrow="off",
)
k2.metric(
    "Churn rate",
    pct(summary["churn_rate"]),
    f"{(summary['churn_rate'] - overall_rate) * 100:+.1f} pp vs all users",
    **(delta_style(summary["churn_rate"] - overall_rate) or {"delta_color": "inverse"}),
)
k3.metric("Median sessions", f"{summary['median_sessions']:,.0f}")
k4.metric("Paid subscribers", pct(summary["paid_share"], 0))

both_outcomes = subset[TARGET_COL].nunique() == 2
tab_dist, tab_corr, tab_data = st.tabs(
    ["Feature distributions", "Correlations", "Data"]
)

with tab_dist:
    c1, c2 = st.columns([3, 1])
    feature = c1.selectbox(
        "Feature",
        features,
        index=features.index("avg_session_gap_hours"),
        help="Hover a chart for exact values.",
    )
    clip = c2.toggle("Hide outliers (1st–99th pct.)", value=True)
    st.caption(FEATURE_DESCRIPTIONS.get(feature, ""))

    left, right = st.columns(2)
    with left:
        st.markdown("**Distribution by outcome**")
        st.caption(
            "Share of each group's users in each bin, so both groups are comparable."
        )
        dist = analysis.distribution_by_label(subset, feature, clip=clip)
        show_chart(viz.distribution_chart(dist, feature), dist, key="dist")
    with right:
        st.markdown("**Churn rate across the feature's range**")
        if both_outcomes:
            st.caption("Users grouped into deciles of the feature (low → high).")
            rates = analysis.churn_rate_by_quantile(subset, feature)
            show_chart(
                viz.churn_rate_bars(rates, summary["churn_rate"]),
                rates,
                key="rates",
            )
        else:
            st.info("Set **Outcome** to *All* to compare churn rates.")

with tab_corr:
    if not both_outcomes:
        st.info("Set **Outcome** to *All* to see how features relate to churn.")
    else:
        st.markdown("**Correlation of each feature with churn**")
        st.caption(
            "Pearson correlation with the 10-day churn label. Red means higher values "
            "go with more churn; blue means higher values go with less churn."
        )
        corr = analysis.target_correlations(subset, features)
        show_chart(
            viz.correlation_bars(corr),
            corr.rename_axis("feature").reset_index(),
            key="corr",
        )

    st.markdown("**How features relate to each other**")
    chosen = st.multiselect(
        "Features",
        [*features, TARGET_COL],
        default=[*KEY_FEATURES, TARGET_COL],
        help="Defaults to the features studied in the research notebook.",
    )
    if len(chosen) >= 2:
        long = analysis.correlation_long(subset, chosen)
        left, right = st.columns([3, 2])
        with left:
            show_chart(viz.correlation_heatmap(long), long, key="heatmap")
        with right:
            st.markdown("Most strongly related pairs")
            pairs = analysis.top_correlated_pairs(subset, chosen, n=8)
            st.dataframe(
                pairs,
                hide_index=True,
                column_config={
                    "correlation": st.column_config.NumberColumn(format="%+.2f")
                },
            )
    else:
        st.info("Pick at least two features.")

with tab_data:
    st.dataframe(subset, hide_index=True, height=420)
    st.download_button(
        "Download these users (CSV)",
        subset.to_csv(index=False).encode(),
        file_name="filtered_users.csv",
        mime="text/csv",
        icon=":material/download:",
    )
