"""Streamlit entry point.

Run locally with:
    streamlit run app/streamlit_app.py

`streamlit run` puts this folder on sys.path, so the pages can `import common`.
"""

import streamlit as st

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="🎧",
    layout="wide",
    menu_items={
        "About": "10-day churn prediction for a music-streaming service. "
        "Source: https://github.com/MaximeDespreaux/Churn_Prediction_Project",
    },
)

navigation = st.navigation(
    [
        st.Page(
            "views/overview.py", title="Overview", icon=":material/home:", default=True
        ),
        st.Page(
            "views/explore.py", title="Explore users", icon=":material/query_stats:"
        ),
        st.Page(
            "views/performance.py", title="Model performance", icon=":material/speed:"
        ),
        st.Page(
            "views/predictions.py",
            title="Test-set predictions",
            icon=":material/online_prediction:",
        ),
    ]
)

with st.sidebar:
    st.caption(
        "Built with Streamlit · "
        "[Source code](https://github.com/MaximeDespreaux/Churn_Prediction_Project)"
    )

navigation.run()
