"""Smoke tests of every Streamlit page with ``streamlit.testing.v1.AppTest``.

The app only reads committed files (feature tables, evaluation reports and
submission files), so no model is fitted or loaded here.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st

from churn_prediction import config
from churn_prediction.config import METRICS_PATH, VALIDATION_PREDICTIONS_PATH

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
TIMEOUT = 120

needs_reports = pytest.mark.skipif(
    not (METRICS_PATH.exists() and VALIDATION_PREDICTIONS_PATH.exists()),
    reason="evaluation reports not generated",
)


def run_page(page: str | None = None) -> "AppTest":
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    at.run()
    if page:
        at.switch_page(page)
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def by_label(widgets, label):
    return next(w for w in widgets if w.label == label)


def test_overview_page():
    at = run_page()
    assert at.title[0].value == "Music-streaming churn prediction"
    assert at.metric[0].label == "Training users"
    assert at.metric[0].value == "17,178"
    assert at.metric[1].value == "20.5%"


def test_explore_page_and_filters():
    at = run_page("views/explore.py")
    assert at.title[0].value == "Explore users"
    assert at.metric[0].value == "17,178"

    by_label(at.selectbox, "Subscription").set_value("Paid").run()
    assert not at.exception
    assert int(at.metric[0].value.replace(",", "")) < 17_178

    by_label(at.selectbox, "Outcome").set_value("Churned").run()
    assert not at.exception
    assert at.metric[1].value == "100.0%"
    assert any("Set **Outcome** to *All*" in i.value for i in at.info)


def test_explore_page_empty_selection():
    at = run_page("views/explore.py")
    sessions = by_label(at.slider, "Number of sessions")
    most_sessions = sessions.value[1]  # heaviest users all have old accounts
    sessions.set_value((most_sessions, most_sessions)).run()
    by_label(at.slider, "Account age (days)").set_value((0, 0)).run()
    assert not at.exception
    assert at.warning and "No users match" in at.warning[0].value


@needs_reports
def test_performance_page():
    at = run_page("views/performance.py")
    assert at.title[0].value == "Model performance"
    assert any(s.value == "Predicted vs actual churn rate" for s in at.subheader)

    by_label(at.segmented_control, "Break down by").set_value("Subscription").run()
    assert not at.exception
    assert any("Free users" in m.value for m in at.markdown)

    by_label(at.selectbox, "Model").set_value("catboost").run()
    assert not at.exception
    by_label(at.slider, "Decision threshold").set_value(0.2).run()
    assert not at.exception
    assert at.metric[0].label == "Predicted churn rate"


def test_predictions_page():
    at = run_page("views/predictions.py")
    assert at.title[0].value == "Test-set predictions"
    assert at.metric[0].value == "2,904"
    assert at.metric[1].value == "1,163"  # users flagged in submission_ensemble.csv

    by_label(at.segmented_control, "Break down by").set_value("Subscription").run()
    assert not at.exception
    assert any("Paid users" in m.value for m in at.markdown)


def test_predictions_page_filters():
    at = run_page("views/predictions.py")
    by_label(at.selectbox, "Subscription").set_value("Free").run()
    assert not at.exception
    free_users = int(at.metric[0].value.replace(",", ""))
    assert 0 < free_users < 2_904

    by_label(at.selectbox, "Submission").set_value("logreg").run()
    assert not at.exception
    by_label(at.slider, "Flagged by at least … models").set_value(4).run()
    assert not at.exception
    by_label(at.selectbox, "Ensemble decision").set_value("Not flagged").run()
    assert not at.exception


def test_predictions_page_single_user():
    at = run_page("views/predictions.py")
    user = next(s for s in at.selectbox if s.label.startswith("User"))
    user.select_index(len(user.options) - 1).run()  # least flagged user
    assert not at.exception
    assert any(m.value == "Likely to stay" for m in at.metric)


def test_predictions_page_no_match():
    at = run_page("views/predictions.py")
    at.text_input[0].input("no-such-user").run()
    assert not at.exception
    assert any("No users match" in i.value for i in at.info)


@pytest.fixture
def fresh_cache():
    """The app caches file reads; start and end these tests with an empty cache."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def test_pages_without_reports_or_submissions(monkeypatch, tmp_path, fresh_cache):
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(
        config, "VALIDATION_PREDICTIONS_PATH", tmp_path / "preds.parquet"
    )
    monkeypatch.setattr(config, "SUBMISSIONS_DIR", tmp_path / "submissions")

    at = run_page()
    assert at.title[0].value == "Music-streaming churn prediction"
    at = run_page("views/performance.py")
    assert "No evaluation results found" in at.info[0].value
    at = run_page("views/predictions.py")
    assert "No submission files found" in at.info[0].value


@needs_reports
def test_performance_page_without_notebook_importance(
    monkeypatch, tmp_path, fresh_cache
):
    monkeypatch.setattr(config, "NOTEBOOK_IMPORTANCE_PATH", tmp_path / "missing.csv")
    at = run_page("views/performance.py")
    assert any("notebook_feature_importance.csv" in i.value for i in at.info)


def test_explore_heatmap_needs_two_features():
    at = run_page("views/explore.py")
    by_label(at.multiselect, "Features").set_value(["level"]).run()
    assert not at.exception
    assert any("Pick at least two features" in i.value for i in at.info)


def test_predictions_page_model_flagging_nobody(monkeypatch, tmp_path, fresh_cache):
    submissions = tmp_path / "submissions"
    shutil.copytree(config.SUBMISSIONS_DIR, submissions)
    logreg = submissions / config.SUBMISSION_FILES["logreg"]
    pd.read_csv(logreg).assign(target=0).to_csv(logreg, index=False)
    monkeypatch.setattr(config, "SUBMISSIONS_DIR", submissions)

    at = run_page("views/predictions.py")
    by_label(at.selectbox, "Submission").set_value("logreg").run()
    assert not at.exception
    assert any("same decision" in i.value for i in at.info)
