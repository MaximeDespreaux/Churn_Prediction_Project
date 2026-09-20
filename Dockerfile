# syntax=docker/dockerfile:1

# Churn prediction: one Dockerfile, three images (see compose.yaml for the services).
#
#   app (default)  Streamlit dashboard, no ML libraries
#       docker build --tag <YOUR_DOCKER_USERNAME>/churn-prediction-app .
#       docker run --detach --name churn-app --publish 8501:8501 <YOUR_DOCKER_USERNAME>/churn-prediction-app
#   pipeline       `churn` CLI: build-features, evaluate, train, predict, tune
#       docker build --target pipeline --tag <YOUR_DOCKER_USERNAME>/churn-prediction-pipeline .
#       docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
#           -v "$PWD/reports:/app/reports" <YOUR_DOCKER_USERNAME>/churn-prediction-pipeline evaluate
#   test           test suite and linters
#       docker build --target test --tag churn-prediction-test .
#       docker run --rm churn-prediction-test            # pytest
#
# Dependencies always come from uv.lock. No model is trained at build time.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.15

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# --- base: interpreter, uv and an unprivileged user -------------------------------
FROM python:${PYTHON_VERSION}-slim AS base

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LOCKED=1 \
    PYTHONUNBUFFERED=1 \
    CHURN_PROJECT_ROOT=/app \
    PATH="/app/.venv/bin:$PATH"

LABEL org.opencontainers.image.source="https://github.com/MaximeDespreaux/Churn_Prediction_Project" \
      org.opencontainers.image.authors="Maxime Despreaux" \
      org.opencontainers.image.description="10-day churn prediction for a music-streaming service"

# The virtual environment stays root-owned (read-only at runtime); the user owns
# /app and the folders the pipeline writes to.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
RUN mkdir -p data models reports && chown appuser:appuser . data models reports

# --- pipeline: data pipeline + ML libraries + optuna ------------------------------
FROM base AS pipeline

LABEL org.opencontainers.image.title="churn-prediction-pipeline"

# LightGBM and XGBoost need the OpenMP runtime
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --no-dev --extra ml --extra tune

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --extra ml --extra tune --no-editable

# Defaults baked in; mount ./data, ./models and ./reports to use and keep your files
COPY --chown=appuser:appuser data/processed ./data/processed
COPY --chown=appuser:appuser models/params ./models/params

USER appuser
ENTRYPOINT ["churn"]
CMD ["--help"]

# --- test: everything above + dev tools, tests, app and reports -------------------
FROM pipeline AS test

LABEL org.opencontainers.image.title="churn-prediction-test"

USER root
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --no-editable

COPY --chown=appuser:appuser data/sample ./data/sample
COPY --chown=appuser:appuser reports ./reports
COPY --chown=appuser:appuser .streamlit ./.streamlit
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser tests ./tests

USER appuser
ENTRYPOINT []
CMD ["pytest"]

# --- app (default target): the Streamlit dashboard ----------------------------------
FROM base AS app

LABEL org.opencontainers.image.title="churn-prediction-app"

# Dependencies first so this layer is cached when only the code changes
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --no-dev

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-editable

# Only what the dashboard reads
COPY --chown=appuser:appuser data/processed ./data/processed
COPY --chown=appuser:appuser models/params ./models/params
COPY --chown=appuser:appuser reports ./reports
COPY --chown=appuser:appuser .streamlit ./.streamlit
COPY --chown=appuser:appuser app ./app

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
