# Churn Prediction Project

[![CI/CD](https://github.com/MaximeDespreaux/Churn_Prediction_Project/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/MaximeDespreaux/Churn_Prediction_Project/actions/workflows/ci-cd.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![uv](https://img.shields.io/badge/deps-uv-6340ac)
![Docker](https://img.shields.io/badge/docker-app%20%7C%20pipeline%20%7C%20test-2496ed)

This repository is the implementation of a **10-day churn prediction model for a music-streaming
service**. From user activity logs, it predicts which users will cancel their subscription within
the next 10 days, using temporal behavioral patterns and a weighted ensemble of XGBoost, LightGBM,
CatBoost and logistic regression.

It comes with a **Streamlit app** to explore the data, the models and their predictions, and
**Docker images** for the app, the data pipeline and the test suite.

## Contents

- [Requirements](#requirements) · [Data](#data) · [Streamlit app](#streamlit-app)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pre-trained Models](#pre-trained-models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing) · [License](#license)

---

## Requirements

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).
Install uv ([instructions](https://docs.astral.sh/uv/getting-started/installation/), e.g.
`pip install uv` or `winget install --id=astral-sh.uv -e`), then:

```setup
uv sync                    # app + data pipeline
uv sync --all-extras       # + model libraries and Optuna (training, tuning, tests)
uv run streamlit run app/streamlit_app.py
```

| Extra / group | Packages |
|---------------|----------|
| `ml` (extra) | xgboost, lightgbm, catboost |
| `tune` (extra) | optuna |
| `dev` (group, installed by default) | pytest, pytest-cov, ruff, black, pre-commit |

Or run everything with Docker. One `Dockerfile` builds three images (`app`, `pipeline`, `test`),
wired together in `compose.yaml`:

```setup
docker compose up --build --detach            # app on http://localhost:8501
docker compose watch                          # app, synced/rebuilt on code changes
docker compose run --rm pipeline <command>    # build-features | evaluate | train | predict | tune
docker compose run --rm tests                 # test suite
docker compose down --volumes
```

Single containers, following the Docker get-started conventions:

```setup
docker build --tag <YOUR_DOCKER_USERNAME>/churn-prediction-app .
docker run --detach --name churn-app --publish 8501:8501 <YOUR_DOCKER_USERNAME>/churn-prediction-app
docker rm --force churn-app

docker build --target pipeline --tag <YOUR_DOCKER_USERNAME>/churn-prediction-pipeline .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" -v "$PWD/reports:/app/reports" \
    <YOUR_DOCKER_USERNAME>/churn-prediction-pipeline evaluate

docker login && docker push <YOUR_DOCKER_USERNAME>/churn-prediction-app

# images published by the CI/CD pipeline
docker pull ghcr.io/maximedespreaux/churn-prediction-app:latest
docker pull ghcr.io/maximedespreaux/churn-prediction-pipeline:latest
```

The [Makefile](Makefile) wraps these commands (`make help` lists every target):

| Target | Runs |
|--------|------|
| `make install` | `uv sync --all-extras` |
| `make app` | the Streamlit app |
| `make test` / `make test-fast` | the test suite with coverage / without the tests that fit models |
| `make check` | every pre-commit hook on all files |
| `make features`, `evaluate`, `train`, `predict`, `tune` | the `churn` pipeline commands |
| `make compose-up` / `make compose-down` | the app with Docker Compose |
| `make docker-build-all` | the app, pipeline and test images |
| `make docker-pipeline CMD=evaluate` / `make docker-test` | a pipeline command / the tests in Docker |

### Data

| Path | Content | Versioned |
|------|---------|-----------|
| `data/raw/train.parquet`, `data/raw/test.parquet` | Raw event logs (~740 MB) | No |
| `data/processed/train_features.parquet` | User-level features + target, 17,178 users | Yes |
| `data/processed/test_features.parquet` | User-level features, 2,904 users | Yes |
| `data/sample/sample_events.parquet` | Raw events of 25 test users (pipeline columns only) | Yes |

Put the raw event logs in `data/raw/` to rebuild the features. The app, the tests and the Docker
images only use the versioned files.

### Streamlit app

Run `uv run streamlit run app/streamlit_app.py` (or `docker compose up`) and open
<http://localhost:8501>.

| Page | Content |
|------|---------|
| **Overview** | Project summary, headline numbers, pipeline diagram, ensemble configuration and tuned hyper-parameters |
| **Explore users** | Training users filtered by subscription, outcome, number of sessions and account age: feature distributions by outcome, churn rate by decile, correlation with churn, correlation heatmap, most correlated pairs, downloadable table |
| **Model performance** | Hold-out evaluation (3,436 users): predicted vs actual churn rate per model and for the ensemble (overall and by subscription), scores, threshold explorer with confusion matrix, probability distributions, ROC curves, feature importance |
| **Test-set predictions** | The submission files (2,904 users): predicted churn rate per model vs the actual churn rate of training users (overall and by subscription), agreement between models, profile of flagged users, filterable/downloadable user table, single-user view |

---

## Data Preprocessing

### Core Transformations
The preprocessing pipeline prepares raw event logs for feature engineering:

1. **Churn flag identification**: Mark cancellation events from authentication status and page visits
2. **Timestamp conversion**: Convert millisecond timestamps to datetime for temporal calculations
3. **Level encoding**: Transform subscription tier (`free` → 0, `paid` → 1) for model compatibility
4. **Temporal feature creation**:
   - `timeInSession`: Elapsed time since the user's first event (engagement duration)
   - `timeSinceRegistered`: Account age in seconds (user lifecycle stage)

### Cumulative Risk Scoring
Page visits are weighted and accumulated to create behavioral risk signals. The coefficients were
computed using logistic regression to test the impact of page visits on churn rate:

```python
CHURN_PAGES = {
    'Cancel': 7, 'Cancellation Confirmation': 7, 'Downgrade': 0.16,
    'Submit Upgrade': 0.14, 'Upgrade': 0.03, 'Settings': 0.02, 'Submit Downgrade': 0.01
}

RETAIN_PAGES = {
    'Save Settings': -0.0007, 'Thumbs Down': -0.016, 'About': -0.054, 'Help': -0.058,
    'Add Friend': -0.158, 'Add to Playlist': -0.383, 'Roll Advert': -0.422,
    'Thumbs Up': -0.716, 'Home': -1.423, 'NextSong': -1.925
}
```

- **`churn_risk_score`**: Cumulative sum of churn-indicating pages (higher = higher risk)
- **`not_churn_score`**: Cumulative sum of retention-indicating pages (lower = lower risk)

These scores capture evolving user sentiment throughout their activity history.

### Forward-Looking Target Creation
**Problem**: Standard churn flags are retrospective (the user has already churned).\
**Solution**: Create forward-looking labels with a prediction window.

- **Prediction Window**: 10 days (label users as positive if they churn within the next 10 days)
- **Buffer Days**: 1 day (exclude the last day before churn to prevent data leakage)
- **Non-churner filtering**: Remove the last 10 days of activity for retained users to balance temporal coverage

This creates a realistic prediction scenario of whether or not a user churns in the next 10 days.

---

## Feature Engineering

Feature engineering transforms raw event logs into **48 user-level behavioral features** across
five categories.

### 1. Basic User Features (4)
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `level` | Subscription tier (0=free, 1=paid) | Payment commitment signals retention likelihood |
| `churn_risk_score` | Cumulative churn page weights | Aggregates negative behavioral signals |
| `not_churn_score` | Cumulative retention page weights | Aggregates positive engagement signals |
| `timeSinceRegistered` | Account age (seconds) | Longer tenure correlates with loyalty |

### 2. Session Aggregate Features (19)
Statistics computed across all user sessions to capture typical behavior patterns:

| Feature Type | Features | Rationale |
|--------------|----------|-----------|
| **Session length** | mean, std, max, min | Items per session indicates engagement depth; variability shows consistency |
| **Session duration** | mean, std, max, min | Time invested per session; higher std suggests erratic usage |
| **Session frequency** | `sessionId_count`, `avg_session_gap_hours`, `std_session_gap_hours` | Usage regularity; gaps reveal disengagement patterns |
| **Risk progression** | `session_churn_risk_*` (mean, std, max, last) | How churn signals evolve; `last` captures current state |
| **Retention signals** | `session_retain_score_*` (mean, std, min, last) | Positive engagement trends; `min` detects disengagement onset |

**Key Insight**: The `_last` features (most recent session values) capture immediate pre-churn behavior.

### 3. Recent Behavior Features (3)
Focused window on the last 3 sessions to detect recent trends:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `last3_avg_length` | Avg items in last 3 sessions | Recent engagement level |
| `last3_avg_duration` | Avg duration of last 3 sessions | Recent time investment |
| `last3_avg_risk` | Avg churn risk in last 3 sessions | Recent sentiment shift |

**Why last 3?** Balances recency (detecting decline) with stability (avoiding single-session noise).

### 4. Page Interaction Features (19)
Quantify specific user actions through key pages:

| Feature Type | Pages Tracked | Rationale |
|--------------|---------------|-----------|
| **Raw counts** | `page_nextsong`, `page_thumbs_up`, `page_thumbs_down`, `page_add_to_playlist`, `page_home`, `page_settings`, `page_downgrade`, `page_upgrade`, `page_roll_advert` | Absolute activity volume per action type |
| **Normalized ratios** | Same pages with `_ratio` suffix | Proportion of total events (e.g., 30% NextSong vs 2% Downgrade) |
| **Total events** | `total_events` | Overall activity level baseline |

**Key Pages:**
- **NextSong**: Core engagement (listening)
- **Thumbs Up/Down**: Active feedback signals
- **Downgrade/Upgrade**: Explicit commitment changes
- **Settings**: Account management (precursor to churn)
- **Roll Advert**: Free-tier friction points

**Why ratios?** Normalize for activity level: a user with 10 downgrades out of 50 events is riskier than 10 out of 5000.

### 5. Derived Composite Features (3)
Engineered signals combining multiple base features:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `risk_per_session` | `churn_risk_score / (sessionId_count + 1)` | Risk intensity per session (high score + few sessions = concentrated negativity) |
| `engagement_decline` | `session_length_mean - last3_avg_length` | Positive values indicate a recent drop in engagement |
| `session_instability` | `session_length_std / (session_length_mean + 1)` | Coefficient of variation (high = erratic behavior, low = consistent) |

**Impact**: `session_instability` and `engagement_decline` rank #12 and #15 in importance, capturing behavioral patterns that raw features miss.

### Design Philosophy
1. **Multi-timescale**: Lifetime aggregates + recent windows detect both stable patterns and decay signals
2. **Normalization**: Ratios account for activity volume differences between heavy and light users
3. **Composite engineering**: Derived features encode domain knowledge (e.g., "risk intensity" > raw risk score)
4. **Temporal focus**: `_last` and `last3_*` features prioritize recent behavior

---

## Training

To build the features, tune the hyper-parameters, train the ensemble and write the submission:

```train
uv run churn build-features                # raw events -> data/processed/*.parquet
uv run churn tune --n-trials 50 --write    # Optuna search -> models/params/*.json
uv run churn train                         # fit the ensemble -> models/churn_models.joblib
uv run churn predict --submission reports/submissions/submission_ensemble.csv
```

Every command accepts `--help`, and runs the same way with `make <command>` or
`docker compose run --rm pipeline <command>`. Random seeds are fixed at 42. Training and tuning need
the `ml` and `tune` extras (`uv sync --all-extras`; already in the pipeline image).

### Model Selection
1. **XGBoost**: Tree-based ensemble with regularization
2. **LightGBM**: Fast gradient boosting with leaf-wise growth
3. **CatBoost**: Handles categorical features natively (though we encode manually)
4. **Logistic Regression**: Linear baseline for comparison

### Hyperparameter Tuning
Bayesian optimization with Optuna's TPE sampler (50-300 trials per model) and 5-fold stratified
cross-validation, maximizing the [combined score](#evaluation-metric).

| Model | Best parameters |
|-------|-----------------|
| XGBoost | `max_depth=9, learning_rate=0.00633, n_estimators=900, scale_pos_weight=4.071, subsample=0.7799, colsample_bytree=0.7988, min_child_weight=1` |
| LightGBM | `max_depth=9, learning_rate=0.01737, n_estimators=1000, num_leaves=57, scale_pos_weight=3.882` |
| CatBoost | `depth=8, learning_rate=0.0249, iterations=1200, scale_pos_weight=4.0436` |
| Logistic Regression | `penalty='l2', C=0.0775, max_iter=588` |
| Ensemble | weights `xgb=0.25, lgb=0.10, cat=0.50, logreg=0.15`, threshold `0.44` |

### Final Model Training
1. **Full dataset retraining**: Retrain on the entire training set with the tuned hyperparameters
2. **Pipeline construction**: `StandardScaler → Model` for each algorithm
3. **Ensemble prediction**: Weighted average of the model probabilities
4. **Threshold application**: Convert probabilities to binary predictions

---

## Evaluation

To evaluate the tuned models on a stratified 80/20 hold-out split:

```eval
uv run churn evaluate    # -> reports/metrics.json, reports/validation_predictions.parquet
```

### Evaluation Metric
**Combined Score** = 0.6 × F1 (macro) + 0.4 × ROC-AUC

- F1 balances precision/recall for imbalanced classes
- ROC-AUC measures ranking quality across thresholds
- The 60/40 weighting prioritizes classification accuracy over probabilistic calibration

---

## Pre-trained Models

The tuned hyper-parameters are versioned as JSON in [`models/params/`](models/params):

| File | Content |
|------|---------|
| `xgboost.json`, `lightgbm.json`, `catboost.json`, `logreg.json` | Tuned hyper-parameters of each model |
| `ensemble.json` | Ensemble weights |
| `thresholds.json` | Decision thresholds (ensemble 0.44, XGBoost 0.50, LightGBM 0.45, CatBoost 0.40, LogReg 0.195) |

`uv run churn train` fits the models from these files into `models/churn_models.joblib`
(not versioned). The test-set predictions are versioned in `reports/submissions/`:

| File | Model | Threshold |
|------|-------|-----------|
| `submission_ensemble.csv` | Weighted ensemble | 0.39 |
| `submission_xgb.csv` | XGBoost | 0.50 |
| `submission_lgb.csv` | LightGBM | 0.45 |
| `submission_cat.csv` | CatBoost | 0.40 |
| `submission_logreg.csv` | Logistic Regression | 0.195 |

---

## Results

### Results Summary
Scores on the 20% stratified hold-out split (3,436 users), produced by
[`uv run churn evaluate`](#evaluation) and saved in [`reports/metrics.json`](reports/metrics.json):

| Metric | Ensemble | XGBoost | LightGBM | CatBoost | LogReg |
|--------|----------|---------|----------|----------|--------|
| Threshold | 0.44 | 0.50 | 0.45 | 0.40 | 0.195 |
| F1 (macro) | **0.730** | 0.728 | 0.726 | 0.705 | 0.614 |
| ROC-AUC | **0.817** | 0.817 | 0.811 | 0.809 | 0.763 |
| Combined | **0.765** | 0.763 | 0.760 | 0.747 | 0.674 |
| Churn recall | 0.597 | 0.541 | 0.592 | 0.656 | 0.760 |
| Churn precision | 0.556 | 0.586 | 0.547 | 0.478 | 0.350 |

**Best model**: weighted ensemble (XGB 0.25 + LGB 0.10 + CAT 0.50 + LogReg 0.15) at threshold 0.44.

The analysis below comes from the research notebook
[`notebooks/churn_research.ipynb`](notebooks/churn_research.ipynb).

### Basic EDA

- **Training data**: User activity logs with session-level and event-level granularity
- **Target**: `will_churn_in_10d`, a binary flag for churn within the 10-day prediction window
- **Churn rate**: 20.5% (3,519 of 17,178 users; class imbalance addressed via `scale_pos_weight`)

**Key findings:**
- Churn risk accumulates over time through cumulative scoring
- Churned users show **higher** `churn_risk_score` (mean: 1.42 vs 0.95)
- Churned users show **lower** `not_churn_score` (mean: -1,591 vs -1,183)
- Churned users have **more** total events (mean: 951 vs 706)
- Session duration and engagement decline precede churn
- Correlation with the target: `churn_risk_score` +0.13, `not_churn_score` -0.10, `session_length_mean` +0.06

### Feature Importance

Top 14 features (ensemble-weighted):

```
Feature                           Importance
─────────────────────────────────────────────
timeSinceRegistered               482.289696
avg_session_gap_hours             446.594877
page_thumbs_up_ratio              389.241942
page_home_ratio                   353.124523
page_thumbs_down_ratio            348.178566
page_roll_advert_ratio            320.990437
page_nextsong_ratio               317.321791
sessionId_count                   309.283015
page_add_to_playlist_ratio        296.059776
std_session_gap_hours             261.242065
page_settings_ratio               236.620717
session_instability               231.697612
session_duration_min              229.061727
page_roll_advert                  195.268107
```

**Key insights:**
1. **Tenure and session rhythm dominate**: the top 2 features are `timeSinceRegistered` and `avg_session_gap_hours`
2. **Behavioral ratios matter**: 7 of the top 14 are normalized page interactions (e.g., `page_nextsong_ratio`), which outperform raw counts
3. **Session patterns count**: `sessionId_count`, `session_instability` and `session_duration_min` are in the top 14
4. **Temporal stability**: `timeSinceRegistered` and `avg_session_gap_hours` provide long-term context

**Model agreement:**
- **High consensus** (top 10 features): `churn_risk_score`, `not_churn_score`, `page_nextsong_ratio`
- **Model-specific strengths**:
  - XGBoost: Emphasizes page-level ratios
  - LightGBM: Prioritizes session duration patterns
  - CatBoost: Balances risk scores and temporal features

The top 10 features account for 47% of the total importance, the top 20 for 68%.

### Feature Correlation with the Target

```
Feature                           Correlation
─────────────────────────────────────────────
page_roll_advert                   0.162531
sessionId_count                    0.104597
page_thumbs_down_ratio             0.092291
session_instability                0.037382
page_roll_advert_ratio             0.028851
page_nextsong_ratio                0.026217
page_add_to_playlist_ratio         0.012024
page_settings_ratio               -0.006888
session_duration_min              -0.016096
page_home_ratio                   -0.033496
timeSinceRegistered               -0.064857
page_thumbs_up_ratio              -0.070442
std_session_gap_hours             -0.225187
avg_session_gap_hours             -0.228363
```

### Multicollinearity Analysis

| Feature A | Feature B | Correlation |
|-----------|-----------|-------------|
| `avg_session_gap_hours` | `std_session_gap_hours` | 0.76 |
| `page_nextsong_ratio` | `page_home_ratio` | -0.76 |
| `sessionId_count` | `page_roll_advert` | 0.54 |
| `session_instability` | `session_duration_min` | -0.43 |

- `avg_session_gap_hours` and `std_session_gap_hours` are highly correlated (0.76).
- `page_nextsong_ratio` and `page_home_ratio` are strongly inversely related (-0.76).
- `sessionId_count` and `page_roll_advert` are moderately correlated (0.54), which is logical: more sessions mean more adverts.

### Key Takeaways
1. **Temporal patterns are critical**: account age and the gaps between sessions are the top predictors and the features most correlated with churn
2. **Normalization matters**: ratio features outperform raw counts
3. **Ensemble beats individuals**: +0.3 macro-F1 points over the best single model (0.730 vs 0.728 for XGBoost)
4. **Threshold choice matters**: the ensemble's macro-F1 is 0.740 at the default 0.5 threshold vs 0.730 at the tuned 0.44
5. **Buffer prevents leakage**: excluding the last day before churn keeps the prediction scenario realistic

---

## Project Structure

```
├── app/                          Streamlit app: streamlit_app.py, common.py, views/
├── src/churn_prediction/         package
│   ├── config.py, data.py        paths and constants; loading, validation, filtering
│   ├── preprocessing.py          churn flags, risk scores, forward-looking target
│   ├── features.py               user-level features
│   ├── modeling.py               models, ensemble, metrics, persistence
│   ├── analysis.py, viz.py       app summaries and charts
│   └── __main__.py               `churn` CLI: build_features, evaluate, train, predict, tune
├── data/
│   ├── raw/                      train.parquet, test.parquet (not versioned)
│   ├── processed/                train_features.parquet, test_features.parquet
│   └── sample/                   sample_events.parquet
├── models/params/                tuned hyper-parameters, ensemble weights, thresholds
├── reports/                      metrics.json, validation_predictions.parquet,
│   │                             notebook_feature_importance.csv
│   └── submissions/              submission_{ensemble,xgb,lgb,cat,logreg}.csv
├── notebooks/churn_research.ipynb
├── tests/                        pytest suite
├── .github/                      workflows/ci-cd.yml, dependabot.yml
├── .streamlit/config.toml
├── Dockerfile                    images: app (default), pipeline, test
├── compose.yaml                  services: app, pipeline, tests
├── LICENSE, Makefile
├── pyproject.toml, uv.lock
└── .dockerignore, .gitattributes, .gitignore, .pre-commit-config.yaml
```

---

## Contributing

Set up the environment and install the pre-commit hooks, then run the checks and the tests before
opening a pull request:

```bash
make install    # uv sync --all-extras
make hooks      # uv run pre-commit install
make check      # all pre-commit hooks: ruff, black, uv.lock sync, file hygiene
make test       # pytest; fails below 100% line and branch coverage
make docker-test
```

| Test file | What it checks |
|-----------|----------------|
| `test_data.py` | Loading and validation of raw logs, feature tables and submissions; user and event filters |
| `test_preprocessing.py` | Churn flags, timestamps, cumulative risk scores, forward-looking target |
| `test_features.py` | Session, page and derived features; output schema; same results as the notebook |
| `test_modeling.py` | Models from `models/params/`, ensemble and thresholds, metrics, save/load |
| `test_pipeline.py` | `build-features`, `evaluate`, `train`, `predict` and `tune` end to end |
| `test_cli.py` | `churn` command dispatch and help |
| `test_analysis.py`, `test_viz.py` | App summaries and chart specifications |
| `test_app.py` | Every Streamlit page, its filters and its fallbacks |

The [CI/CD pipeline](.github/workflows/ci-cd.yml) runs on pull requests, pushes to `main` and `v*` tags:

1. **Checks**: every pre-commit hook (`make check`)
2. **Tests**: `make test` on Python 3.12 and 3.13
3. **Build**: app, pipeline and test images; app health check, `churn --help` in the pipeline image, test suite in the test image
4. **Publish** (`main` and `v*` tags): app and pipeline images to `ghcr.io/maximedespreaux/`, and to Docker Hub when the `DOCKERHUB_USERNAME` variable and `DOCKERHUB_TOKEN` secret are set
5. **Release** (`v*` tags): GitHub Release with generated notes

[Dependabot](.github/dependabot.yml) opens weekly pull requests for the uv dependencies, GitHub
Actions and the Docker base image.

## License

Released under the [MIT License](LICENSE).
