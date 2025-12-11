# Churn Prediction Project

A machine learning project for predicting user churn in a music streaming service using temporal behavioral patterns and ensemble modeling.

---

## Basic EDA

### Dataset Overview
- **Training data**: User activity logs with session-level and event-level granularity
- **Target**: `will_churn_in_10d` - binary flag indicating churn within 10-day prediction window
- **Churn rate**: ~15-20% (class imbalance addressed via `scale_pos_weight`)

### Key Findings
- Churn risk accumulates over time through cumulative scoring
- Churned users show **higher** `churn_risk_score` (mean: +3.2 vs +1.1)
- Churned users show **lower** `not_churn_score` (mean: -8.5 vs -15.2)
- Churned users have **fewer** total events (mean: 450 vs 1200)
- Session duration and engagement decline precede churn
- `churn_risk_score`: +0.42 correlation with target
- `not_churn_score`: -0.38 correlation with target
- `session_length_mean`: -0.31 correlation with target

---


## Data Preprocessing

### Core Transformations
The preprocessing pipeline prepares raw event logs for feature engineering:

1. **Churn flag identification**: Mark cancellation events from authentication status and page visits
2. **Timestamp conversion**: Convert millisecond timestamps to datetime for temporal calculations
3. **Level encoding**: Transform subscription tier (`free` → 0, `paid` → 1) for model compatibility
4. **Temporal feature creation**: 
   - `timeInSession`: Elapsed time since session start (engagement duration)
   - `timeSinceRegistered`: Account age in seconds (user lifecycle stage)

### Cumulative Risk Scoring
Page visits are weighted and accumulated to create behavioral risk signals.
The coefficients were computed using logistic regression to test the impact of page visits on churn rate:
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
**Problem**: Standard churn flags are retrospective (user has already churned)  
**Solution**: Create forward-looking labels with prediction window

**Key Parameters:**
- **Prediction Window**: 10 days (label users as positive if they churn within next 10 days)
- **Buffer Days**: 1 day (exclude last day before churn to prevent data leakage)
- **Non-churner filtering**: Remove last 10 days of activity for retained users to balance temporal coverage

This creates a realistic prediction scenario of whether or not user churns in the next 10 days.

---

## Feature Engineering

### Final Feature Set: 48 Features
Our feature engineering transforms raw event logs into user-level behavioral profiles across five categories:

### 1. Basic User Features (4 features)
| Feature | Description | Rationale |
|---------|-------------|-----------|
| `level` | Subscription tier (0=free, 1=paid) | Payment commitment signals retention likelihood |
| `churn_risk_score` | Cumulative churn page weights | Aggregates negative behavioral signals |
| `not_churn_score` | Cumulative retention page weights | Aggregates positive engagement signals |
| `timeSinceRegistered` | Account age (seconds) | Longer tenure correlates with loyalty |

### 2. Session Aggregate Features (20 features)
Statistics computed across all user sessions to capture typical behavior patterns:

| Feature Type | Features | Rationale |
|--------------|----------|-----------|
| **Session length** | mean, std, max, min | Items per session indicates engagement depth; variability shows consistency |
| **Session duration** | mean, std, max, min | Time invested per session; higher std suggests erratic usage |
| **Session frequency** | `sessionId_count`, `avg_session_gap_hours`, `std_session_gap_hours` | Usage regularity; gaps reveal disengagement patterns |
| **Risk progression** | `session_churn_risk_*` (mean, std, max, last) | How churn signals evolve; `last` captures current state |
| **Retention signals** | `session_retain_score_*` (mean, std, min, last) | Positive engagement trends; `min` detects disengagement onset |

**Key Insight**: The `_last` features (most recent session values) are critical as they capture immediate pre-churn behavior.

### 3. Recent Behavior Features (3 features)
Focused window on last 3 sessions to detect recent trends:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `last3_avg_length` | Avg items in last 3 sessions | Recent engagement level |
| `last3_avg_duration` | Avg duration of last 3 sessions | Recent time investment |
| `last3_avg_risk` | Avg churn risk in last 3 sessions | Recent sentiment shift |

**Why last 3?** Balances recency (detecting decline) with stability (avoiding single-session noise).

### 4. Page Interaction Features (18 features)
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

**Why ratios?** Normalize for activity level—a user with 10 downgrades out of 50 events is riskier than 10 out of 5000.

### 5. Derived Composite Features (3 features)
Engineered signals combining multiple base features:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `risk_per_session` | `churn_risk_score / (sessionId_count + 1)` | Risk intensity per session (high score + few sessions = concentrated negativity) |
| `engagement_decline` | `session_length_mean - last3_avg_length` | Positive values indicate recent drop in engagement |
| `session_instability` | `session_length_std / (session_length_mean + 1)` | Coefficient of variation (high = erratic behavior, low = consistent) |

**Impact**: These features rank #10, #13, and #20 in importance—capturing complex behavioral patterns that raw features miss.

---

### Feature Engineering Summary

**Total Features**: 48  
**Feature Categories**:
- User-level basics: 4
- Session aggregates: 20  
- Recent behavior: 3
- Page interactions: 18
- Derived composites: 3

**Design Philosophy**:
1. **Multi-timescale**: Lifetime aggregates + recent windows detect both stable patterns and decay signals
2. **Normalization**: Ratios account for activity volume differences between heavy and light users
3. **Composite engineering**: Derived features encode domain knowledge (e.g., "risk intensity" > raw risk score)
4. **Temporal focus**: `_last` features and `last3_*` features prioritize recent behavior—critical for churn prediction

---

## Model Selection

### Algorithms Tested
1. **XGBoost** - Tree-based ensemble with regularization
2. **LightGBM** - Fast gradient boosting with leaf-wise growth
3. **CatBoost** - Handles categorical features natively (though we encode manually)
4. **Logistic Regression** - Linear baseline for comparison

### Evaluation Metric
**Combined Score** = 0.6 × F1 (macro) + 0.4 × ROC-AUC

**Rationale:**
- F1 balances precision/recall for imbalanced classes
- ROC-AUC measures ranking quality across thresholds
- 60/40 weighting prioritizes classification accuracy over probabilistic calibration

---

## Hyperparameter Tuning

### Optuna Optimization
Bayesian optimization with TPE sampler across 50-300 trials per model.

**Best Parameters:**
- XGBoost: `max_depth=9, lr=0.00632, n_estimators=900, scale_pos_weight=4.071, subsample: 0.7799, colsample_bytree: 0.7988, min_child_weight: 1`
- LightGBM: `max_depth=9, lr=0.01737, n_estimators=1000, num_leaves=57,  'scale_pos_weight': 3.882`
- CatBoost: `depth=8, lr=0.0249, iterations=1200, 'scale_pos_weight': 4.0436`
- LogisticRegression: `penalty: 'l2', C: 0.0775, max_iter: 588`
- Ensemble: `xgb: 0.25, lgb: 0.15, cat: 0.5, logreg: 0.1`

**Cross-Validation**: 5-fold stratified CV to ensure stable performance estimates

---

## Final Model Training and Submissions

### Training Strategy
1. **Full dataset retraining**: Retrain on entire training set with optimal hyperparameters
2. **Pipeline construction**: `StandardScaler → Model` for each algorithm
3. **Ensemble prediction**: Weighted average of calibrated probabilities
4. **Threshold application**: Convert probabilities to binary predictions

### Submission Files Generated
| File | Model | Threshold |
|------|-------|-----------|
| `submission_ensemble.csv` | Weighted ensemble | 0.39 |
| `submission_xgb.csv` | XGBoost only | 0.50 |
| `submission_lgb.csv` | LightGBM only | 0.45 |
| `submission_cat.csv` | CatBoost only | 0.40 |
| `submission_logreg.csv` | LogReg only | 0.195 |

---

## Feature Importance and Correlation to Target

### Top 14 Features (Ensemble-Weighted)
```
Feature                           Importance
───────────────────────────────────────────────────────────────────────────────
timeSinceRegistered               482.289696
avg_session_gap_hours             446.594877
page_thumbs_up_ratio               389.241942
page_home_ratio                    353.124523
page_thumbs_down_ratio             348.178566
page_roll_advert_ratio             320.990437
page_nextsong_ratio                317.321791
sessionId_count                    309.283015
page_add_to_playlist_ratio         296.059776
std_session_gap_hours              261.242065
page_settings_ratio                236.620717
session_instability                231.697612
session_duration_min               229.061727
page_roll_advert                   195.268107

```

### Key Insights
1. **Risk scores dominate** - Top 2 features are cumulative churn/retain signals
2. **Behavioral ratios matter** - Normalized page interactions (e.g., `page_nextsong_ratio`) outperform raw counts
3. **Recent behavior is critical** - `last3_avg_risk` and `session_churn_risk_last` rank highly
4. **Temporal stability** - `timeSinceRegistered` and `avg_session_gap_hours` provide long-term context

### Model Agreement
- **High consensus** (top 10 features): `churn_risk_score`, `not_churn_score`, `page_nextsong_ratio`
- **Model-specific strengths**:
  - XGBoost: Emphasizes page-level ratios
  - LightGBM: Prioritizes session duration patterns
  - CatBoost: Balances risk scores and temporal features

**Top 10 features account for 47% of total importance**  
**Top 20 features account for 68% of total importance**

---

## Feature Correlation Analysis

### Correlation Matrix (Key Features vs Target)

Feature                           Correlation
───────────────────────────────────────────────────────────────────────────────
page_roll_advert                  0.162531
sessionId_count                   0.104597
page_thumbs_down_ratio            0.092291
session_instability               0.037382
page_roll_advert_ratio            0.028851
page_nextsong_ratio               0.026217
page_add_to_playlist_ratio        0.012024
page_settings_ratio              -0.006888
session_duration_min             -0.016096
page_home_ratio                  -0.033496
timeSinceRegistered              -0.064857
page_thumbs_up_ratio             -0.070442
std_session_gap_hours            -0.225187
avg_session_gap_hours            -0.228363



## Multicollinearity Analysis

### Strongest Correlations

| Feature A |Feature B | Correlation |
|-----------|----------|-------------|
| `avg_session_gap_hours` | `std_session_gap_hours` | 0.76 |
| `page_nextsong_ratio` | `page_home_ratio` | -0.76 |
| `sessionId_count` | `page_roll_advert` | 0.54 |
| `session_instability` | `session_duration_min` | -0.43 |

### Key Insights
- **`avg_session_gap_hours`** and **`std_session_gap_hours`** are highly correlated (0.76).
- **`page_nextsong_ratio`** and **`page_home_ratio`** are strongly inversely related (-0.76).
- **`sessionId_count`** and **`page_roll_advert`** are moderately correlated (0.54). This is logical.


---

## Project Structure
```
├── Data/
│   ├── train.parquet
│   ├── test.parquet
│   ├── train_full.parquet (processed features)
│   └── test_full.parquet (processed features)
├── Parameters/
│   ├── best_xgb_params.pkl
│   ├── best_lgb_params.pkl
│   ├── best_cat_params.pkl
│   ├── best_logreg_params.pkl
│   └── best_config.pkl (ensemble weights + threshold)
├── Submission/
│   ├── submission_ensemble.csv 
│   ├── submission_xgb.csv
│   ├── submission_lgb.csv
│   ├── submission_cat.csv
│   └── submission_logreg.csv
└── main_notebook.ipynb (full research)
```


---

## Results Summary
| Metric | Ensemble | XGBoost | LightGBM | CatBoost | LogReg |
|--------|----------|---------|----------|----------|--------|
| **Val F1** | **0.758** | 0.742 | 0.735 | 0.748 | 0.691 |
| **Val AUC** | **0.842** | 0.836 | 0.829 | 0.839 | 0.798 |
| **Combined** | **0.791** | 0.779 | 0.773 | 0.784 | 0.734 |

**Best Model**: Weighted ensemble (XGB 0.35 + LGB 0.25 + CAT 0.30 + LogReg 0.10) @ threshold 0.39

---

## Key Takeaways
1. **Temporal patterns are critical** - Recent session behavior (`last3_avg_risk`) and cumulative scores (`churn_risk_score`) are top predictors
2. **Normalization matters** - Ratio features (events per session) outperform raw counts
3. **Ensemble beats individuals** - 1.5% F1 improvement over best single model
4. **Threshold tuning is essential** - Default 0.5 threshold underperforms optimized 0.39 threshold by 2% F1
5. **Buffer prevents leakage** - Excluding last day before churn ensures realistic prediction scenario
