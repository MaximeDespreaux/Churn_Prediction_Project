# Churn Prediction Project

# Features Overview

We are given the following features for churn prediction:

## User Information
- **status**: Logged In or Cancelled  
- **gender**: M or F — not significant for churn prediction  
- **firstName**: Not significant for churn prediction  
- **lastName**: Not significant for churn prediction  
- **userId**: Hash identifier for user  
- **registration**: Registration timestamp

## Session and Activity
- **ts**: Timestamp of action (ms)  
- **auth**: Authentication status  
- **page**: Page visited  
- **sessionId**: Session identifier  
- **itemInSession**: Item number in session  
- **method**: GET or PUT SQL queries  
- **time**: Time spent on action  

## Content and Interaction
- **level**: Free or Paid  
- **location**: City and State of user  
- **userAgent**: Browser and device info  
- **length**: Duration of song  
- **song**: Name of song  
- **artist**: Name of artist


We will remove features with little predictive power (high p-value or many degrees of freedom). For example, location binned by state has p < 0.05, but the large DoF and missing values in the test set reduce its usefulness. Rather than dropping rows, we drop the column; further testing confirms its minimal predictive impact.

## Features kept (FOR NOW):
- **userId**: Useful to know who we are checking at time ts
- **level**: Very important metric for commitment. Transform into binary flag 0:free, 1:paid
- **itemInSession**: Action number in session (bigger => more involved)
- **not_churn_score**:  Negative value, smaller value implies lower churn risk
- **churn_risk_score**: Positive value, bigger value implies higher churn risk
- **registration**: When account was created
- **timeInSession**: How long session lasted (replaces **ts** and **time** features)
- **status**: Has 2 degrees of freedom and $\text{p-value} \approx 2.31\times 10^{-86}$

## Target Variable:
- **churn_flag**: 0 if not yet churned and 1 if user churned

Could be done over different prediction horizons with rolling window, e.g., churn_ts+h for h in days

## NOTE: bigger horizon <==> more uncertainty
