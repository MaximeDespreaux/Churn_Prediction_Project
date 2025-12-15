import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score


def combined_score(y_true, y_pred_proba, f1_weight=0.6, auc_weight=0.4):
    """
    Compute weighted combination of F1 (macro) and ROC-AUC.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        f1_weight: Weight for F1 score
        auc_weight: Weight for AUC score
    
    Returns:
        Combined score
    """
    y_pred = (y_pred_proba >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_pred_proba)
    return f1_weight * f1 + auc_weight * auc


def objective_xgb(trial, X_train, y_train, scale_pos_weight):
    """Optuna objective for XGBoost hyperparameter tuning."""
    import xgboost as xgb
    
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.005, 0.03, log=True
        ),
        'n_estimators': trial.suggest_int('n_estimators', 700, 1500, step=100),
        'scale_pos_weight': trial.suggest_float(
            'scale_pos_weight', scale_pos_weight, scale_pos_weight * 1.5
        ),
        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 3),
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    model = xgb.XGBClassifier(**params)
    
    f1_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1
    )
    auc_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1
    )
    
    combined = 0.6 * f1_scores.mean() + 0.4 * auc_scores.mean()
    
    trial.set_user_attr('f1_mean', f1_scores.mean())
    trial.set_user_attr('auc_mean', auc_scores.mean())
    
    return combined


def objective_lgb(trial, X_train, y_train, scale_pos_weight):
    """Optuna objective for LightGBM hyperparameter tuning."""
    import lightgbm as lgb
    
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.005, 0.03, log=True
        ),
        'n_estimators': trial.suggest_int('n_estimators', 700, 1500, step=100),
        'scale_pos_weight': trial.suggest_float(
            'scale_pos_weight', scale_pos_weight, scale_pos_weight * 1.5
        ),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    
    f1_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1
    )
    auc_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1
    )
    
    combined = 0.6 * f1_scores.mean() + 0.4 * auc_scores.mean()
    
    trial.set_user_attr('f1_mean', f1_scores.mean())
    trial.set_user_attr('auc_mean', auc_scores.mean())
    
    return combined


def objective_cat(trial, X_train, y_train, scale_pos_weight):
    """Optuna objective for CatBoost hyperparameter tuning."""
    from catboost import CatBoostClassifier
    
    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.005, 0.03, log=True
        ),
        'iterations': trial.suggest_int('iterations', 700, 1500, step=100),
        'scale_pos_weight': trial.suggest_float(
            'scale_pos_weight', scale_pos_weight, scale_pos_weight * 1.5
        ),
        'random_state': 42,
        'verbose': 0
    }
    
    model = CatBoostClassifier(**params)
    
    f1_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1
    )
    auc_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1
    )
    
    combined = 0.6 * f1_scores.mean() + 0.4 * auc_scores.mean()
    
    trial.set_user_attr('f1_mean', f1_scores.mean())
    trial.set_user_attr('auc_mean', auc_scores.mean())
    
    return combined


def optimize_ensemble_weights(xgb_proba, lgb_proba, cat_proba, y_val):
    """
    Exhaustive grid search for optimal ensemble weights and threshold.
    
    Args:
        xgb_proba, lgb_proba, cat_proba: Model predictions on validation set
        y_val: True validation labels
    
    Returns:
        dict: Best configuration with weights, threshold, and scores
    """
    best_f1 = 0
    best_auc = 0
    best_combined = 0
    best_config = None
    
    for xgb_w in np.arange(0.1, 0.7, 0.05):
        for lgb_w in np.arange(0.1, 0.7, 0.05):
            cat_w = 1 - xgb_w - lgb_w
            if cat_w < 0.1 or cat_w > 0.7:
                continue
            
            ens_proba = xgb_w * xgb_proba + lgb_w * lgb_proba + cat_w * cat_proba
            ens_auc = roc_auc_score(y_val, ens_proba)
            
            for t in np.arange(0.05, 0.8, 0.005):
                f1 = f1_score(
                    y_val, (ens_proba >= t).astype(int), average='macro'
                )
                combined = 0.6 * f1 + 0.4 * ens_auc
                
                if combined > best_combined:
                    best_combined = combined
                    best_f1 = f1
                    best_auc = ens_auc
                    best_config = {
                        'xgb': xgb_w,
                        'lgb': lgb_w,
                        'cat': cat_w,
                        'threshold': t
                    }
    
    best_config['f1'] = best_f1
    best_config['auc'] = best_auc
    best_config['combined'] = best_combined
    
    return best_config


def create_submission(test_user_ids, predictions, example_path, output_path):
    """
    Create submission file with correct format.
    
    Args:
        test_user_ids: User IDs from test set
        predictions: Binary predictions
        example_path: Path to example submission file
        output_path: Where to save submission
    """
    import pandas as pd
    
    example = pd.read_csv(example_path)
    submission = pd.DataFrame({'id': example['id']})
    
    submission = submission.merge(
        pd.DataFrame({
            'id': test_user_ids.astype(example['id'].dtype),
            'target': predictions
        }),
        on='id',
        how='left'
    )
    
    submission['target'] = submission['target'].fillna(0).astype(int)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")