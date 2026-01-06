"""
LightGBM Tuning - Tight Search Around Baseline
==============================================
Baseline: 36,186 RMSE (53 features)
Target: ≤ 33,500 RMSE
"""

import sys
sys.path.insert(0, 'src')

import os
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import timedelta
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from preprocessing_v3 import HousePricePreprocessorV3, load_data

# ============================================================
# CONFIGURATION  
# ============================================================
CHECKPOINT_FILE = 'models/tuning_checkpoints/lgb_tune_v2.json'
N_FOLDS = 5
EARLY_STOPPING = 200

os.makedirs('models/tuning_checkpoints', exist_ok=True)

# Baseline that works
BASELINE = {
    'learning_rate': 0.03,
    'num_leaves': 197,
    'max_depth': 10,
    'min_child_samples': 11,
    'subsample': 0.89,
    'colsample_bytree': 0.74,
    'reg_alpha': 0.81,
    'reg_lambda': 4.55,
}


def prepare_data():
    """Prepare data exactly as train_v3.py"""
    train_df, test_df = load_data()
    
    # Outlier removal
    y_full = train_df['price']
    q_low, q_high = y_full.quantile(0.01), y_full.quantile(0.99)
    mask = (y_full >= q_low) & (y_full <= q_high)
    train_df = train_df[mask].reset_index(drop=True)
    
    # Extract features
    preprocessor = HousePricePreprocessorV3()
    X_train = preprocessor.extract_features(train_df, target=train_df['price'], is_train=True)
    preprocessor.fit(X_train)
    X = preprocessor.transform(X_train)
    y = np.log1p(train_df['price'].values)
    
    return X, y, preprocessor.feature_cols


def cv_score(params, X, y, feature_names):
    """5-Fold CV score"""
    full_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'n_jobs': -1,
        **params
    }
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        dtrain = lgb.Dataset(X_tr, y_tr, feature_name=feature_names)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain)
        
        model = lgb.train(
            full_params, dtrain, num_boost_round=10000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)]
        )
        oof[val_idx] = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(np.expm1(y), np.expm1(oof)))
    return rmse


def run_tuning(n_trials=50):
    print("=" * 60)
    print("LIGHTGBM TUNING - 53 FEATURES")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    X, y, feature_names = prepare_data()
    print(f"  Samples: {len(y):,} | Features: {len(feature_names)}")
    
    # Verify baseline
    print("\nVerifying baseline...")
    baseline_rmse = cv_score(BASELINE, X, y, feature_names)
    print(f"  Baseline RMSE: {baseline_rmse:,.0f}")
    
    results = [{'trial': 0, 'rmse': baseline_rmse, 'params': BASELINE.copy(), 'name': 'baseline'}]
    best_rmse = baseline_rmse
    best_params = BASELINE.copy()
    
    # Save
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'best_rmse': best_rmse, 'best_params': best_params, 'results': results}, f, indent=2)
    
    print(f"\nRunning {n_trials} trials...")
    print("-" * 60)
    
    start = time.time()
    
    # Optuna study
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    
    for i in range(n_trials):
        trial = study.ask()
        
        # Tight search around baseline
        params = {
            'learning_rate': trial.suggest_float('lr', 0.02, 0.05),
            'num_leaves': trial.suggest_int('num_leaves', 150, 250),
            'max_depth': trial.suggest_int('max_depth', 8, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 25),
            'subsample': trial.suggest_float('subsample', 0.8, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.85),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.4, 1.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 2.0, 7.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.05),
        }
        
        t0 = time.time()
        rmse = cv_score(params, X, y, feature_names)
        elapsed = time.time() - start
        eta = (elapsed / (i + 1)) * (n_trials - i - 1)
        
        is_best = rmse < best_rmse
        if is_best:
            best_rmse = rmse
            best_params = params.copy()
        
        status = "★ BEST" if is_best else ""
        print(f"  {i+1:2d}/{n_trials} | RMSE: {rmse:,.0f} | Time: {timedelta(seconds=int(elapsed))} | ETA: {timedelta(seconds=int(eta))} {status}")
        
        results.append({'trial': i+1, 'rmse': rmse, 'params': params})
        study.tell(trial, rmse)
        
        # Save checkpoint
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({'best_rmse': best_rmse, 'best_params': best_params, 'results': results}, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"\n  Baseline RMSE: {baseline_rmse:,.0f}")
    print(f"  Best RMSE:     {best_rmse:,.0f}")
    print(f"  Improvement:   {baseline_rmse - best_rmse:+,.0f}")
    
    print("\n  Best Parameters:")
    for k, v in best_params.items():
        baseline_v = BASELINE.get(k, 'N/A')
        if isinstance(v, float):
            print(f"    {k}: {v:.4f} (baseline: {baseline_v})")
        else:
            print(f"    {k}: {v} (baseline: {baseline_v})")
    
    return best_rmse, best_params


if __name__ == '__main__':
    run_tuning(n_trials=50)
