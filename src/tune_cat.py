"""
CatBoost Tuning - Around Baseline
=================================
Current CAT RMSE: 36,994 (worst in ensemble)
Target: Match LGB at ~36,200
"""

import sys
sys.path.insert(0, 'src')

import os
import json
import time
import numpy as np
import pandas as pd
import catboost as cb
from datetime import timedelta
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from preprocessing_v3 import HousePricePreprocessorV3, load_data

CHECKPOINT_FILE = 'models/tuning_checkpoints/cat_tune.json'
N_FOLDS = 5

os.makedirs('models/tuning_checkpoints', exist_ok=True)

# Current baseline
BASELINE = {
    'iterations': 10000,
    'learning_rate': 0.03,
    'depth': 8,
    'l2_leaf_reg': 5,
}


def prepare_data():
    train_df, _ = load_data()
    y_full = train_df['price']
    q_low, q_high = y_full.quantile(0.01), y_full.quantile(0.99)
    mask = (y_full >= q_low) & (y_full <= q_high)
    train_df = train_df[mask].reset_index(drop=True)
    
    preprocessor = HousePricePreprocessorV3()
    X_train = preprocessor.extract_features(train_df, target=train_df['price'], is_train=True)
    preprocessor.fit(X_train)
    X = preprocessor.transform(X_train)
    y = np.log1p(train_df['price'].values)
    
    return X, y


def cv_score(params, X, y):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = cb.CatBoostRegressor(
            iterations=params.get('iterations', 10000),
            learning_rate=params.get('learning_rate', 0.03),
            depth=params.get('depth', 8),
            l2_leaf_reg=params.get('l2_leaf_reg', 5),
            random_seed=42,
            verbose=0,
            early_stopping_rounds=200
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        oof[val_idx] = model.predict(X_val)
    
    return np.sqrt(mean_squared_error(np.expm1(y), np.expm1(oof)))


def run_tuning(n_trials=50):
    print("=" * 60)
    print("CATBOOST TUNING")
    print("=" * 60)
    
    print("\nLoading data...")
    X, y = prepare_data()
    print(f"  Samples: {len(y):,}")
    
    print("\nVerifying baseline...")
    baseline_rmse = cv_score(BASELINE, X, y)
    print(f"  Baseline RMSE: {baseline_rmse:,.0f}")
    
    best_rmse = baseline_rmse
    best_params = BASELINE.copy()
    results = [{'trial': 0, 'rmse': baseline_rmse, 'params': BASELINE}]
    
    print(f"\nRunning {n_trials} trials...")
    print("-" * 60)
    
    start = time.time()
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    
    for i in range(n_trials):
        trial = study.ask()
        
        params = {
            'iterations': 10000,
            'learning_rate': trial.suggest_float('lr', 0.02, 0.05),
            'depth': trial.suggest_int('depth', 6, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temp', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 1),
        }
        
        rmse = cv_score(params, X, y)
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
        
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({'best_rmse': best_rmse, 'best_params': best_params}, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"  Baseline: {baseline_rmse:,.0f} | Best: {best_rmse:,.0f} | Improvement: {baseline_rmse - best_rmse:+,.0f}")
    print("  Best params:", best_params)
    
    return best_rmse, best_params


if __name__ == '__main__':
    run_tuning(n_trials=50)
