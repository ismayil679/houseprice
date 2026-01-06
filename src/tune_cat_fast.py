"""
Fast CatBoost Tuning - Start from current params
Baseline: 36,994 RMSE
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
optuna.logging.set_verbosity(optuna.logging.WARNING)

from preprocessing_v3 import HousePricePreprocessorV3, load_data

os.makedirs('models/tuning_checkpoints', exist_ok=True)

# Current CatBoost params (from train_v3.py)
BASELINE_CAT = {
    'iterations': 10000,
    'learning_rate': 0.03,
    'depth': 8,
    'l2_leaf_reg': 5,
}

print("Loading data...")
train_df, test_df = load_data()

# Outlier removal
y_full = train_df['price']
q_low, q_high = y_full.quantile(0.01), y_full.quantile(0.99)
train_df = train_df[(y_full >= q_low) & (y_full <= q_high)].reset_index(drop=True)

# Features
preprocessor = HousePricePreprocessorV3()
X_df = preprocessor.extract_features(train_df, target=train_df['price'], is_train=True)
preprocessor.fit(X_df)
X = preprocessor.transform(X_df)
y = np.log1p(train_df['price'].values)

print(f"Samples: {len(y):,} | Features: {X.shape[1]}")

def eval_cat(params):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    
    for train_idx, val_idx in kf.split(X):
        model = cb.CatBoostRegressor(**params, random_seed=42, verbose=0)
        model.fit(X[train_idx], y[train_idx], 
                  eval_set=(X[val_idx], y[val_idx]), 
                  early_stopping_rounds=200, verbose=0)
        oof[val_idx] = model.predict(X[val_idx])
    
    return np.sqrt(mean_squared_error(np.expm1(y), np.expm1(oof)))

results = []
best_rmse = 36994
best_params = BASELINE_CAT.copy()

print(f"\nTuning CatBoost (50 trials) | Baseline: {best_rmse:,}")
print("-" * 60)

start = time.time()

for i in range(50):
    trial_start = time.time()
    
    params = {
        'iterations': 10000,
        'learning_rate': np.random.uniform(0.02, 0.05),
        'depth': np.random.randint(6, 10),
        'l2_leaf_reg': np.random.uniform(1, 10),
        'bagging_temperature': np.random.uniform(0, 1),
        'random_strength': np.random.uniform(0, 2),
        'border_count': np.random.choice([64, 128, 254]),
        'early_stopping_rounds': 200
    }
    
    try:
        rmse = eval_cat(params)
        trial_time = time.time() - trial_start
        
        elapsed = time.time() - start
        eta = (elapsed / (i+1)) * (50 - i - 1)
        
        is_best = rmse < best_rmse
        if is_best:
            best_rmse = rmse
            best_params = params.copy()
        
        status = "★ BEST!" if is_best else ""
        print(f"Trial {i+1:2d}/50 | RMSE: {rmse:,.0f} | Time: {trial_time:.0f}s | ETA: {timedelta(seconds=int(eta))} {status}")
        
        results.append({'trial': i+1, 'rmse': rmse, 'params': params})
        
        # Save checkpoint
        with open('models/tuning_checkpoints/catboost_tune.json', 'w') as f:
            json.dump({'best_rmse': best_rmse, 'best_params': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in best_params.items()}, 'trials': len(results)}, f, indent=2)
            
    except Exception as e:
        print(f"Trial {i+1:2d}/50 | ERROR: {e}")

print("\n" + "=" * 60)
print(f"Best CatBoost RMSE: {best_rmse:,.0f}")
print(f"Improvement: {36994 - best_rmse:+,.0f}")
print("Best params:", {k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()})
