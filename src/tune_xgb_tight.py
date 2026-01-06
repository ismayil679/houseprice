"""
Tight XGBoost Tuning - Around baseline params
Baseline: lr=0.03, max_depth=8, subsample=0.8, colsample=0.8
"""

import sys
sys.path.insert(0, 'src')

import os
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import timedelta
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from preprocessing_v3 import HousePricePreprocessorV3, load_data

os.makedirs('models/tuning_checkpoints', exist_ok=True)

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

def eval_xgb(params):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    
    for train_idx, val_idx in kf.split(X):
        model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
        model.fit(X[train_idx], y[train_idx], 
                  eval_set=[(X[val_idx], y[val_idx])], 
                  verbose=False)
        oof[val_idx] = model.predict(X[val_idx])
    
    return np.sqrt(mean_squared_error(np.expm1(y), np.expm1(oof)))

# First verify baseline
print("\nVerifying baseline...")
baseline_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 10000,
    'early_stopping_rounds': 200,
    'learning_rate': 0.03,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
baseline_rmse = eval_xgb(baseline_params)
print(f"Baseline RMSE: {baseline_rmse:,.0f}")

best_rmse = baseline_rmse
best_params = baseline_params.copy()

print(f"\nTight tuning (50 trials)")
print("-" * 60)

start = time.time()

for i in range(50):
    trial_start = time.time()
    
    # TIGHT search around baseline
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 10000,
        'early_stopping_rounds': 200,
        # Tight ranges around baseline values
        'learning_rate': np.random.uniform(0.025, 0.04),  # baseline 0.03
        'max_depth': np.random.choice([7, 8, 9, 10]),     # baseline 8
        'subsample': np.random.uniform(0.75, 0.9),        # baseline 0.8
        'colsample_bytree': np.random.uniform(0.7, 0.85), # baseline 0.8
        'reg_alpha': np.random.uniform(0.0, 1.0),
        'reg_lambda': np.random.uniform(0.5, 3.0),
        'min_child_weight': np.random.choice([1, 3, 5, 7]),
    }
    
    try:
        rmse = eval_xgb(params)
        trial_time = time.time() - trial_start
        
        elapsed = time.time() - start
        eta = (elapsed / (i+1)) * (50 - i - 1)
        
        is_best = rmse < best_rmse
        if is_best:
            best_rmse = rmse
            best_params = params.copy()
        
        status = "★ BEST!" if is_best else ""
        print(f"Trial {i+1:2d}/50 | RMSE: {rmse:,.0f} | {trial_time:.1f}s | ETA: {timedelta(seconds=int(eta))} {status}")
        
        # Save checkpoint
        with open('models/tuning_checkpoints/xgb_tight_tune.json', 'w') as f:
            json.dump({
                'best_rmse': best_rmse, 
                'best_params': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in best_params.items() if k not in ['objective']},
                'trials': i+1
            }, f, indent=2)
            
    except Exception as e:
        print(f"Trial {i+1:2d}/50 | ERROR: {e}")

print("\n" + "=" * 60)
print(f"Best XGBoost RMSE: {best_rmse:,.0f}")
print(f"Improvement: {baseline_rmse - best_rmse:+,.0f}")
print("Best params:")
for k, v in best_params.items():
    if k != 'objective':
        print(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")
