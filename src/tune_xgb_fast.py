"""
Fast XGBoost Tuning
Baseline: 36,225 RMSE
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

best_rmse = 36225
best_params = {}

print(f"\nTuning XGBoost (50 trials) | Baseline: {best_rmse:,}")
print("-" * 60)

start = time.time()

for i in range(50):
    trial_start = time.time()
    
    # Search around baseline params
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 5000,
        'early_stopping_rounds': 100,
        'learning_rate': np.random.uniform(0.02, 0.06),
        'max_depth': np.random.randint(6, 12),
        'subsample': np.random.uniform(0.7, 0.95),
        'colsample_bytree': np.random.uniform(0.6, 0.9),
        'reg_alpha': np.random.uniform(0.1, 2.0),
        'reg_lambda': np.random.uniform(1.0, 8.0),
        'min_child_weight': np.random.randint(1, 20),
        'gamma': np.random.uniform(0, 0.5),
    }
    
    try:
        rmse = eval_xgb(params)
        trial_time = time.time() - trial_start
        
        elapsed = time.time() - start
        eta = (elapsed / (i+1)) * (50 - i - 1)
        
        is_best = rmse < best_rmse
        if is_best:
            best_rmse = rmse
            best_params = {k: v for k, v in params.items() if k not in ['objective', 'n_estimators', 'early_stopping_rounds']}
        
        status = "★ BEST!" if is_best else ""
        print(f"Trial {i+1:2d}/50 | RMSE: {rmse:,.0f} | {trial_time:.1f}s | ETA: {timedelta(seconds=int(eta))} {status}")
        
        # Save checkpoint
        with open('models/tuning_checkpoints/xgb_tune.json', 'w') as f:
            json.dump({
                'best_rmse': best_rmse, 
                'best_params': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in best_params.items()},
                'trials': i+1
            }, f, indent=2)
            
    except Exception as e:
        print(f"Trial {i+1:2d}/50 | ERROR: {e}")

print("\n" + "=" * 60)
print(f"Best XGBoost RMSE: {best_rmse:,.0f}")
print(f"Improvement: {36225 - best_rmse:+,.0f}")
print("Best params:")
for k, v in best_params.items():
    print(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")
