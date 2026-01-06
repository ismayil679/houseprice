"""
Manual XGBoost Tuning - Round 2 (building on LR=0.02 + Depth=10)
"""

import sys
sys.path.insert(0, 'src')

import os
import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from preprocessing_v3 import HousePricePreprocessorV3, load_data

print("Loading data...")
train_df, _ = load_data()

y_full = train_df['price']
q_low, q_high = y_full.quantile(0.01), y_full.quantile(0.99)
train_df = train_df[(y_full >= q_low) & (y_full <= q_high)].reset_index(drop=True)

preprocessor = HousePricePreprocessorV3()
X_df = preprocessor.extract_features(train_df, target=train_df['price'], is_train=True)
preprocessor.fit(X_df)
X = preprocessor.transform(X_df)
y = np.log1p(train_df['price'].values)

print(f"Ready: {len(y):,} samples\n")

def test_xgb(name, params):
    full_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 5000,
        'early_stopping_rounds': 100,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        **params
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    folds = []
    
    for train_idx, val_idx in kf.split(X):
        model = xgb.XGBRegressor(**full_params)
        model.fit(X[train_idx], y[train_idx], 
                  eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        oof[val_idx] = model.predict(X[val_idx])
        folds.append(np.sqrt(mean_squared_error(np.expm1(y[val_idx]), np.expm1(oof[val_idx]))))
    
    rmse = np.sqrt(mean_squared_error(np.expm1(y), np.expm1(oof)))
    print(f"{name:35} | RMSE: {rmse:,.0f} | Folds: [{', '.join(f'{f:,.0f}' for f in folds)}]")
    return rmse, params

best_rmse = 36026
best_params = {'learning_rate': 0.02, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8}
best_name = "LR=0.02+Depth=10"

def save_if_best(name, rmse, params):
    global best_rmse, best_params, best_name
    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params
        best_name = name
        with open('models/tuning_checkpoints/xgb_manual_best.json', 'w') as f:
            json.dump({'name': name, 'rmse': rmse, 'params': params}, f, indent=2)
        print(f"  ★ NEW BEST! Saved.\n")
    else:
        print()

print("=" * 90)
print("ROUND 2: Building on LR=0.02 + Depth=10 (current best: 36,026)")
print("=" * 90)

# Base config
base = {'learning_rate': 0.02, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8}

# Test 1: Depth=11
rmse, p = test_xgb("Depth=11", {**base, 'max_depth': 11})
save_if_best("Depth=11", rmse, p)

# Test 2: Depth=12
rmse, p = test_xgb("Depth=12", {**base, 'max_depth': 12})
save_if_best("Depth=12", rmse, p)

# Test 3: Add reg_alpha
rmse, p = test_xgb("+ reg_alpha=0.3", {**base, 'reg_alpha': 0.3})
save_if_best("+reg_alpha", rmse, p)

# Test 4: Add reg_lambda
rmse, p = test_xgb("+ reg_lambda=1.5", {**base, 'reg_lambda': 1.5})
save_if_best("+reg_lambda", rmse, p)

# Test 5: subsample=0.85
rmse, p = test_xgb("subsample=0.85", {**base, 'subsample': 0.85})
save_if_best("subsample=0.85", rmse, p)

# Test 6: colsample=0.75
rmse, p = test_xgb("colsample=0.75", {**base, 'colsample_bytree': 0.75})
save_if_best("colsample=0.75", rmse, p)

# Test 7: min_child_weight=3
rmse, p = test_xgb("min_child=3", {**base, 'min_child_weight': 3})
save_if_best("min_child=3", rmse, p)

# Test 8: LR=0.015 + Depth=10
rmse, p = test_xgb("LR=0.015 Depth=10", {'learning_rate': 0.015, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8})
save_if_best("LR=0.015+Depth=10", rmse, p)

# Test 9: LR=0.02 + Depth=10 + all improvements
rmse, p = test_xgb("Combined: sub=0.85 col=0.75", {**base, 'subsample': 0.85, 'colsample_bytree': 0.75})
save_if_best("Combined1", rmse, p)

# Test 10: LR=0.018
rmse, p = test_xgb("LR=0.018 Depth=10", {'learning_rate': 0.018, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8})
save_if_best("LR=0.018", rmse, p)

print("=" * 90)
print(f"BEST: {best_name} with RMSE {best_rmse:,.0f}")
print("=" * 90)
