"""
Manual XGBoost Tuning - Test specific configs
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

os.makedirs('models/tuning_checkpoints', exist_ok=True)

print("Loading data...")
train_df, _ = load_data()

# Outlier removal
y_full = train_df['price']
q_low, q_high = y_full.quantile(0.01), y_full.quantile(0.99)
train_df = train_df[(y_full >= q_low) & (y_full <= q_high)].reset_index(drop=True)

preprocessor = HousePricePreprocessorV3()
X_df = preprocessor.extract_features(train_df, target=train_df['price'], is_train=True)
preprocessor.fit(X_df)
X = preprocessor.transform(X_df)
y = np.log1p(train_df['price'].values)

print(f"Ready: {len(y):,} samples, {X.shape[1]} features\n")

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
    print(f"{name:30} | RMSE: {rmse:,.0f} | Folds: [{', '.join(f'{f:,.0f}' for f in folds)}]")
    return rmse, params

# Track best
best_rmse = float('inf')
best_params = {}
best_name = ""

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

print("=" * 80)
print("MANUAL XGB TUNING - Target: < 35,000")
print("=" * 80)

# Test 1: Baseline
rmse, p = test_xgb("Baseline", {'learning_rate': 0.03, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8})
save_if_best("Baseline", rmse, p)

# Test 2: Lower learning rate
rmse, p = test_xgb("LR=0.02", {'learning_rate': 0.02, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8})
save_if_best("LR=0.02", rmse, p)

# Test 3: Higher max_depth
rmse, p = test_xgb("Depth=10", {'learning_rate': 0.03, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8})
save_if_best("Depth=10", rmse, p)

# Test 4: Add regularization
rmse, p = test_xgb("Reg alpha=0.5 lambda=2", {'learning_rate': 0.03, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 2.0})
save_if_best("Reg", rmse, p)

# Test 5: LR=0.02 + Depth=10
rmse, p = test_xgb("LR=0.02 + Depth=10", {'learning_rate': 0.02, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8})
save_if_best("LR=0.02+Depth=10", rmse, p)

# Test 6: More sampling
rmse, p = test_xgb("Subsample=0.85 Col=0.75", {'learning_rate': 0.03, 'max_depth': 8, 'subsample': 0.85, 'colsample_bytree': 0.75})
save_if_best("Sampling", rmse, p)

# Test 7: min_child_weight
rmse, p = test_xgb("min_child=5", {'learning_rate': 0.03, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5})
save_if_best("min_child=5", rmse, p)

# Test 8: Combined best ideas
rmse, p = test_xgb("LR=0.025 Depth=9 Reg", {'learning_rate': 0.025, 'max_depth': 9, 'subsample': 0.85, 'colsample_bytree': 0.75, 'reg_alpha': 0.3, 'reg_lambda': 1.5})
save_if_best("Combined1", rmse, p)

# Test 9: gamma
rmse, p = test_xgb("gamma=0.1", {'learning_rate': 0.03, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1})
save_if_best("gamma", rmse, p)

# Test 10: LR=0.015 (slower, more trees)
rmse, p = test_xgb("LR=0.015", {'learning_rate': 0.015, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8})
save_if_best("LR=0.015", rmse, p)

print("=" * 80)
print(f"BEST: {best_name} with RMSE {best_rmse:,.0f}")
print("=" * 80)
