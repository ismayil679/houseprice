"""
Manual LightGBM Tuning
Baseline: 36,219 RMSE
Best XGB: 36,026
Target: < 35,000
"""

import sys
sys.path.insert(0, 'src')

import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from preprocessing_v3 import HousePricePreprocessorV3, load_data

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
feature_names = preprocessor.feature_cols

print(f"Samples: {len(y):,} | Features: {X.shape[1]}")

def eval_lgb(params):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    
    full_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'n_jobs': -1,
        **params
    }
    
    for train_idx, val_idx in kf.split(X):
        train_data = lgb.Dataset(X[train_idx], y[train_idx], feature_name=feature_names)
        val_data = lgb.Dataset(X[val_idx], y[val_idx], feature_name=feature_names)
        
        model = lgb.train(full_params, train_data, num_boost_round=10000,
                         valid_sets=[val_data], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[val_idx] = model.predict(X[val_idx])
    
    rmse = np.sqrt(mean_squared_error(np.expm1(y), np.expm1(oof)))
    return rmse

# Baseline params
baseline = {
    'learning_rate': 0.03,
    'num_leaves': 197,
    'max_depth': 10,
    'min_child_samples': 11,
    'subsample': 0.89,
    'colsample_bytree': 0.74,
    'reg_alpha': 0.81,
    'reg_lambda': 4.55,
}

best_rmse = 36219
best_params = baseline.copy()

# Test configs
configs = [
    ("Baseline", baseline),
    ("Lower LR 0.02", {**baseline, 'learning_rate': 0.02}),
    ("Lower LR 0.015", {**baseline, 'learning_rate': 0.015}),
    ("More leaves 255", {**baseline, 'num_leaves': 255}),
    ("Deeper max_depth=12", {**baseline, 'max_depth': 12}),
    ("LR=0.02 + depth=12", {**baseline, 'learning_rate': 0.02, 'max_depth': 12}),
    ("LR=0.02 + leaves=255", {**baseline, 'learning_rate': 0.02, 'num_leaves': 255}),
    ("Less reg alpha=0.3", {**baseline, 'reg_alpha': 0.3}),
    ("More reg lambda=8", {**baseline, 'reg_lambda': 8.0}),
    ("Higher subsample 0.95", {**baseline, 'subsample': 0.95}),
    ("Lower colsample 0.65", {**baseline, 'colsample_bytree': 0.65}),
    ("Combo: LR=0.02, depth=12, leaves=255", {**baseline, 'learning_rate': 0.02, 'max_depth': 12, 'num_leaves': 255}),
]

print(f"\nManual LightGBM Tuning | Target: <35,000")
print("=" * 60)

for name, params in configs:
    rmse = eval_lgb(params)
    
    is_best = rmse < best_rmse
    if is_best:
        best_rmse = rmse
        best_params = params.copy()
        with open('models/tuning_checkpoints/best_lgb_manual.json', 'w') as f:
            json.dump({'rmse': best_rmse, 'params': best_params}, f, indent=2)
    
    status = "★ BEST!" if is_best else ""
    print(f"{name:40} | RMSE: {rmse:,.0f} {status}")

print("\n" + "=" * 60)
print(f"Best LGB RMSE: {best_rmse:,.0f}")
print("Best params:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
