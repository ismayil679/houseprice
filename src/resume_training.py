"""
Resume Training - Continue from checkpoints
Only trains XGBoost final models and generates predictions
"""

import sys
sys.path.insert(0, 'src')

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from preprocessing import HousePricePreprocessor

CHECKPOINT_DIR = 'models/checkpoints'
XGB_SEEDS = [42, 123, 456]
XGB_ITERATIONS = 8000

print("=" * 70)
print("   RESUMING FROM CHECKPOINTS")
print("=" * 70)

# Load checkpoints
print("\n  Loading saved checkpoints...")
oof_data = joblib.load(f"{CHECKPOINT_DIR}/oof_predictions.pkl")
lgb_final_models = joblib.load(f"{CHECKPOINT_DIR}/lgb_final_models.pkl")
cat_final_models = joblib.load(f"{CHECKPOINT_DIR}/cat_final_models.pkl")

print(f"  ✓ OOF predictions loaded")
print(f"  ✓ LGB final models: {len(lgb_final_models)}")
print(f"  ✓ CAT final models: {len(cat_final_models)}")

# Load data
preprocessor = joblib.load('models/fine_tuned_preprocessor.pkl')
train = pd.read_csv('data/binaaz_train.csv')
test = pd.read_csv('data/binaaz_test.csv')

y_full = train['price']
q_low = y_full.quantile(0.01)
q_high = y_full.quantile(0.99)
mask = (y_full >= q_low) & (y_full <= q_high)
train_filtered = train[mask].reset_index(drop=True)

X_train = preprocessor.extract_features(train_filtered)
X_train = preprocessor.transform(X_train)
y_train = train_filtered['price'].values

X_test = preprocessor.extract_features(test)
X_test = preprocessor.transform(X_test)

print(f"  ✓ Data loaded: {len(y_train):,} train, {len(X_test):,} test")

# Load XGBoost params
xgb_study = joblib.load('models/study_xgb_power.pkl')
xgb_params = xgb_study.best_params

# Train XGBoost final models
print("\n" + "=" * 70)
print("TRAINING XGBOOST FINAL MODELS (no early stopping)")
print("=" * 70)

xgb_final_models = []
start_time = datetime.now()

for i, seed in enumerate(XGB_SEEDS):
    print(f"\n  Training XGB seed {i+1}/{len(XGB_SEEDS)} (seed={seed})...")
    
    model_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': XGB_ITERATIONS,
        'learning_rate': xgb_params.get('learning_rate', 0.02),
        'max_depth': xgb_params.get('max_depth', 8),
        'min_child_weight': xgb_params.get('min_child_weight', 10),
        'subsample': xgb_params.get('subsample', 0.8),
        'colsample_bytree': xgb_params.get('colsample_bytree', 0.7),
        'reg_alpha': xgb_params.get('reg_alpha', 0.5),
        'reg_lambda': xgb_params.get('reg_lambda', 2.0),
        'random_state': seed,
        'verbosity': 0
        # NO early_stopping_rounds - training on full data
    }
    
    model = xgb.XGBRegressor(**model_params)
    model.fit(X_train, np.log1p(y_train))
    xgb_final_models.append(model)
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    eta = (elapsed / (i+1)) * (len(XGB_SEEDS) - i - 1)
    print(f"    Done! Elapsed: {elapsed:.1f}min, ETA: {eta:.1f}min")
    
    # Checkpoint
    joblib.dump(xgb_final_models, f"{CHECKPOINT_DIR}/xgb_final_models.pkl")

print(f"\n  ✓ All XGB models trained!")

# Get weights from OOF optimization
weights = {
    'lgb': 0.138,  # From previous run
    'cat': 0.409,
    'xgb': 0.452
}
ensemble_cv = 37169.87

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

# LightGBM predictions
lgb_preds = np.zeros(len(X_test))
for model in lgb_final_models:
    lgb_preds += np.expm1(model.predict(X_test))
lgb_preds /= len(lgb_final_models)
print(f"  LightGBM: mean={lgb_preds.mean():,.0f}")

# CatBoost predictions
cat_preds = np.zeros(len(X_test))
for model in cat_final_models:
    cat_preds += np.expm1(model.predict(X_test))
cat_preds /= len(cat_final_models)
print(f"  CatBoost: mean={cat_preds.mean():,.0f}")

# XGBoost predictions
xgb_preds = np.zeros(len(X_test))
for model in xgb_final_models:
    xgb_preds += np.expm1(model.predict(X_test))
xgb_preds /= len(xgb_final_models)
print(f"  XGBoost:  mean={xgb_preds.mean():,.0f}")

# Weighted ensemble
final_preds = (weights['lgb'] * lgb_preds + 
               weights['cat'] * cat_preds + 
               weights['xgb'] * xgb_preds)

# Post-processing: clip extreme values
p1, p99 = np.percentile(final_preds, [1, 99])
final_preds = np.clip(final_preds, p1 * 0.8, p99 * 1.2)

print(f"\n  Final ensemble: mean={final_preds.mean():,.0f}, std={final_preds.std():,.0f}")

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'price': final_preds
})

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
submission_path = f'outputs/predictions/submission_final_{timestamp}.csv'
submission.to_csv(submission_path, index=False)
submission.to_csv('outputs/predictions/submission_latest.csv', index=False)

# Save final ensemble
final_data = {
    'lgb_models': lgb_final_models,
    'cat_models': cat_final_models,
    'xgb_models': xgb_final_models,
    'weights': weights,
    'ensemble_cv': ensemble_cv,
    'preprocessor': preprocessor
}
joblib.dump(final_data, 'models/final_ensemble.pkl')

print("\n" + "=" * 70)
print("   COMPLETE!")
print("=" * 70)
print(f"  Ensemble CV RMSE: {ensemble_cv:,.2f}")
print(f"  Estimated LB: {ensemble_cv * 0.81:,.0f}")
print(f"  Submission: {submission_path}")
print("=" * 70)
