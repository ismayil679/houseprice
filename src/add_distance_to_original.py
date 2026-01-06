"""
Add Distance Features to Original Working Pipeline
===================================================
Previous best: LB 29,318 (CV 37,170) with 15 features
Goal: Add landmark features WITHOUT breaking what works

Strategy:
1. Use the EXACT same preprocessing as before
2. ONLY add: num_landmarks_2km, min_landmark_dist
3. Re-tune with new features
"""

import sys
sys.path.insert(0, 'src')

import os
import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from preprocessing import HousePricePreprocessor

# ============================================================
# CONFIGURATION
# ============================================================
N_FOLDS = 5
TUNING_TRIALS = 15
EARLY_STOPPING = 50

# ============================================================
# DISTANCE FEATURES (ONLY NEW STUFF)
# ============================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def compute_landmark_features(df):
    """Add landmark distance features to dataframe."""
    landmarks = pd.read_excel('baku_coordinates.xlsx')
    
    # Clip coordinates to Baku area
    lat = df['latitude'].clip(39.5, 41.5).values
    lon = df['longitude'].clip(48.5, 51.0).values
    
    lm_lats = landmarks['Latitude'].values
    lm_lons = landmarks['Longitude'].values
    
    n = len(df)
    num_within_2km = np.zeros(n)
    min_distance = np.zeros(n)
    
    batch_size = 5000
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_lats = lat[start:end]
        batch_lons = lon[start:end]
        
        dist_matrix = np.zeros((end - start, len(lm_lats)))
        for j in range(len(lm_lats)):
            dist_matrix[:, j] = haversine_distance(
                batch_lats, batch_lons, lm_lats[j], lm_lons[j]
            )
        
        num_within_2km[start:end] = (dist_matrix <= 2.0).sum(axis=1)
        min_distance[start:end] = dist_matrix.min(axis=1)
    
    return num_within_2km, min_distance


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# ============================================================
# LOAD DATA USING ORIGINAL PREPROCESSING
# ============================================================
def load_data():
    """Load data using the ORIGINAL working preprocessor, then add new features."""
    print("=" * 70)
    print("LOADING DATA (ORIGINAL PREPROCESSING + NEW DISTANCE FEATURES)")
    print("=" * 70)
    
    # Original preprocessing
    preprocessor = HousePricePreprocessor()
    
    train = pd.read_csv('data/binaaz_train.csv')
    test = pd.read_csv('data/binaaz_test.csv')
    
    # Extract original features
    X_train_df = preprocessor.extract_features(train)
    X_test_df = preprocessor.extract_features(test)
    
    # Remove outliers (same as original)
    y_full = train['price']
    q_low = y_full.quantile(0.01)
    q_high = y_full.quantile(0.99)
    mask = (y_full >= q_low) & (y_full <= q_high)
    
    train_filtered = train[mask].reset_index(drop=True)
    X_train_df = X_train_df[mask].reset_index(drop=True)
    y_train = train_filtered['price'].values
    
    # Fit preprocessor and transform
    preprocessor.fit(X_train_df)
    X_train_orig = preprocessor.transform(X_train_df)
    X_test_orig = preprocessor.transform(X_test_df)
    
    print(f"Original features: {X_train_orig.shape[1]}")
    
    # Add NEW distance features
    print("Computing landmark features...")
    train_landmarks_2km, train_min_dist = compute_landmark_features(train_filtered)
    test_landmarks_2km, test_min_dist = compute_landmark_features(test)
    
    # Scale new features with same approach
    scaler_new = StandardScaler()
    new_train = np.column_stack([train_landmarks_2km, train_min_dist])
    new_test = np.column_stack([test_landmarks_2km, test_min_dist])
    
    new_train_scaled = scaler_new.fit_transform(new_train)
    new_test_scaled = scaler_new.transform(new_test)
    
    # Combine: original 15 + 2 new = 17 features
    X_train = np.hstack([X_train_orig, new_train_scaled])
    X_test = np.hstack([X_test_orig, new_test_scaled])
    
    feature_names = preprocessor.feature_cols + ['num_landmarks_2km', 'min_landmark_dist']
    
    print(f"Final features: {X_train.shape[1]}")
    print(f"Training samples: {len(y_train):,}")
    print(f"Test samples: {len(X_test):,}")
    
    return X_train, y_train, X_test, test, feature_names, preprocessor


# ============================================================
# TUNING FUNCTIONS
# ============================================================
def tune_lightgbm(X, y, n_trials=15):
    """Tune LightGBM with new features."""
    print("\n" + "=" * 70)
    print("TUNING LIGHTGBM")
    print("=" * 70)
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': 3000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'random_state': 42,
            'force_col_wise': True
        }
        
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, np.log1p(y_tr),
                     eval_set=[(X_val, np.log1p(y_val))],
                     callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                               lgb.log_evaluation(0)])
            
            oof[val_idx] = np.expm1(model.predict(X_val))
        
        return rmse(y, oof)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Best LGB CV: {study.best_value:.2f}")
    return study.best_params


def tune_catboost(X, y, n_trials=15):
    """Tune CatBoost."""
    print("\n" + "=" * 70)
    print("TUNING CATBOOST")
    print("=" * 70)
    
    def objective(trial):
        params = {
            'iterations': 3000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_seed': 42,
            'verbose': False,
            'loss_function': 'RMSE',
            'early_stopping_rounds': EARLY_STOPPING
        }
        
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = CatBoostRegressor(**params)
            model.fit(X_tr, np.log1p(y_tr),
                     eval_set=(X_val, np.log1p(y_val)),
                     verbose=False)
            
            oof[val_idx] = np.expm1(model.predict(X_val))
        
        return rmse(y, oof)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Best CAT CV: {study.best_value:.2f}")
    return study.best_params


def tune_xgboost(X, y, n_trials=15):
    """Tune XGBoost."""
    print("\n" + "=" * 70)
    print("TUNING XGBOOST")
    print("=" * 70)
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 3000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'random_state': 42,
            'verbosity': 0,
            'early_stopping_rounds': EARLY_STOPPING
        }
        
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, np.log1p(y_tr),
                     eval_set=[(X_val, np.log1p(y_val))],
                     verbose=False)
            
            oof[val_idx] = np.expm1(model.predict(X_val))
        
        return rmse(y, oof)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Best XGB CV: {study.best_value:.2f}")
    return study.best_params


# ============================================================
# FINAL TRAINING
# ============================================================
def train_final(X, y, X_test, lgb_params, cat_params, xgb_params):
    """Train with seed averaging."""
    print("\n" + "=" * 70)
    print("FINAL TRAINING WITH SEED AVERAGING")
    print("=" * 70)
    
    LGB_SEEDS = [42, 123, 456, 789, 1024]
    CAT_SEEDS = [42, 123, 456, 789, 1024]
    XGB_SEEDS = [42, 123, 456]
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    oof_lgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    test_lgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    
    # LightGBM
    print("\nLightGBM...")
    for seed in LGB_SEEDS:
        seed_oof = np.zeros(len(y))
        seed_test = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            model = lgb.LGBMRegressor(
                objective='regression', metric='rmse', verbosity=-1,
                n_estimators=10000, random_state=seed, force_col_wise=True,
                **lgb_params
            )
            model.fit(X[train_idx], np.log1p(y[train_idx]),
                     eval_set=[(X[val_idx], np.log1p(y[val_idx]))],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            seed_oof[val_idx] = np.expm1(model.predict(X[val_idx]))
            
            model_full = lgb.LGBMRegressor(
                objective='regression', metric='rmse', verbosity=-1,
                n_estimators=model.best_iteration_, random_state=seed, force_col_wise=True,
                **lgb_params
            )
            model_full.fit(X, np.log1p(y))
            seed_test += np.expm1(model_full.predict(X_test)) / N_FOLDS
        
        oof_lgb += seed_oof / len(LGB_SEEDS)
        test_lgb += seed_test / len(LGB_SEEDS)
        print(f"  Seed {seed}: {rmse(y, seed_oof):.2f}")
    
    print(f"LGB CV: {rmse(y, oof_lgb):.2f}")
    
    # CatBoost
    print("\nCatBoost...")
    for seed in CAT_SEEDS:
        seed_oof = np.zeros(len(y))
        seed_test = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            model = CatBoostRegressor(
                iterations=10000, random_seed=seed, verbose=False,
                loss_function='RMSE', early_stopping_rounds=100,
                **cat_params
            )
            model.fit(X[train_idx], np.log1p(y[train_idx]),
                     eval_set=(X[val_idx], np.log1p(y[val_idx])), verbose=False)
            seed_oof[val_idx] = np.expm1(model.predict(X[val_idx]))
            
            model_full = CatBoostRegressor(
                iterations=model.best_iteration_, random_seed=seed, verbose=False,
                loss_function='RMSE', **cat_params
            )
            model_full.fit(X, np.log1p(y), verbose=False)
            seed_test += np.expm1(model_full.predict(X_test)) / N_FOLDS
        
        oof_cat += seed_oof / len(CAT_SEEDS)
        test_cat += seed_test / len(CAT_SEEDS)
        print(f"  Seed {seed}: {rmse(y, seed_oof):.2f}")
    
    print(f"CAT CV: {rmse(y, oof_cat):.2f}")
    
    # XGBoost
    print("\nXGBoost...")
    for seed in XGB_SEEDS:
        seed_oof = np.zeros(len(y))
        seed_test = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            model = xgb.XGBRegressor(
                objective='reg:squarederror', eval_metric='rmse',
                n_estimators=10000, random_state=seed, verbosity=0,
                early_stopping_rounds=100, **xgb_params
            )
            model.fit(X[train_idx], np.log1p(y[train_idx]),
                     eval_set=[(X[val_idx], np.log1p(y[val_idx]))], verbose=False)
            seed_oof[val_idx] = np.expm1(model.predict(X[val_idx]))
            
            model_full = xgb.XGBRegressor(
                objective='reg:squarederror', eval_metric='rmse',
                n_estimators=model.best_iteration, random_state=seed, verbosity=0,
                **xgb_params
            )
            model_full.fit(X, np.log1p(y), verbose=False)
            seed_test += np.expm1(model_full.predict(X_test)) / N_FOLDS
        
        oof_xgb += seed_oof / len(XGB_SEEDS)
        test_xgb += seed_test / len(XGB_SEEDS)
        print(f"  Seed {seed}: {rmse(y, seed_oof):.2f}")
    
    print(f"XGB CV: {rmse(y, oof_xgb):.2f}")
    
    return oof_lgb, oof_cat, oof_xgb, test_lgb, test_cat, test_xgb


def main():
    # Load data with ORIGINAL preprocessing + new features
    X_train, y_train, X_test, test_df, feature_names, preprocessor = load_data()
    
    # Tune
    lgb_params = tune_lightgbm(X_train, y_train, TUNING_TRIALS)
    cat_params = tune_catboost(X_train, y_train, TUNING_TRIALS)
    xgb_params = tune_xgboost(X_train, y_train, TUNING_TRIALS)
    
    # Train
    oof_lgb, oof_cat, oof_xgb, test_lgb, test_cat, test_xgb = train_final(
        X_train, y_train, X_test, lgb_params, cat_params, xgb_params
    )
    
    # Find weights
    from scipy.optimize import minimize
    def ensemble_rmse(w):
        w = w / w.sum()
        return rmse(y_train, w[0]*oof_lgb + w[1]*oof_cat + w[2]*oof_xgb)
    
    result = minimize(ensemble_rmse, [0.33, 0.33, 0.34], bounds=[(0,1)]*3, method='L-BFGS-B')
    weights = result.x / result.x.sum()
    
    print(f"\nWeights: LGB={weights[0]:.3f}, CAT={weights[1]:.3f}, XGB={weights[2]:.3f}")
    
    # Final predictions
    oof_ens = weights[0]*oof_lgb + weights[1]*oof_cat + weights[2]*oof_xgb
    test_ens = weights[0]*test_lgb + weights[1]*test_cat + weights[2]*test_xgb
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"LGB CV: {rmse(y_train, oof_lgb):.2f}")
    print(f"CAT CV: {rmse(y_train, oof_cat):.2f}")
    print(f"XGB CV: {rmse(y_train, oof_xgb):.2f}")
    print(f"Ensemble CV: {rmse(y_train, oof_ens):.2f}")
    
    # Save
    pd.DataFrame({'id': test_df['_id'], 'price': test_ens}).to_csv(
        'outputs/predictions/submission_original_plus_landmarks.csv', index=False
    )
    print("\nSaved: outputs/predictions/submission_original_plus_landmarks.csv")


if __name__ == "__main__":
    main()
