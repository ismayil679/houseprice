"""
Enhanced Feature Engineering + Model Tuning Pipeline
=====================================================
This script:
1. Creates new distance-based and derived features
2. Re-tunes hyperparameters for the new feature set
3. Trains final ensemble with seed averaging

Target: Beat LB 29,318 → reach 28,500
"""

import sys
sys.path.insert(0, 'src')

import os
import pandas as pd
import numpy as np
import re
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

# ============================================================
# CONFIGURATION
# ============================================================
N_FOLDS = 5
TUNING_TRIALS = 15  # Reduced for faster tuning
EARLY_STOPPING = 50  # Faster early stopping during tuning

# Baku center coordinates
BAKU_CENTER = (40.4093, 49.8671)

# ============================================================
# FEATURE ENGINEERING FUNCTIONS
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


def parse_area(s):
    """Extract numeric area from string like '135 m²'"""
    if pd.isna(s): return np.nan
    nums = re.findall(r'[\d.]+', str(s))
    return float(nums[0]) if nums else np.nan


def parse_floor(s):
    """Extract floor and total floors from '5 / 17'"""
    if pd.isna(s): return np.nan, np.nan
    parts = str(s).split('/')
    if len(parts) == 2:
        try: return float(parts[0].strip()), float(parts[1].strip())
        except: pass
    return np.nan, np.nan


def load_landmarks():
    """Load landmark coordinates."""
    landmarks = pd.read_excel('baku_coordinates.xlsx')
    return landmarks


def compute_landmark_features(df, landmarks):
    """Compute distance-based features from landmarks."""
    df = df.copy()
    
    # Clip coordinates to Baku area (filter outliers)
    df['lat_clean'] = df['latitude'].clip(39.5, 41.5)
    df['lon_clean'] = df['longitude'].clip(48.5, 51.0)
    
    # Get landmark coordinates
    lm_lats = landmarks['Latitude'].values
    lm_lons = landmarks['Longitude'].values
    
    n = len(df)
    num_within_2km = np.zeros(n)
    min_distance = np.zeros(n)
    
    batch_size = 5000
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_lats = df['lat_clean'].values[start:end]
        batch_lons = df['lon_clean'].values[start:end]
        
        # Distance matrix for batch
        dist_matrix = np.zeros((end - start, len(lm_lats)))
        for j in range(len(lm_lats)):
            dist_matrix[:, j] = haversine_distance(
                batch_lats, batch_lons, lm_lats[j], lm_lons[j]
            )
        
        num_within_2km[start:end] = (dist_matrix <= 2.0).sum(axis=1)
        min_distance[start:end] = dist_matrix.min(axis=1)
    
    df['num_landmarks_2km'] = num_within_2km
    df['min_landmark_dist'] = min_distance
    
    return df


def extract_all_features(df, landmarks, is_train=True, target_stats=None):
    """
    Extract all features from raw dataframe.
    
    For training: computes target encoding stats
    For test: uses provided target_stats
    """
    df = df.copy()
    
    # === BASIC FEATURES ===
    df['area'] = df['Sahə'].apply(parse_area)
    df['room_count'] = pd.to_numeric(df['Otaq sayı'].replace('6+', 6), errors='coerce')
    
    floors = df['Mərtəbə'].apply(parse_floor)
    df['floor'] = [f[0] for f in floors]
    df['total_floors'] = [f[1] for f in floors]
    
    # === BINARY FEATURES ===
    df['has_deed'] = (df['Kupça'] == 'var').astype(int) if 'Kupça' in df.columns else 0
    df['has_mortgage'] = (df['İpoteka'] == 'var').astype(int) if 'İpoteka' in df.columns else 0
    df['is_owner'] = (df['poster_type'] == 'mülkiyyətçi').astype(int) if 'poster_type' in df.columns else 0
    df['is_agent'] = (df['poster_type'] == 'vasitəçi (agent)').astype(int) if 'poster_type' in df.columns else 0
    
    # === LOCATION FEATURES ===
    df['lat'] = df['latitude'].clip(39.5, 41.5)
    df['lon'] = df['longitude'].clip(48.5, 51.0)
    
    # Distance from center (simple Euclidean - fast)
    df['dist_from_center'] = np.sqrt(
        (df['lat'] - BAKU_CENTER[0])**2 + (df['lon'] - BAKU_CENTER[1])**2
    )
    
    # City encoding
    df['city_baki'] = (df['seher'] == 'baki').astype(int)
    
    # === TEXT FEATURES ===
    df['title_length'] = df['title'].str.len().fillna(0)
    df['desc_length'] = df['description'].fillna('').str.len()
    df['is_new_building'] = df['title'].str.contains('yeni tikili', case=False, na=False).astype(int)
    
    # === DERIVED FEATURES ===
    df['floor_ratio'] = df['floor'] / df['total_floors'].replace(0, 1)
    df['area_per_room'] = df['area'] / df['room_count'].replace(0, 1)
    df['is_ground_floor'] = (df['floor'] == 1).astype(int)
    df['is_top_floor'] = (df['floor'] == df['total_floors']).astype(int)
    df['is_mid_floor'] = ((df['floor'] > 1) & (df['floor'] < df['total_floors'])).astype(int)
    df['is_high_rise'] = (df['total_floors'] >= 15).astype(int)
    
    # === LANDMARK FEATURES ===
    df = compute_landmark_features(df, landmarks)
    
    # Interaction: area × landmark density
    df['area_x_landmarks'] = df['area'] * df['num_landmarks_2km']
    
    # === TARGET ENCODING (location) ===
    if is_train:
        # Compute target encoding stats from training data
        target_stats = {}
        global_mean = df['price'].mean()
        
        # Location target encoding
        loc_means = df.groupby('locations')['price'].mean()
        target_stats['locations'] = loc_means
        target_stats['global_mean'] = global_mean
        
        df['location_price_mean'] = df['locations'].map(loc_means).fillna(global_mean)
    else:
        # Use provided stats
        df['location_price_mean'] = df['locations'].map(target_stats['locations']).fillna(target_stats['global_mean'])
    
    # === FINAL FEATURE LIST ===
    feature_cols = [
        # Basic
        'area', 'room_count', 'floor', 'total_floors',
        # Binary
        'has_deed', 'has_mortgage', 'is_owner', 'is_agent',
        # Location
        'lat', 'lon', 'dist_from_center', 'city_baki',
        # Text
        'title_length', 'desc_length', 'is_new_building',
        # Derived
        'floor_ratio', 'area_per_room', 'is_ground_floor', 
        'is_top_floor', 'is_mid_floor', 'is_high_rise',
        # Landmark-based (NEW!)
        'num_landmarks_2km', 'min_landmark_dist', 'area_x_landmarks',
        # Target encoded
        'location_price_mean'
    ]
    
    return df, feature_cols, target_stats


def preprocess_features(X_df, feature_cols, scaler=None, medians=None, fit=True):
    """Fill missing values and scale features."""
    X = X_df[feature_cols].copy()
    
    if fit:
        medians = X.median()
    
    for col in feature_cols:
        X[col] = X[col].fillna(medians[col])
    
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler, medians


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
def load_and_prepare():
    """Load data and extract all features."""
    print("=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)
    
    train = pd.read_csv('data/binaaz_train.csv')
    test = pd.read_csv('data/binaaz_test.csv')
    landmarks = load_landmarks()
    
    print(f"Train: {len(train):,} rows")
    print(f"Test: {len(test):,} rows")
    print(f"Landmarks: {len(landmarks)} locations")
    
    # Extract features
    print("\nExtracting features...")
    train_fe, feature_cols, target_stats = extract_all_features(train, landmarks, is_train=True)
    test_fe, _, _ = extract_all_features(test, landmarks, is_train=False, target_stats=target_stats)
    
    # Remove outliers (1st and 99th percentile)
    y_full = train_fe['price']
    q_low, q_high = y_full.quantile(0.01), y_full.quantile(0.99)
    mask = (y_full >= q_low) & (y_full <= q_high)
    train_fe = train_fe[mask].reset_index(drop=True)
    
    print(f"After outlier removal: {len(train_fe):,} rows")
    
    # Preprocess
    X_train, scaler, medians = preprocess_features(train_fe, feature_cols, fit=True)
    X_test, _, _ = preprocess_features(test_fe, feature_cols, scaler=scaler, medians=medians, fit=False)
    
    y_train = train_fe['price'].values
    
    print(f"Features: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")
    
    return X_train, y_train, X_test, test, feature_cols, target_stats, scaler, medians


# ============================================================
# CROSS-VALIDATION WITH LOG TRANSFORM
# ============================================================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def cv_score(X, y, model_fn, n_folds=5):
    """Calculate CV score using log-transformed target."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = model_fn()
        model.fit(X_tr, np.log1p(y_tr), 
                 eval_set=[(X_val, np.log1p(y_val))],
                 callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                           lgb.log_evaluation(0)] if hasattr(model, 'n_estimators') else None)
        
        oof[val_idx] = np.expm1(model.predict(X_val))
    
    return rmse(y, oof)


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================
def tune_lightgbm(X, y, n_trials=30):
    """Tune LightGBM hyperparameters."""
    print("\n" + "=" * 70)
    print("TUNING LIGHTGBM")
    print("=" * 70)
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': 2000,  # Reduced for faster tuning
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
    print(f"Best params: {study.best_params}")
    
    return study.best_params


def tune_catboost(X, y, n_trials=30):
    """Tune CatBoost hyperparameters."""
    print("\n" + "=" * 70)
    print("TUNING CATBOOST")
    print("=" * 70)
    
    def objective(trial):
        params = {
            'iterations': 2000,  # Reduced for faster tuning
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
    print(f"Best params: {study.best_params}")
    
    return study.best_params


def tune_xgboost(X, y, n_trials=30):
    """Tune XGBoost hyperparameters."""
    print("\n" + "=" * 70)
    print("TUNING XGBOOST")
    print("=" * 70)
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 2000,  # Reduced for faster tuning
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
    print(f"Best params: {study.best_params}")
    
    return study.best_params


# ============================================================
# FINAL TRAINING WITH SEED AVERAGING
# ============================================================
def train_final_models(X, y, X_test, lgb_params, cat_params, xgb_params):
    """Train final models with seed averaging."""
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODELS WITH SEED AVERAGING")
    print("=" * 70)
    
    LGB_SEEDS = [42, 123, 456, 789, 1024]
    CAT_SEEDS = [42, 123, 456, 789, 1024]
    XGB_SEEDS = [42, 123, 456]
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # Storage
    oof_lgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    
    test_lgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    
    # LightGBM
    print("\nTraining LightGBM...")
    for seed in LGB_SEEDS:
        seed_oof = np.zeros(len(y))
        seed_test = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = lgb.LGBMRegressor(
                objective='regression', metric='rmse', verbosity=-1,
                boosting_type='gbdt', n_estimators=10000,
                random_state=seed, force_col_wise=True,
                **lgb_params
            )
            model.fit(X_tr, np.log1p(y_tr),
                     eval_set=[(X_val, np.log1p(y_val))],
                     callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                               lgb.log_evaluation(0)])
            
            seed_oof[val_idx] = np.expm1(model.predict(X_val))
            
            # Full data model for test
            model_full = lgb.LGBMRegressor(
                objective='regression', metric='rmse', verbosity=-1,
                boosting_type='gbdt', n_estimators=model.best_iteration_,
                random_state=seed, force_col_wise=True,
                **lgb_params
            )
            model_full.fit(X, np.log1p(y))
            seed_test += np.expm1(model_full.predict(X_test)) / N_FOLDS
        
        oof_lgb += seed_oof / len(LGB_SEEDS)
        test_lgb += seed_test / len(LGB_SEEDS)
        print(f"  Seed {seed}: CV = {rmse(y, seed_oof):.2f}")
    
    print(f"LightGBM Final CV: {rmse(y, oof_lgb):.2f}")
    
    # CatBoost
    print("\nTraining CatBoost...")
    for seed in CAT_SEEDS:
        seed_oof = np.zeros(len(y))
        seed_test = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = CatBoostRegressor(
                iterations=10000, random_seed=seed,
                verbose=False, loss_function='RMSE',
                early_stopping_rounds=EARLY_STOPPING,
                **cat_params
            )
            model.fit(X_tr, np.log1p(y_tr),
                     eval_set=(X_val, np.log1p(y_val)),
                     verbose=False)
            
            seed_oof[val_idx] = np.expm1(model.predict(X_val))
            
            # Full data model
            model_full = CatBoostRegressor(
                iterations=model.best_iteration_, random_seed=seed,
                verbose=False, loss_function='RMSE',
                **cat_params
            )
            model_full.fit(X, np.log1p(y), verbose=False)
            seed_test += np.expm1(model_full.predict(X_test)) / N_FOLDS
        
        oof_cat += seed_oof / len(CAT_SEEDS)
        test_cat += seed_test / len(CAT_SEEDS)
        print(f"  Seed {seed}: CV = {rmse(y, seed_oof):.2f}")
    
    print(f"CatBoost Final CV: {rmse(y, oof_cat):.2f}")
    
    # XGBoost
    print("\nTraining XGBoost...")
    for seed in XGB_SEEDS:
        seed_oof = np.zeros(len(y))
        seed_test = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBRegressor(
                objective='reg:squarederror', eval_metric='rmse',
                n_estimators=10000, random_state=seed, verbosity=0,
                early_stopping_rounds=EARLY_STOPPING,
                **xgb_params
            )
            model.fit(X_tr, np.log1p(y_tr),
                     eval_set=[(X_val, np.log1p(y_val))],
                     verbose=False)
            
            seed_oof[val_idx] = np.expm1(model.predict(X_val))
            
            # Full data model
            model_full = xgb.XGBRegressor(
                objective='reg:squarederror', eval_metric='rmse',
                n_estimators=model.best_iteration, random_state=seed, verbosity=0,
                **xgb_params
            )
            model_full.fit(X, np.log1p(y), verbose=False)
            seed_test += np.expm1(model_full.predict(X_test)) / N_FOLDS
        
        oof_xgb += seed_oof / len(XGB_SEEDS)
        test_xgb += seed_test / len(XGB_SEEDS)
        print(f"  Seed {seed}: CV = {rmse(y, seed_oof):.2f}")
    
    print(f"XGBoost Final CV: {rmse(y, oof_xgb):.2f}")
    
    return oof_lgb, oof_cat, oof_xgb, test_lgb, test_cat, test_xgb


def find_optimal_weights(y, oof_lgb, oof_cat, oof_xgb):
    """Find optimal ensemble weights."""
    from scipy.optimize import minimize
    
    def ensemble_rmse(weights):
        w = weights / weights.sum()
        pred = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
        return rmse(y, pred)
    
    result = minimize(
        ensemble_rmse,
        x0=[0.33, 0.33, 0.34],
        bounds=[(0, 1), (0, 1), (0, 1)],
        method='L-BFGS-B'
    )
    
    weights = result.x / result.x.sum()
    return weights


# ============================================================
# MAIN
# ============================================================
def main():
    # Load data
    X_train, y_train, X_test, test_df, feature_cols, target_stats, scaler, medians = load_and_prepare()
    
    # Tune hyperparameters
    lgb_params = tune_lightgbm(X_train, y_train, n_trials=TUNING_TRIALS)
    cat_params = tune_catboost(X_train, y_train, n_trials=TUNING_TRIALS)
    xgb_params = tune_xgboost(X_train, y_train, n_trials=TUNING_TRIALS)
    
    # Save tuned params
    joblib.dump({
        'lgb': lgb_params,
        'cat': cat_params,
        'xgb': xgb_params,
        'feature_cols': feature_cols,
        'target_stats': target_stats,
        'scaler': scaler,
        'medians': medians
    }, 'models/tuned_params_v2.pkl')
    
    # Train final models
    oof_lgb, oof_cat, oof_xgb, test_lgb, test_cat, test_xgb = train_final_models(
        X_train, y_train, X_test, lgb_params, cat_params, xgb_params
    )
    
    # Find optimal weights
    weights = find_optimal_weights(y_train, oof_lgb, oof_cat, oof_xgb)
    print(f"\nOptimal weights: LGB={weights[0]:.3f}, CAT={weights[1]:.3f}, XGB={weights[2]:.3f}")
    
    # Ensemble predictions
    oof_ensemble = weights[0] * oof_lgb + weights[1] * oof_cat + weights[2] * oof_xgb
    test_ensemble = weights[0] * test_lgb + weights[1] * test_cat + weights[2] * test_xgb
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"LightGBM CV: {rmse(y_train, oof_lgb):.2f}")
    print(f"CatBoost CV:  {rmse(y_train, oof_cat):.2f}")
    print(f"XGBoost CV:  {rmse(y_train, oof_xgb):.2f}")
    print(f"Ensemble CV: {rmse(y_train, oof_ensemble):.2f}")
    
    # Save submission
    submission = pd.DataFrame({
        'id': test_df['_id'],
        'price': test_ensemble
    })
    submission.to_csv('outputs/predictions/submission_tuned_v2.csv', index=False)
    print(f"\nSubmission saved: outputs/predictions/submission_tuned_v2.csv")
    
    return oof_ensemble, y_train


if __name__ == "__main__":
    oof, y = main()
