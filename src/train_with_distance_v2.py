"""
Training Pipeline with Distance Features V2
============================================
Properly handles target encoding within CV folds.

Current best: LB 29,318 (CV 37,170)
Target: LB 28,500
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# Import feature engineering functions
from feature_engineering import load_landmarks, create_distance_features, METRO_NAME_MAPPING

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def parse_floor(floor_str):
    """Parse floor string like '5 / 17' into (floor, total_floors)."""
    if pd.isna(floor_str):
        return None, None
    try:
        parts = str(floor_str).split('/')
        if len(parts) == 2:
            return int(parts[0].strip()), int(parts[1].strip())
        return None, None
    except:
        return None, None

def parse_area(area_str):
    """Parse area string like '135 m²' into numeric value."""
    if pd.isna(area_str):
        return None
    try:
        return float(str(area_str).replace('m²', '').replace(',', '.').strip())
    except:
        return None

def extract_year(title_or_desc, default=2010):
    """Try to extract build year from title or description."""
    import re
    if pd.isna(title_or_desc):
        return default
    match = re.search(r'(19[89]\d|20[0-2]\d)', str(title_or_desc))
    if match:
        return int(match.group(1))
    return default


def load_and_prepare_data():
    """Load data and create all features including new distance features."""
    print("=" * 60)
    print("LOADING AND PREPARING DATA WITH NEW FEATURES")
    print("=" * 60)
    
    # Load raw data
    train = pd.read_csv('data/binaaz_train.csv')
    test = pd.read_csv('data/binaaz_test.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # Load landmarks
    landmarks_df, metro_df, other_df = load_landmarks()
    print(f"\nLandmarks: {len(landmarks_df)} total, {len(metro_df)} metros")
    
    # Create distance features
    print("\nCreating distance features for train...")
    train = create_distance_features(train, landmarks_df, metro_df, other_df)
    
    print("Creating distance features for test...")
    test = create_distance_features(test, landmarks_df, metro_df, other_df)
    
    return train, test


def prepare_base_features(train, test):
    """Prepare base numeric features (no target encoding)."""
    print("\n" + "=" * 60)
    print("PREPARING BASE FEATURES")
    print("=" * 60)
    
    # Parse area
    train['area'] = train['Sahə'].apply(parse_area)
    test['area'] = test['Sahə'].apply(parse_area)
    
    # Parse rooms
    train['rooms'] = pd.to_numeric(train['Otaq sayı'].replace('6+', 6), errors='coerce')
    test['rooms'] = pd.to_numeric(test['Otaq sayı'].replace('6+', 6), errors='coerce')
    
    # Parse floor
    floor_data = train['Mərtəbə'].apply(parse_floor)
    train['floor'] = [f[0] for f in floor_data]
    train['total_floors'] = [f[1] for f in floor_data]
    
    floor_data_test = test['Mərtəbə'].apply(parse_floor)
    test['floor'] = [f[0] for f in floor_data_test]
    test['total_floors'] = [f[1] for f in floor_data_test]
    
    # Fill numeric NaN
    for col in ['area', 'rooms', 'floor', 'total_floors']:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)
    
    # Fill lat/lon
    lat_median = train['latitude'].median()
    lon_median = train['longitude'].median()
    train['latitude'] = train['latitude'].fillna(lat_median)
    train['longitude'] = train['longitude'].fillna(lon_median)
    test['latitude'] = test['latitude'].fillna(lat_median)
    test['longitude'] = test['longitude'].fillna(lon_median)
    
    # Age feature
    train['age'] = 2024 - train['title'].apply(lambda x: extract_year(x, 2010))
    test['age'] = 2024 - test['title'].apply(lambda x: extract_year(x, 2010))
    
    # Has repair - infer from title
    train['has_repair'] = train['title'].str.contains('təmirli|temirli|repair', case=False, na=False).astype(int)
    test['has_repair'] = test['title'].str.contains('təmirli|temirli|repair', case=False, na=False).astype(int)
    
    # Derived features
    train['floor_ratio'] = train['floor'] / train['total_floors'].replace(0, 1)
    test['floor_ratio'] = test['floor'] / test['total_floors'].replace(0, 1)
    
    train['rooms_per_area'] = train['rooms'] / train['area'].replace(0, 1)
    test['rooms_per_area'] = test['rooms'] / test['area'].replace(0, 1)
    
    train['is_new'] = (train['age'] <= 5).astype(int)
    test['is_new'] = (test['age'] <= 5).astype(int)
    
    train['area_per_room'] = train['area'] / train['rooms'].replace(0, 1)
    test['area_per_room'] = test['area'] / test['rooms'].replace(0, 1)
    
    train['is_ground_floor'] = (train['floor'] == 1).astype(int)
    test['is_ground_floor'] = (test['floor'] == 1).astype(int)
    
    train['is_top_floor'] = (train['floor'] == train['total_floors']).astype(int)
    test['is_top_floor'] = (test['floor'] == test['total_floors']).astype(int)
    
    # Has metro nearby
    train['has_metro'] = train['metro_name'].notna().astype(int)
    test['has_metro'] = test['metro_name'].notna().astype(int)
    
    # Interaction features
    train['area_x_landmarks'] = train['area'] * train['num_landmarks_within_2km']
    test['area_x_landmarks'] = test['area'] * test['num_landmarks_within_2km']
    
    train['rooms_x_metro_dist'] = train['rooms'] / (train['distance_to_nearest_metro'] + 0.1)
    test['rooms_x_metro_dist'] = test['rooms'] / (test['distance_to_nearest_metro'] + 0.1)
    
    # Label encoding for categorical (no target leakage)
    # District
    district_col = 'locations'
    le_district = LabelEncoder()
    train['district_encoded'] = le_district.fit_transform(train[district_col].astype(str))
    test['district_encoded'] = test[district_col].astype(str).apply(
        lambda x: le_district.transform([x])[0] if x in le_district.classes_ else -1
    )
    
    # Metro station label encoding
    train['metro_encoded'] = LabelEncoder().fit_transform(train['metro_name'].fillna('none').astype(str))
    test['metro_encoded'] = train['metro_name'].fillna('none').astype(str)  # Will use cat_features
    
    # Nearest metro landmark encoding
    le_nearest = LabelEncoder()
    train['nearest_metro_encoded'] = le_nearest.fit_transform(train['nearest_metro_landmark'].astype(str))
    test['nearest_metro_encoded'] = test['nearest_metro_landmark'].astype(str).apply(
        lambda x: le_nearest.transform([x])[0] if x in le_nearest.classes_ else -1
    )
    
    # Seher (city) encoding
    le_seher = LabelEncoder()
    train['seher_encoded'] = le_seher.fit_transform(train['seher'].fillna('unknown').astype(str))
    test['seher_encoded'] = test['seher'].fillna('unknown').astype(str).apply(
        lambda x: le_seher.transform([x])[0] if x in le_seher.classes_ else -1
    )
    
    # Define feature columns
    numeric_features = [
        'area', 'rooms', 'floor', 'total_floors', 'age',
        'has_repair', 'latitude', 'longitude', 'floor_ratio',
        'rooms_per_area', 'is_new', 'area_per_room',
        'is_ground_floor', 'is_top_floor', 'has_metro',
        # New distance features
        'distance_to_center', 'distance_to_nearest_metro',
        'distance_to_nearest_landmark', 'num_landmarks_within_2km',
        'avg_distance_top5_landmarks', 'distance_to_old_city',
        # Interactions
        'area_x_landmarks', 'rooms_x_metro_dist',
        # Label encoded categoricals
        'district_encoded', 'nearest_metro_encoded', 'seher_encoded'
    ]
    
    # Fill any remaining NaN
    for col in numeric_features:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)
    
    # Extract target and IDs
    y = train['price'].values
    test_ids = test['_id'] if '_id' in test.columns else test.index
    
    X_train = train[numeric_features].values
    X_test = test[numeric_features].values
    
    print(f"Feature count: {len(numeric_features)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    return X_train, y, X_test, numeric_features, test_ids, train, test


def train_ensemble_cv(X, y, feature_names, train_df):
    """Train ensemble with proper CV (target encoding within folds)."""
    print("\n" + "=" * 60)
    print("TRAINING ENSEMBLE WITH 5-FOLD CV + IN-FOLD TARGET ENCODING")
    print("=" * 60)
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Storage
    oof_lgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    
    lgb_models = []
    cat_models = []
    xgb_models = []
    
    # Target encoding columns
    te_cols = ['locations', 'metro_name', 'nearest_metro_landmark', 'seher']
    
    # LightGBM parameters (from previous tuning)
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1
    }
    
    # CatBoost parameters
    cat_params = {
        'iterations': 2000,
        'learning_rate': 0.03,
        'depth': 8,
        'l2_leaf_reg': 3,
        'random_strength': 0.5,
        'bagging_temperature': 0.5,
        'border_count': 128,
        'loss_function': 'RMSE',
        'verbose': False,
        'random_seed': 42
    }
    
    # XGBoost parameters  
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.03,
        'max_depth': 8,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'random_state': 42
    }
    
    # Prepare augmented feature names (with target encoding)
    te_feature_names = feature_names + [f'{col}_te' for col in te_cols]
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        # Base features
        X_tr_base = X[train_idx].copy()
        X_val_base = X[val_idx].copy()
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Add target encoding features (computed on training fold only)
        te_train = []
        te_val = []
        
        for col in te_cols:
            # Compute mean on training fold
            train_fold_df = train_df.iloc[train_idx]
            val_fold_df = train_df.iloc[val_idx]
            
            col_means = train_fold_df.groupby(col)['price'].mean()
            global_mean = y_tr.mean()
            
            # Map to training and validation
            te_train.append(train_fold_df[col].map(col_means).fillna(global_mean).values.reshape(-1, 1))
            te_val.append(val_fold_df[col].map(col_means).fillna(global_mean).values.reshape(-1, 1))
        
        # Combine base + target encoded features
        X_tr = np.hstack([X_tr_base] + te_train)
        X_val = np.hstack([X_val_base] + te_val)
        
        # LightGBM
        lgb_train = lgb.Dataset(X_tr, y_tr, feature_name=te_feature_names)
        lgb_val = lgb.Dataset(X_val, y_val, feature_name=te_feature_names, reference=lgb_train)
        
        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=2000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        lgb_models.append(lgb_model)
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        lgb_rmse = rmse(y_val, oof_lgb[val_idx])
        
        # CatBoost
        cat_model = cb.CatBoostRegressor(**cat_params)
        cat_model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=False
        )
        cat_models.append(cat_model)
        oof_cat[val_idx] = cat_model.predict(X_val)
        cat_rmse = rmse(y_val, oof_cat[val_idx])
        
        # XGBoost
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=te_feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=te_feature_names)
        
        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=False
        )
        xgb_models.append(xgb_model)
        oof_xgb[val_idx] = xgb_model.predict(dval)
        xgb_rmse = rmse(y_val, oof_xgb[val_idx])
        
        print(f"  LGB: {lgb_rmse:.2f}, CAT: {cat_rmse:.2f}, XGB: {xgb_rmse:.2f}")
    
    # Overall CV scores
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"LightGBM CV: {rmse(y, oof_lgb):.2f}")
    print(f"CatBoost CV:  {rmse(y, oof_cat):.2f}")
    print(f"XGBoost CV:  {rmse(y, oof_xgb):.2f}")
    
    # Find optimal weights
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
    
    optimal_weights = result.x / result.x.sum()
    print(f"\nOptimal weights: LGB={optimal_weights[0]:.3f}, CAT={optimal_weights[1]:.3f}, XGB={optimal_weights[2]:.3f}")
    
    oof_ensemble = (optimal_weights[0] * oof_lgb + 
                   optimal_weights[1] * oof_cat + 
                   optimal_weights[2] * oof_xgb)
    print(f"Ensemble CV: {rmse(y, oof_ensemble):.2f}")
    
    # Compute global target encodings for test
    global_te = {}
    for col in te_cols:
        global_te[col] = train_df.groupby(col)['price'].mean()
    global_mean = y.mean()
    
    return lgb_models, cat_models, xgb_models, optimal_weights, oof_ensemble, global_te, global_mean, te_feature_names


def make_predictions(X_test, test_df, feature_names, lgb_models, cat_models, xgb_models, weights, global_te, global_mean, te_feature_names):
    """Generate test predictions."""
    print("\n" + "=" * 60)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 60)
    
    te_cols = ['locations', 'metro_name', 'nearest_metro_landmark', 'seher']
    
    # Add target encoding to test
    te_test = []
    for col in te_cols:
        te_test.append(test_df[col].map(global_te[col]).fillna(global_mean).values.reshape(-1, 1))
    
    X_test_full = np.hstack([X_test] + te_test)
    
    # LightGBM predictions
    lgb_preds = np.mean([m.predict(X_test_full) for m in lgb_models], axis=0)
    
    # CatBoost predictions
    cat_preds = np.mean([m.predict(X_test_full) for m in cat_models], axis=0)
    
    # XGBoost predictions
    dtest = xgb.DMatrix(X_test_full, feature_names=te_feature_names)
    xgb_preds = np.mean([m.predict(dtest) for m in xgb_models], axis=0)
    
    # Ensemble
    final_preds = (weights[0] * lgb_preds + 
                  weights[1] * cat_preds + 
                  weights[2] * xgb_preds)
    
    # Clip negative predictions
    final_preds = np.maximum(final_preds, 0)
    
    print(f"LGB pred range: [{lgb_preds.min():.0f}, {lgb_preds.max():.0f}]")
    print(f"CAT pred range: [{cat_preds.min():.0f}, {cat_preds.max():.0f}]")
    print(f"XGB pred range: [{xgb_preds.min():.0f}, {xgb_preds.max():.0f}]")
    print(f"Ensemble range: [{final_preds.min():.0f}, {final_preds.max():.0f}]")
    
    return final_preds


def main():
    """Main training pipeline."""
    # Load and prepare data
    train_raw, test_raw = load_and_prepare_data()
    
    # Prepare base features
    X_train, y, X_test, feature_names, test_ids, train_df, test_df = prepare_base_features(train_raw, test_raw)
    
    # Train ensemble with proper CV
    lgb_models, cat_models, xgb_models, weights, oof_preds, global_te, global_mean, te_feature_names = train_ensemble_cv(
        X_train, y, feature_names, train_df
    )
    
    # Generate predictions
    test_preds = make_predictions(X_test, test_df, feature_names, lgb_models, cat_models, xgb_models, weights, global_te, global_mean, te_feature_names)
    
    # Save submission
    submission = pd.DataFrame({
        'id': test_ids,
        'price': test_preds
    })
    submission.to_csv('outputs/predictions/submission_distance_features_v2.csv', index=False)
    print(f"\nSubmission saved to outputs/predictions/submission_distance_features_v2.csv")
    print(f"Predictions: {len(submission)}")
    
    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (LightGBM)")
    print("=" * 60)
    importance = lgb_models[0].feature_importance(importance_type='gain')
    feat_imp = pd.DataFrame({
        'feature': te_feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    print(feat_imp.head(25).to_string(index=False))
    
    return oof_preds, y


if __name__ == "__main__":
    oof_preds, y = main()
