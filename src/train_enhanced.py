"""
Enhanced Preprocessing with Distance Features
==============================================
Adds distance-based features from baku_coordinates.xlsx to the existing pipeline.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

from preprocessing import HousePricePreprocessor
from feature_engineering import load_landmarks, create_distance_features

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def load_and_enhance_data():
    """Load data and add distance features."""
    print("=" * 60)
    print("LOADING DATA WITH DISTANCE FEATURES")
    print("=" * 60)
    
    # Use the saved preprocessor (contains target encoding stats)
    preprocessor = joblib.load('models/fine_tuned_preprocessor.pkl')
    
    # Load raw data
    train = pd.read_csv('data/binaaz_train.csv')
    test = pd.read_csv('data/binaaz_test.csv')
    
    # Load landmarks
    landmarks_df, metro_df, other_df = load_landmarks()
    print(f"Landmarks: {len(landmarks_df)} total, {len(metro_df)} metros")
    
    # Create distance features
    print("\nCreating distance features...")
    train = create_distance_features(train, landmarks_df, metro_df, other_df)
    test = create_distance_features(test, landmarks_df, metro_df, other_df)
    
    y_full = train['price']
    
    # Remove outliers (1st and 99th percentile) - same as original
    q_low = y_full.quantile(0.01)
    q_high = y_full.quantile(0.99)
    mask = (y_full >= q_low) & (y_full <= q_high)
    train_filtered = train[mask].reset_index(drop=True)
    
    # Transform base features using preprocessor
    X_train_base = preprocessor.extract_features(train_filtered)
    X_train_base = preprocessor.transform(X_train_base)
    y_train = train_filtered['price'].values
    
    X_test_base = preprocessor.extract_features(test)
    X_test_base = preprocessor.transform(X_test_base)
    
    # Add distance features (already in train_filtered and test)
    # Only add features that provide NEW information (not redundant with dist_from_center)
    distance_cols = [
        'num_landmarks_within_2km',  # Key feature! 0.378 correlation with price
        'distance_to_nearest_metro',  # Specific to metro, not just center
    ]
    
    # Normalize distance features
    from sklearn.preprocessing import StandardScaler
    dist_scaler = StandardScaler()
    
    X_dist_train = train_filtered[distance_cols].fillna(0).values
    X_dist_train = dist_scaler.fit_transform(X_dist_train)
    
    X_dist_test = test[distance_cols].fillna(0).values
    X_dist_test = dist_scaler.transform(X_dist_test)
    
    # Combine base + distance features
    X_train = np.hstack([X_train_base, X_dist_train])
    X_test = np.hstack([X_test_base, X_dist_test])
    
    # Feature names
    base_features = preprocessor.feature_cols
    all_features = base_features + distance_cols
    
    print(f"\nBase features: {len(base_features)}")
    print(f"Distance features: {len(distance_cols)}")
    print(f"Total features: {len(all_features)}")
    print(f"Train samples: {len(y_train):,}")
    print(f"Test samples: {len(X_test):,}")
    
    return X_train, y_train, X_test, test, all_features


def train_ensemble_with_log(X, y, feature_names):
    """Train ensemble using log-transformed target."""
    print("\n" + "=" * 60)
    print("TRAINING ENSEMBLE WITH LOG TARGET")
    print("=" * 60)
    
    # Log transform target
    y_log = np.log1p(y)
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Storage for OOF predictions (in log scale)
    oof_lgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    
    lgb_models = []
    cat_models = []
    xgb_models = []
    
    # LightGBM parameters (from tuning)
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 197,
        'max_depth': 10,
        'min_child_samples': 11,
        'subsample': 0.89,
        'colsample_bytree': 0.74,
        'reg_alpha': 0.81,
        'reg_lambda': 4.55,
        'verbose': -1,
        'n_jobs': -1
    }
    
    # CatBoost parameters
    cat_params = {
        'iterations': 10000,
        'learning_rate': 0.03,
        'depth': 8,
        'l2_leaf_reg': 3.0,
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
        'learning_rate': 0.02,
        'max_depth': 8,
        'min_child_weight': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'tree_method': 'hist',
        'random_state': 42
    }
    
    n_boost = 10000  # More iterations
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_log[train_idx], y_log[val_idx]
        y_val_orig = y[val_idx]  # Original scale for RMSE
        
        # LightGBM
        lgb_train = lgb.Dataset(X_tr, y_tr, feature_name=feature_names)
        lgb_val = lgb.Dataset(X_val, y_val, feature_name=feature_names, reference=lgb_train)
        
        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=n_boost,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        lgb_models.append(lgb_model)
        oof_lgb[val_idx] = np.expm1(lgb_model.predict(X_val))
        lgb_rmse = rmse(y_val_orig, oof_lgb[val_idx])
        
        # CatBoost
        cat_model = cb.CatBoostRegressor(**cat_params)
        cat_model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=False
        )
        cat_models.append(cat_model)
        oof_cat[val_idx] = np.expm1(cat_model.predict(X_val))
        cat_rmse = rmse(y_val_orig, oof_cat[val_idx])
        
        # XGBoost
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        
        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=n_boost,
            evals=[(dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=False
        )
        xgb_models.append(xgb_model)
        oof_xgb[val_idx] = np.expm1(xgb_model.predict(dval))
        xgb_rmse = rmse(y_val_orig, oof_xgb[val_idx])
        
        print(f"  LGB: {lgb_rmse:.2f}, CAT: {cat_rmse:.2f}, XGB: {xgb_rmse:.2f}")
    
    # Overall CV scores (in original scale)
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
    
    return lgb_models, cat_models, xgb_models, optimal_weights, oof_ensemble


def make_predictions(X_test, feature_names, lgb_models, cat_models, xgb_models, weights):
    """Generate test predictions."""
    print("\n" + "=" * 60)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 60)
    
    # LightGBM predictions (inverse log transform)
    lgb_preds = np.mean([np.expm1(m.predict(X_test)) for m in lgb_models], axis=0)
    
    # CatBoost predictions
    cat_preds = np.mean([np.expm1(m.predict(X_test)) for m in cat_models], axis=0)
    
    # XGBoost predictions
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    xgb_preds = np.mean([np.expm1(m.predict(dtest)) for m in xgb_models], axis=0)
    
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
    """Main pipeline."""
    # Load data with distance features
    X_train, y_train, X_test, test_df, feature_names = load_and_enhance_data()
    
    # Train ensemble
    lgb_models, cat_models, xgb_models, weights, oof_preds = train_ensemble_with_log(
        X_train, y_train, feature_names
    )
    
    # Generate predictions
    test_preds = make_predictions(X_test, feature_names, lgb_models, cat_models, xgb_models, weights)
    
    # Save submission
    submission = pd.DataFrame({
        'id': test_df['_id'],
        'price': test_preds
    })
    submission.to_csv('outputs/predictions/submission_with_distance.csv', index=False)
    print(f"\nSubmission saved to outputs/predictions/submission_with_distance.csv")
    
    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (LightGBM)")
    print("=" * 60)
    importance = lgb_models[0].feature_importance(importance_type='gain')
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    print(feat_imp.to_string(index=False))
    
    return oof_preds, y_train


if __name__ == "__main__":
    oof_preds, y = main()
