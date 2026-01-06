"""
Compare CV: With vs Without Distance Features
==============================================
Fair comparison using same settings to see if distance features help.
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

warnings.filterwarnings('ignore')

from preprocessing import HousePricePreprocessor
from feature_engineering import load_landmarks, create_distance_features

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def run_cv_comparison():
    """Compare CV with and without distance features."""
    print("=" * 70)
    print("CV COMPARISON: WITH vs WITHOUT DISTANCE FEATURES")
    print("=" * 70)
    
    # Load preprocessor
    preprocessor = joblib.load('models/fine_tuned_preprocessor.pkl')
    
    # Load raw data
    train = pd.read_csv('data/binaaz_train.csv')
    
    y_full = train['price']
    
    # Remove outliers (same as original)
    q_low = y_full.quantile(0.01)
    q_high = y_full.quantile(0.99)
    mask = (y_full >= q_low) & (y_full <= q_high)
    train_filtered = train[mask].reset_index(drop=True)
    
    # Base features
    X_base = preprocessor.extract_features(train_filtered)
    X_base = preprocessor.transform(X_base)
    y = train_filtered['price'].values
    
    print(f"\nBase features: {X_base.shape[1]}")
    print(f"Samples: {len(y):,}")
    
    # Create distance features
    print("\nCreating distance features...")
    landmarks_df, metro_df, other_df = load_landmarks()
    train_with_dist = create_distance_features(train_filtered, landmarks_df, metro_df, other_df)
    
    # Get distance features - ONLY the most valuable one
    distance_cols = ['num_landmarks_within_2km']  # 0.378 correlation with price
    from sklearn.preprocessing import StandardScaler
    dist_scaler = StandardScaler()
    X_dist = dist_scaler.fit_transform(train_with_dist[distance_cols].fillna(0).values)
    
    # Combined features
    X_enhanced = np.hstack([X_base, X_dist])
    print(f"Enhanced features: {X_enhanced.shape[1]}")
    
    # LightGBM parameters (same as original)
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
    
    # Run 5-fold CV for both
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_base = np.zeros(len(y))
    oof_enhanced = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_base)):
        X_tr_base = X_base[train_idx]
        X_val_base = X_base[val_idx]
        X_tr_enh = X_enhanced[train_idx]
        X_val_enh = X_enhanced[val_idx]
        y_tr = np.log1p(y[train_idx])
        y_val = np.log1p(y[val_idx])
        y_val_orig = y[val_idx]
        
        # Base model
        lgb_train_base = lgb.Dataset(X_tr_base, y_tr)
        lgb_val_base = lgb.Dataset(X_val_base, y_val, reference=lgb_train_base)
        model_base = lgb.train(
            lgb_params, lgb_train_base, num_boost_round=10000,
            valid_sets=[lgb_val_base],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        oof_base[val_idx] = np.expm1(model_base.predict(X_val_base))
        
        # Enhanced model
        lgb_train_enh = lgb.Dataset(X_tr_enh, y_tr)
        lgb_val_enh = lgb.Dataset(X_val_enh, y_val, reference=lgb_train_enh)
        model_enh = lgb.train(
            lgb_params, lgb_train_enh, num_boost_round=10000,
            valid_sets=[lgb_val_enh],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        oof_enhanced[val_idx] = np.expm1(model_enh.predict(X_val_enh))
        
        base_rmse = rmse(y_val_orig, oof_base[val_idx])
        enh_rmse = rmse(y_val_orig, oof_enhanced[val_idx])
        diff = base_rmse - enh_rmse
        
        print(f"Fold {fold+1}: Base={base_rmse:.0f}, Enhanced={enh_rmse:.0f}, Improvement={diff:.0f}")
    
    # Overall CV
    print("\n" + "=" * 70)
    print("OVERALL CV RESULTS")
    print("=" * 70)
    base_cv = rmse(y, oof_base)
    enh_cv = rmse(y, oof_enhanced)
    print(f"Base CV (15 features):     {base_cv:.2f}")
    print(f"Enhanced CV (17 features): {enh_cv:.2f}")
    print(f"Improvement:               {base_cv - enh_cv:.2f}")
    
    # Show which is better
    if enh_cv < base_cv:
        print(f"\n✅ Distance features IMPROVED CV by {base_cv - enh_cv:.2f}")
    else:
        print(f"\n❌ Distance features HURT CV by {enh_cv - base_cv:.2f}")


if __name__ == "__main__":
    run_cv_comparison()
