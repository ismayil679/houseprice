"""
Training Script V3 - Uses Enhanced Preprocessing with Luxury Indicators
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')

from preprocessing_v3 import HousePricePreprocessorV3, load_data


def train_and_evaluate():
    """Train ensemble with V3 preprocessing"""
    
    # Load data
    train_df, test_df = load_data()
    
    # Initialize preprocessor
    preprocessor = HousePricePreprocessorV3()
    
    # =====================================================================
    # OUTLIER REMOVAL (1st and 99th percentile)
    # =====================================================================
    y_full = train_df['price']
    q_low = y_full.quantile(0.01)
    q_high = y_full.quantile(0.99)
    mask = (y_full >= q_low) & (y_full <= q_high)
    train_df_filtered = train_df[mask].reset_index(drop=True)
    
    print(f"\nOutlier removal: {len(train_df)} -> {len(train_df_filtered)} samples")
    print(f"Price range: {q_low:,.0f} - {q_high:,.0f} AZN")
    
    # Extract features
    print("\n" + "="*60)
    print("EXTRACTING FEATURES WITH V3 PREPROCESSOR")
    print("="*60)
    
    X_train = preprocessor.extract_features(train_df_filtered, target=train_df_filtered['price'], is_train=True)
    X_test = preprocessor.extract_features(test_df, is_train=False)
    
    print(f"Training features: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    
    # Prepare target (log transform)
    y_train = np.log1p(train_df_filtered['price'].values)
    
    # Fit and transform
    preprocessor.fit(X_train)
    X_train_arr = preprocessor.transform(X_train)
    X_test_arr = preprocessor.transform(X_test)
    
    feature_names = preprocessor.feature_cols
    
    # Hyperparameters (tuned for original features - may need re-tuning)
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
        'n_estimators': 10000,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    cat_params = {
        'iterations': 10000,
        'learning_rate': 0.03,
        'depth': 8,
        'l2_leaf_reg': 5,
        'random_seed': 42,
        'verbose': 0,
        'early_stopping_rounds': 200
    }
    
    xgb_params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.03,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 10000,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 200
    }
    
    # Cross-validation
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_lgb = np.zeros(len(X_train_arr))
    oof_cat = np.zeros(len(X_train_arr))
    oof_xgb = np.zeros(len(X_train_arr))
    
    test_preds_lgb = np.zeros(len(X_test_arr))
    test_preds_cat = np.zeros(len(X_test_arr))
    test_preds_xgb = np.zeros(len(X_test_arr))
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION TRAINING")
    print("="*60)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_arr)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        X_tr, X_val = X_train_arr[train_idx], X_train_arr[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # LightGBM
        lgb_train = lgb.Dataset(X_tr, y_tr, feature_name=feature_names)
        lgb_val = lgb.Dataset(X_val, y_val, feature_name=feature_names, reference=lgb_train)
        
        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )
        
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        test_preds_lgb += lgb_model.predict(X_test_arr) / n_folds
        
        # CatBoost
        cat_model = cb.CatBoostRegressor(**cat_params)
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        
        oof_cat[val_idx] = cat_model.predict(X_val)
        test_preds_cat += cat_model.predict(X_test_arr) / n_folds
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=0
        )
        
        oof_xgb[val_idx] = xgb_model.predict(X_val)
        test_preds_xgb += xgb_model.predict(X_test_arr) / n_folds
        
        # Fold ensemble
        oof_ensemble = 0.4 * oof_lgb[val_idx] + 0.35 * oof_cat[val_idx] + 0.25 * oof_xgb[val_idx]
        
        fold_rmse_lgb = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(oof_lgb[val_idx])))
        fold_rmse_cat = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(oof_cat[val_idx])))
        fold_rmse_xgb = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(oof_xgb[val_idx])))
        fold_rmse_ens = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(oof_ensemble)))
        
        fold_scores.append(fold_rmse_ens)
        
        print(f"  LGB: {fold_rmse_lgb:,.0f} | CAT: {fold_rmse_cat:,.0f} | XGB: {fold_rmse_xgb:,.0f} | ENS: {fold_rmse_ens:,.0f}")
    
    # Overall scores
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    
    oof_ensemble = 0.4 * oof_lgb + 0.35 * oof_cat + 0.25 * oof_xgb
    
    overall_rmse_lgb = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(oof_lgb)))
    overall_rmse_cat = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(oof_cat)))
    overall_rmse_xgb = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(oof_xgb)))
    overall_rmse_ens = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(oof_ensemble)))
    
    print(f"\nLightGBM CV RMSE:  {overall_rmse_lgb:,.0f}")
    print(f"CatBoost CV RMSE:  {overall_rmse_cat:,.0f}")
    print(f"XGBoost CV RMSE:   {overall_rmse_xgb:,.0f}")
    print(f"Ensemble CV RMSE:  {overall_rmse_ens:,.0f}")
    
    print(f"\nFold scores: {[f'{s:,.0f}' for s in fold_scores]}")
    print(f"Fold std: {np.std(fold_scores):,.0f}")
    
    # Feature importance (from last LGB model)
    print("\n" + "="*60)
    print("TOP 20 FEATURE IMPORTANCE (LightGBM)")
    print("="*60)
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': lgb_model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    for i, row in importance.head(20).iterrows():
        print(f"  {row['feature']:35} {row['importance']:,.0f}")
    
    # Generate submission
    test_preds_ensemble = 0.4 * test_preds_lgb + 0.35 * test_preds_cat + 0.25 * test_preds_xgb
    test_preds_final = np.expm1(test_preds_ensemble)
    test_preds_final = np.maximum(test_preds_final, 0)
    
    submission = pd.DataFrame({
        'id': test_df['_id'],
        'price': test_preds_final
    })
    
    os.makedirs('outputs/predictions', exist_ok=True)
    submission.to_csv('outputs/predictions/submission_v3_luxury.csv', index=False)
    print(f"\nSubmission saved to outputs/predictions/submission_v3_luxury.csv")
    
    return overall_rmse_ens


if __name__ == '__main__':
    rmse = train_and_evaluate()
