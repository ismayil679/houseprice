"""
Train Tuned XGBoost + Combine with Existing LGB/CAT
====================================================
- Use tuned XGB params from v2
- Train with 8000 iterations, seed averaging (3 seeds)
- Combine with already trained LGB and CAT models
- Generate new submission
"""

import sys
sys.path.insert(0, 'src')

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from preprocessing import HousePricePreprocessor

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def load_data():
    """Load data using the saved preprocessor."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
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
    
    print(f"Training samples: {len(y_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")
    
    return X_train, y_train, X_test, test


def train_xgboost_with_seeds(X, y, X_test, xgb_params):
    """Train XGBoost with seed averaging."""
    print("\n" + "=" * 70)
    print("TRAINING TUNED XGBOOST (8000 iterations, 3 seeds)")
    print("=" * 70)
    
    XGB_SEEDS = [42, 123, 456]
    N_FOLDS = 5
    EARLY_STOPPING = 100
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    oof_xgb = np.zeros(len(y))
    test_xgb = np.zeros(len(X_test))
    xgb_models = []
    
    for seed in XGB_SEEDS:
        print(f"\n--- Seed {seed} ---")
        seed_oof = np.zeros(len(y))
        seed_test = np.zeros(len(X_test))
        seed_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                n_estimators=8000,
                early_stopping_rounds=EARLY_STOPPING,
                verbosity=0,
                random_state=seed,
                **xgb_params
            )
            
            model.fit(
                X_tr, np.log1p(y_tr),
                eval_set=[(X_val, np.log1p(y_val))],
                verbose=False
            )
            
            seed_oof[val_idx] = np.expm1(model.predict(X_val))
            
            # Train on full data with best iteration
            model_full = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                n_estimators=model.best_iteration,
                verbosity=0,
                random_state=seed,
                **xgb_params
            )
            model_full.fit(X, np.log1p(y), verbose=False)
            seed_test += np.expm1(model_full.predict(X_test)) / N_FOLDS
            seed_models.append(model_full)
            
            print(f"  Fold {fold+1}: best_iter={model.best_iteration}")
        
        cv = rmse(y, seed_oof)
        print(f"  Seed {seed} CV: {cv:.2f}")
        
        oof_xgb += seed_oof / len(XGB_SEEDS)
        test_xgb += seed_test / len(XGB_SEEDS)
        xgb_models.extend(seed_models)
    
    final_cv = rmse(y, oof_xgb)
    print(f"\nXGBoost Final CV (seed averaged): {final_cv:.2f}")
    
    return oof_xgb, test_xgb, xgb_models


def main():
    # Load data
    X_train, y_train, X_test, test_df = load_data()
    
    # Load tuned XGB params
    tuned_params = joblib.load('models/tuned_params_v2.pkl')
    xgb_params = tuned_params['xgb']
    print(f"\nUsing tuned XGB params: {xgb_params}")
    
    # Train XGBoost
    oof_xgb, test_xgb, xgb_models = train_xgboost_with_seeds(
        X_train, y_train, X_test, xgb_params
    )
    
    # Load existing LGB and CAT OOF predictions
    print("\n" + "=" * 70)
    print("LOADING EXISTING LGB AND CAT PREDICTIONS")
    print("=" * 70)
    
    oof_data = joblib.load('models/checkpoints/oof_predictions.pkl')
    oof_lgb = oof_data['lgb']
    oof_cat = oof_data['cat']
    
    print(f"LGB CV: {rmse(y_train, oof_lgb):.2f}")
    print(f"CAT CV: {rmse(y_train, oof_cat):.2f}")
    print(f"New XGB CV: {rmse(y_train, oof_xgb):.2f}")
    
    # Load existing test predictions from LGB and CAT
    lgb_models = joblib.load('models/checkpoints/lgb_final_models.pkl')
    cat_models = joblib.load('models/checkpoints/cat_final_models.pkl')
    
    # Generate LGB test predictions
    test_lgb = np.mean([np.expm1(m.predict(X_test)) for m in lgb_models], axis=0)
    
    # Generate CAT test predictions
    test_cat = np.mean([np.expm1(m.predict(X_test)) for m in cat_models], axis=0)
    
    print(f"\nTest prediction ranges:")
    print(f"  LGB: [{test_lgb.min():.0f}, {test_lgb.max():.0f}]")
    print(f"  CAT: [{test_cat.min():.0f}, {test_cat.max():.0f}]")
    print(f"  XGB: [{test_xgb.min():.0f}, {test_xgb.max():.0f}]")
    
    # Find optimal weights with new XGB
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL ENSEMBLE WEIGHTS")
    print("=" * 70)
    
    from scipy.optimize import minimize
    
    def ensemble_rmse(weights):
        w = weights / weights.sum()
        pred = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
        return rmse(y_train, pred)
    
    result = minimize(
        ensemble_rmse,
        x0=[0.33, 0.33, 0.34],
        bounds=[(0, 1), (0, 1), (0, 1)],
        method='L-BFGS-B'
    )
    
    weights = result.x / result.x.sum()
    print(f"Optimal weights: LGB={weights[0]:.3f}, CAT={weights[1]:.3f}, XGB={weights[2]:.3f}")
    
    # Final ensemble
    oof_ensemble = weights[0] * oof_lgb + weights[1] * oof_cat + weights[2] * oof_xgb
    test_ensemble = weights[0] * test_lgb + weights[1] * test_cat + weights[2] * test_xgb
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"LGB CV: {rmse(y_train, oof_lgb):.2f}")
    print(f"CAT CV: {rmse(y_train, oof_cat):.2f}")
    print(f"XGB CV (NEW tuned): {rmse(y_train, oof_xgb):.2f}")
    print(f"Ensemble CV: {rmse(y_train, oof_ensemble):.2f}")
    
    # Compare with old ensemble
    old_xgb = oof_data['xgb']
    old_ensemble = weights[0] * oof_lgb + weights[1] * oof_cat + weights[2] * old_xgb
    print(f"\nOld XGB CV: {rmse(y_train, old_xgb):.2f}")
    print(f"Old Ensemble CV: {rmse(y_train, old_ensemble):.2f}")
    
    # Save submission
    submission = pd.DataFrame({
        'id': test_df['_id'],
        'price': test_ensemble
    })
    submission.to_csv('outputs/predictions/submission_tuned_xgb.csv', index=False)
    print(f"\nSubmission saved: outputs/predictions/submission_tuned_xgb.csv")
    
    # Save new XGB models and OOF
    joblib.dump(xgb_models, 'models/checkpoints/xgb_tuned_models.pkl')
    joblib.dump({
        'lgb': oof_lgb,
        'cat': oof_cat,
        'xgb': oof_xgb,
        'y_true': y_train
    }, 'models/checkpoints/oof_predictions_with_tuned_xgb.pkl')
    print("Saved tuned XGB models and OOF predictions")


if __name__ == "__main__":
    main()
