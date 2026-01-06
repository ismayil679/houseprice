"""
Generate Submission from Saved Models
=====================================
Use the EXACT same models that achieved LB 29,318
No new tuning - just regenerate predictions
"""

import sys
sys.path.insert(0, 'src')

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from preprocessing import HousePricePreprocessor

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def main():
    print("=" * 70)
    print("REGENERATING SUBMISSION FROM SAVED MODELS (LB 29,318)")
    print("=" * 70)
    
    # Load preprocessor
    preprocessor = joblib.load('models/fine_tuned_preprocessor.pkl')
    
    # Load data
    train = pd.read_csv('data/binaaz_train.csv')
    test = pd.read_csv('data/binaaz_test.csv')
    
    # Extract features
    X_train_df = preprocessor.extract_features(train)
    X_test_df = preprocessor.extract_features(test)
    
    # Remove outliers (same as original)
    y_full = train['price']
    q_low = y_full.quantile(0.01)
    q_high = y_full.quantile(0.99)
    mask = (y_full >= q_low) & (y_full <= q_high)
    
    X_train_df = X_train_df[mask].reset_index(drop=True)
    y_train = train[mask]['price'].values
    
    # Transform
    X_train = preprocessor.transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    
    print(f"Train: {X_train.shape}")
    print(f"Test: {X_test.shape}")
    
    # Load saved models
    print("\nLoading saved models...")
    lgb_models = joblib.load('models/checkpoints/lgb_final_models.pkl')
    cat_models = joblib.load('models/checkpoints/cat_final_models.pkl')
    xgb_models = joblib.load('models/checkpoints/xgb_final_models.pkl')
    
    print(f"LGB models: {len(lgb_models)}")
    print(f"CAT models: {len(cat_models)}")
    print(f"XGB models: {len(xgb_models)}")
    
    # Load OOF
    oof = joblib.load('models/checkpoints/oof_predictions.pkl')
    lgb_oof = oof['lgb']
    cat_oof = oof['cat']
    xgb_oof = oof['xgb']
    y = oof['y_true']
    
    print(f"\nOOF CVs:")
    print(f"  LGB: {rmse(y, lgb_oof):.2f}")
    print(f"  CAT: {rmse(y, cat_oof):.2f}")
    print(f"  XGB: {rmse(y, xgb_oof):.2f}")
    
    # Generate test predictions
    print("\nGenerating test predictions...")
    
    # LGB
    test_lgb = np.zeros(len(X_test))
    for model in lgb_models:
        test_lgb += np.expm1(model.predict(X_test)) / len(lgb_models)
    print(f"LGB test mean: {test_lgb.mean():.2f}")
    
    # CAT
    test_cat = np.zeros(len(X_test))
    for model in cat_models:
        test_cat += np.expm1(model.predict(X_test)) / len(cat_models)
    print(f"CAT test mean: {test_cat.mean():.2f}")
    
    # XGB
    test_xgb = np.zeros(len(X_test))
    for model in xgb_models:
        test_xgb += np.expm1(model.predict(X_test)) / len(xgb_models)
    print(f"XGB test mean: {test_xgb.mean():.2f}")
    
    # Find optimal weights (on OOF)
    from scipy.optimize import minimize
    
    def ensemble_rmse(weights):
        w = weights / weights.sum()
        pred = w[0] * lgb_oof + w[1] * cat_oof + w[2] * xgb_oof
        return rmse(y, pred)
    
    result = minimize(
        ensemble_rmse,
        x0=[0.33, 0.33, 0.34],
        bounds=[(0, 1), (0, 1), (0, 1)],
        method='L-BFGS-B'
    )
    
    weights = result.x / result.x.sum()
    print(f"\nOptimal weights: LGB={weights[0]:.3f}, CAT={weights[1]:.3f}, XGB={weights[2]:.3f}")
    
    # Ensemble
    oof_ensemble = weights[0] * lgb_oof + weights[1] * cat_oof + weights[2] * xgb_oof
    test_ensemble = weights[0] * test_lgb + weights[1] * test_cat + weights[2] * test_xgb
    
    print(f"Ensemble CV: {rmse(y, oof_ensemble):.2f}")
    print(f"Test ensemble mean: {test_ensemble.mean():.2f}")
    
    # Save
    submission = pd.DataFrame({
        'id': test['_id'],
        'price': test_ensemble
    })
    submission.to_csv('outputs/predictions/submission_from_saved_models.csv', index=False)
    print(f"\nSaved: outputs/predictions/submission_from_saved_models.csv")
    
    # Compare with previous submission
    old_sub = pd.read_csv('outputs/predictions/submission_latest.csv')
    print(f"\nComparison with submission_latest (LB 29,318):")
    print(f"  Correlation: {np.corrcoef(submission['price'], old_sub['price'])[0,1]:.6f}")
    print(f"  Mean diff: {np.abs(submission['price'] - old_sub['price']).mean():.2f}")


if __name__ == "__main__":
    main()
