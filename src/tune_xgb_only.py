"""
XGBoost Fast Tuning + Combine with Existing LGB/CAT
====================================================
We have:
- LGB: 5 seeds trained, OOF ready
- CAT: 5 seeds trained, OOF ready
- XGB: 3 seeds trained BUT we want to tune it properly

Strategy:
1. Fast tune XGBoost (15 trials)
2. Retrain XGBoost with tuned params
3. Combine with existing LGB and CAT predictions
4. Find optimal weights
5. Generate submission
"""

import sys
sys.path.insert(0, 'src')

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from preprocessing import HousePricePreprocessor

# ============================================================
# CONFIGURATION
# ============================================================
N_FOLDS = 5
TUNING_TRIALS = 50  # More trials for XGB
EARLY_STOPPING = 100
XGB_SEEDS = [42, 123, 456]

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# ============================================================
# LOAD EXISTING DATA
# ============================================================
def load_data_and_existing_oof():
    """Load data and existing OOF predictions."""
    print("=" * 70)
    print("LOADING DATA AND EXISTING MODELS")
    print("=" * 70)
    
    # Load existing OOF
    oof = joblib.load('models/checkpoints/oof_predictions.pkl')
    print(f"Existing OOF loaded: {list(oof.keys())}")
    
    y_train = oof['y_true']
    lgb_oof = oof['lgb']  # Key is 'lgb' not 'lgb_oof'
    cat_oof = oof['cat']  # Key is 'cat' not 'cat_oof'
    xgb_oof_old = oof['xgb']  # Key is 'xgb' not 'xgb_oof'
    
    print(f"LGB OOF CV: {rmse(y_train, lgb_oof):.2f}")
    print(f"CAT OOF CV: {rmse(y_train, cat_oof):.2f}")
    print(f"XGB OOF CV (old): {rmse(y_train, xgb_oof_old):.2f}")
    
    # Load preprocessor and data
    preprocessor = joblib.load('models/fine_tuned_preprocessor.pkl')
    
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
    
    train_filtered = train[mask].reset_index(drop=True)
    X_train_df = X_train_df[mask].reset_index(drop=True)
    y_train_data = train_filtered['price'].values
    
    # Transform
    X_train = preprocessor.transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    
    print(f"\nTraining samples: {len(y_train_data):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")
    
    return X_train, y_train_data, X_test, test, lgb_oof, cat_oof, preprocessor


# ============================================================
# TUNE XGBOOST
# ============================================================
def tune_xgboost(X, y, n_trials=50):
    """Fast tune XGBoost."""
    print("\n" + "=" * 70)
    print(f"TUNING XGBOOST ({n_trials} TRIALS)")
    print("=" * 70)
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 3000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'random_state': 42,
            'verbosity': 0,
            'tree_method': 'hist',
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
    
    print(f"\nBest XGB CV: {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")
    
    return study.best_params


# ============================================================
# TRAIN XGBOOST WITH SEED AVERAGING
# ============================================================
def train_xgboost_seeds(X, y, X_test, params):
    """Train XGBoost with multiple seeds."""
    print("\n" + "=" * 70)
    print("TRAINING XGBOOST WITH SEED AVERAGING")
    print("=" * 70)
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    oof_xgb = np.zeros(len(y))
    test_xgb = np.zeros(len(X_test))
    
    xgb_models = []
    
    for seed in XGB_SEEDS:
        seed_oof = np.zeros(len(y))
        seed_test = np.zeros(len(X_test))
        seed_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                n_estimators=10000,
                random_state=seed,
                verbosity=0,
                tree_method='hist',
                early_stopping_rounds=EARLY_STOPPING,
                **params
            )
            model.fit(X_tr, np.log1p(y_tr),
                     eval_set=[(X_val, np.log1p(y_val))],
                     verbose=False)
            
            seed_oof[val_idx] = np.expm1(model.predict(X_val))
            seed_models.append(model)
            
            # Full data model for test
            model_full = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                n_estimators=model.best_iteration,
                random_state=seed,
                verbosity=0,
                tree_method='hist',
                **params
            )
            model_full.fit(X, np.log1p(y), verbose=False)
            seed_test += np.expm1(model_full.predict(X_test)) / N_FOLDS
        
        oof_xgb += seed_oof / len(XGB_SEEDS)
        test_xgb += seed_test / len(XGB_SEEDS)
        xgb_models.append(seed_models)
        
        print(f"  Seed {seed}: CV = {rmse(y, seed_oof):.2f}")
    
    print(f"\nXGB Final CV: {rmse(y, oof_xgb):.2f}")
    
    return oof_xgb, test_xgb, xgb_models


# ============================================================
# GENERATE LGB/CAT TEST PREDICTIONS FROM SAVED MODELS
# ============================================================
def load_existing_test_predictions(X_test):
    """Load saved LGB and CAT models and generate test predictions."""
    print("\n" + "=" * 70)
    print("GENERATING LGB/CAT TEST PREDICTIONS FROM SAVED MODELS")
    print("=" * 70)
    
    # Load saved models
    lgb_models = joblib.load('models/checkpoints/lgb_final_models.pkl')
    cat_models = joblib.load('models/checkpoints/cat_final_models.pkl')
    
    print(f"Loaded {len(lgb_models)} LGB model sets")
    print(f"Loaded {len(cat_models)} CAT model sets")
    
    # Generate LGB predictions
    test_lgb = np.zeros(len(X_test))
    for seed_models in lgb_models:
        seed_pred = np.zeros(len(X_test))
        for model in seed_models:
            seed_pred += np.expm1(model.predict(X_test)) / len(seed_models)
        test_lgb += seed_pred / len(lgb_models)
    
    print(f"LGB test predictions: mean={test_lgb.mean():.2f}")
    
    # Generate CAT predictions
    test_cat = np.zeros(len(X_test))
    for seed_models in cat_models:
        seed_pred = np.zeros(len(X_test))
        for model in seed_models:
            seed_pred += np.expm1(model.predict(X_test)) / len(seed_models)
        test_cat += seed_pred / len(cat_models)
    
    print(f"CAT test predictions: mean={test_cat.mean():.2f}")
    
    return test_lgb, test_cat


# ============================================================
# FIND OPTIMAL WEIGHTS
# ============================================================
def find_optimal_weights(y, lgb_oof, cat_oof, xgb_oof):
    """Find optimal ensemble weights."""
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
    return weights


# ============================================================
# MAIN
# ============================================================
def main():
    # Load data and existing OOF
    X_train, y_train, X_test, test_df, lgb_oof, cat_oof, preprocessor = load_data_and_existing_oof()
    
    # Tune XGBoost
    xgb_params = tune_xgboost(X_train, y_train, n_trials=TUNING_TRIALS)
    
    # Train XGBoost with tuned params
    xgb_oof, test_xgb, xgb_models = train_xgboost_seeds(X_train, y_train, X_test, xgb_params)
    
    # Load existing LGB/CAT test predictions
    test_lgb, test_cat = load_existing_test_predictions(X_test)
    
    # Find optimal weights
    weights = find_optimal_weights(y_train, lgb_oof, cat_oof, xgb_oof)
    
    print(f"\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"LGB CV: {rmse(y_train, lgb_oof):.2f}")
    print(f"CAT CV: {rmse(y_train, cat_oof):.2f}")
    print(f"XGB CV (NEW): {rmse(y_train, xgb_oof):.2f}")
    print(f"\nWeights: LGB={weights[0]:.3f}, CAT={weights[1]:.3f}, XGB={weights[2]:.3f}")
    
    # Ensemble
    oof_ensemble = weights[0] * lgb_oof + weights[1] * cat_oof + weights[2] * xgb_oof
    test_ensemble = weights[0] * test_lgb + weights[1] * test_cat + weights[2] * test_xgb
    
    print(f"Ensemble CV: {rmse(y_train, oof_ensemble):.2f}")
    
    # Save submission
    submission = pd.DataFrame({
        'id': test_df['_id'],
        'price': test_ensemble
    })
    submission.to_csv('outputs/predictions/submission_xgb_tuned.csv', index=False)
    print(f"\nSubmission saved: outputs/predictions/submission_xgb_tuned.csv")
    
    # Save tuned XGB params
    joblib.dump(xgb_params, 'models/xgb_tuned_params.pkl')
    print("XGB params saved: models/xgb_tuned_params.pkl")
    
    # Update OOF with new XGB
    oof_updated = {
        'y_true': y_train,
        'lgb_oof': lgb_oof,
        'cat_oof': cat_oof,
        'xgb_oof': xgb_oof
    }
    joblib.dump(oof_updated, 'models/checkpoints/oof_predictions_xgb_tuned.pkl')
    

if __name__ == "__main__":
    main()
