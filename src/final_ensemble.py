"""
Final Ensemble Training with Seed Averaging
============================================
- LightGBM: 5 seeds × 10,000 iterations
- CatBoost: 5 seeds × 10,000 iterations  
- XGBoost: 3 seeds × 8,000 iterations
- Frequent checkpoints to avoid data loss
- Optimal weight finding via Optuna
"""

import sys
sys.path.insert(0, 'src')

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
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
LGB_SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds
CAT_SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds
XGB_SEEDS = [42, 123, 456]  # 3 seeds

LGB_ITERATIONS = 10000
CAT_ITERATIONS = 10000
XGB_ITERATIONS = 8000

EARLY_STOPPING = 100
N_FOLDS = 5

CHECKPOINT_DIR = 'models/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA & PREPROCESSOR
# ============================================================
def load_data():
    """Load and preprocess data using SAVED preprocessor"""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    # Use the saved preprocessor (contains target encoding stats)
    preprocessor = joblib.load('models/fine_tuned_preprocessor.pkl')
    
    # Load training data
    train = pd.read_csv('data/binaaz_train.csv')
    test = pd.read_csv('data/binaaz_test.csv')
    
    y_full = train['price']
    
    # Remove outliers (1st and 99th percentile)
    q_low = y_full.quantile(0.01)
    q_high = y_full.quantile(0.99)
    mask = (y_full >= q_low) & (y_full <= q_high)
    train_filtered = train[mask].reset_index(drop=True)
    
    # Transform data
    X_train = preprocessor.extract_features(train_filtered)
    X_train = preprocessor.transform(X_train)
    y_train = train_filtered['price'].values
    
    X_test = preprocessor.extract_features(test)
    X_test = preprocessor.transform(X_test)
    
    print(f"  Training samples: {len(y_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {X_train.shape[1]}")
    
    return X_train, y_train, X_test, test, preprocessor


# ============================================================
# PROGRESS BAR UTILITIES
# ============================================================
def progress_bar(current, total, width=40, prefix='', suffix='', eta_min=None):
    """Display a nice progress bar"""
    filled = int(width * current / total)
    bar = '█' * filled + '░' * (width - filled)
    percent = 100 * current / total
    eta_str = f"ETA: {eta_min:.1f}min" if eta_min is not None else ""
    print(f"\r{prefix} |{bar}| {percent:5.1f}% {suffix} {eta_str}", end='', flush=True)


def format_time(seconds):
    """Format seconds to mm:ss"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


# ============================================================
# MODEL TRAINING FUNCTIONS
# ============================================================
def train_lightgbm_seed(X, y, seed, params, fold_idx=None):
    """Train a single LightGBM model with given seed"""
    model_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': LGB_ITERATIONS,
        'learning_rate': params.get('learning_rate', 0.03),
        'num_leaves': params.get('num_leaves', 197),
        'max_depth': params.get('max_depth', 10),
        'min_child_samples': params.get('min_child_samples', 11),
        'subsample': params.get('subsample', 0.89),
        'colsample_bytree': params.get('colsample_bytree', 0.74),
        'reg_alpha': params.get('reg_alpha', 0.81),
        'reg_lambda': params.get('reg_lambda', 4.55),
        'random_state': seed,
        'force_col_wise': True
    }
    
    model = lgb.LGBMRegressor(**model_params)
    
    if fold_idx is not None:
        # Training on fold
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        for i, (train_idx, val_idx) in enumerate(kf.split(X)):
            if i == fold_idx:
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                model.fit(X_tr, np.log1p(y_tr),
                         eval_set=[(X_val, np.log1p(y_val))],
                         callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                                   lgb.log_evaluation(0)])
                return model
    else:
        # Training on full data
        model.fit(X, np.log1p(y))
        return model


def train_catboost_seed(X, y, seed, params, fold_idx=None):
    """Train a single CatBoost model with given seed"""
    model_params = {
        'iterations': CAT_ITERATIONS,
        'learning_rate': params.get('learning_rate', 0.03),
        'depth': params.get('depth', 8),
        'l2_leaf_reg': params.get('l2_leaf_reg', 3.0),
        'bagging_temperature': params.get('bagging_temperature', 0.5),
        'random_strength': params.get('random_strength', 0.5),
        'border_count': params.get('border_count', 128),
        'random_seed': seed,
        'verbose': False,
        'loss_function': 'RMSE',
        'early_stopping_rounds': EARLY_STOPPING
    }
    
    model = CatBoostRegressor(**model_params)
    
    if fold_idx is not None:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        for i, (train_idx, val_idx) in enumerate(kf.split(X)):
            if i == fold_idx:
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                model.fit(X_tr, np.log1p(y_tr),
                         eval_set=(X_val, np.log1p(y_val)),
                         verbose=False)
                return model
    else:
        model.fit(X, np.log1p(y), verbose=False)
        return model


def train_xgboost_seed(X, y, seed, params, fold_idx=None):
    """Train a single XGBoost model with given seed"""
    model_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': XGB_ITERATIONS,
        'learning_rate': params.get('learning_rate', 0.02),
        'max_depth': params.get('max_depth', 8),
        'min_child_weight': params.get('min_child_weight', 10),
        'subsample': params.get('subsample', 0.8),
        'colsample_bytree': params.get('colsample_bytree', 0.7),
        'reg_alpha': params.get('reg_alpha', 0.5),
        'reg_lambda': params.get('reg_lambda', 2.0),
        'random_state': seed,
        'verbosity': 0
    }
    
    # Only add early stopping when we have validation data (fold_idx is not None)
    if fold_idx is not None:
        model_params['early_stopping_rounds'] = EARLY_STOPPING
    
    model = xgb.XGBRegressor(**model_params)
    
    if fold_idx is not None:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        for i, (train_idx, val_idx) in enumerate(kf.split(X)):
            if i == fold_idx:
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                model.fit(X_tr, np.log1p(y_tr),
                         eval_set=[(X_val, np.log1p(y_val))],
                         verbose=False)
                return model
    else:
        # Training on full data - no early stopping
        model.fit(X, np.log1p(y))
        return model


# ============================================================
# CROSS-VALIDATION WITH SEED AVERAGING
# ============================================================
def cv_with_seed_averaging(X, y, model_type, seeds, params):
    """
    Perform CV with seed averaging.
    Returns: OOF predictions, list of trained models for each seed
    """
    n_samples = len(y)
    n_seeds = len(seeds)
    
    # OOF predictions for each seed
    oof_preds_all = np.zeros((n_seeds, n_samples))
    all_models = []
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_indices = list(kf.split(X))
    
    total_tasks = n_seeds * N_FOLDS
    completed = 0
    start_time = datetime.now()
    
    print(f"\n  Training {model_type.upper()} with {n_seeds} seeds × {N_FOLDS} folds...")
    
    for seed_idx, seed in enumerate(seeds):
        seed_models = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            # Train model
            if model_type == 'lgb':
                model = train_lightgbm_seed(X, y, seed, params, fold_idx)
            elif model_type == 'cat':
                model = train_catboost_seed(X, y, seed, params, fold_idx)
            else:  # xgb
                model = train_xgboost_seed(X, y, seed, params, fold_idx)
            
            seed_models.append(model)
            
            # Predict on validation fold
            X_val = X[val_idx]
            oof_preds_all[seed_idx, val_idx] = np.expm1(model.predict(X_val))
            
            completed += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            eta = (elapsed / completed) * (total_tasks - completed) / 60
            
            progress_bar(completed, total_tasks, prefix=f"  {model_type.upper()}", 
                        suffix=f"Seed {seed_idx+1}/{n_seeds} Fold {fold_idx+1}/{N_FOLDS}", 
                        eta_min=eta)
        
        all_models.append(seed_models)
        
        # Checkpoint after each seed
        checkpoint = {
            'model_type': model_type,
            'seed': seed,
            'seed_idx': seed_idx,
            'models': seed_models,
            'oof_preds': oof_preds_all[seed_idx]
        }
        joblib.dump(checkpoint, f"{CHECKPOINT_DIR}/{model_type}_seed{seed}.pkl")
    
    print()  # New line after progress bar
    
    # Average OOF predictions across seeds
    oof_avg = np.mean(oof_preds_all, axis=0)
    rmse = np.sqrt(mean_squared_error(y, oof_avg))
    print(f"  {model_type.upper()} Seed-Averaged OOF RMSE: {rmse:,.2f}")
    
    return oof_avg, all_models, oof_preds_all


# ============================================================
# TRAIN FINAL MODELS ON FULL DATA
# ============================================================
def train_final_models(X, y, model_type, seeds, params):
    """Train final models on full data for each seed"""
    models = []
    
    print(f"\n  Training final {model_type.upper()} models on full data...")
    start_time = datetime.now()
    
    for i, seed in enumerate(seeds):
        if model_type == 'lgb':
            model = train_lightgbm_seed(X, y, seed, params)
        elif model_type == 'cat':
            model = train_catboost_seed(X, y, seed, params)
        else:
            model = train_xgboost_seed(X, y, seed, params)
        
        models.append(model)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        eta = (elapsed / (i+1)) * (len(seeds) - i - 1) / 60
        progress_bar(i+1, len(seeds), prefix=f"  {model_type.upper()}", 
                    suffix=f"Seed {i+1}/{len(seeds)}", eta_min=eta)
    
    print()
    
    # Save checkpoint
    joblib.dump(models, f"{CHECKPOINT_DIR}/{model_type}_final_models.pkl")
    
    return models


# ============================================================
# FIND OPTIMAL WEIGHTS
# ============================================================
def find_optimal_weights(oof_lgb, oof_cat, oof_xgb, y_true):
    """Use Optuna to find optimal ensemble weights"""
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL ENSEMBLE WEIGHTS")
    print("=" * 70)
    
    def objective(trial):
        w_lgb = trial.suggest_float('w_lgb', 0, 1)
        w_cat = trial.suggest_float('w_cat', 0, 1)
        w_xgb = trial.suggest_float('w_xgb', 0, 1)
        
        # Normalize weights
        total = w_lgb + w_cat + w_xgb
        w_lgb, w_cat, w_xgb = w_lgb/total, w_cat/total, w_xgb/total
        
        # Blend predictions
        pred = w_lgb * oof_lgb + w_cat * oof_cat + w_xgb * oof_xgb
        return np.sqrt(mean_squared_error(y_true, pred))
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=200, show_progress_bar=False)
    
    # Get normalized weights
    w_lgb = study.best_params['w_lgb']
    w_cat = study.best_params['w_cat']
    w_xgb = study.best_params['w_xgb']
    total = w_lgb + w_cat + w_xgb
    
    weights = {
        'lgb': w_lgb / total,
        'cat': w_cat / total,
        'xgb': w_xgb / total
    }
    
    print(f"\n  Optimal Weights:")
    print(f"    LightGBM: {weights['lgb']*100:.1f}%")
    print(f"    CatBoost: {weights['cat']*100:.1f}%")
    print(f"    XGBoost:  {weights['xgb']*100:.1f}%")
    print(f"\n  Ensemble CV RMSE: {study.best_value:,.2f}")
    
    return weights, study.best_value


# ============================================================
# GENERATE PREDICTIONS
# ============================================================
def generate_predictions(X_test, lgb_models, cat_models, xgb_models, weights):
    """Generate seed-averaged predictions for test set"""
    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)
    
    # LightGBM predictions (average across all seeds)
    lgb_preds = np.zeros(len(X_test))
    for models in lgb_models:
        lgb_preds += np.expm1(models.predict(X_test))
    lgb_preds /= len(lgb_models)
    print(f"  LightGBM predictions: mean={lgb_preds.mean():,.0f}")
    
    # CatBoost predictions
    cat_preds = np.zeros(len(X_test))
    for models in cat_models:
        cat_preds += np.expm1(models.predict(X_test))
    cat_preds /= len(cat_models)
    print(f"  CatBoost predictions: mean={cat_preds.mean():,.0f}")
    
    # XGBoost predictions
    xgb_preds = np.zeros(len(X_test))
    for models in xgb_models:
        xgb_preds += np.expm1(models.predict(X_test))
    xgb_preds /= len(xgb_models)
    print(f"  XGBoost predictions: mean={xgb_preds.mean():,.0f}")
    
    # Weighted ensemble
    final_preds = (weights['lgb'] * lgb_preds + 
                   weights['cat'] * cat_preds + 
                   weights['xgb'] * xgb_preds)
    
    # Post-processing: clip extreme values
    p1, p99 = np.percentile(final_preds, [1, 99])
    final_preds = np.clip(final_preds, p1 * 0.8, p99 * 1.2)
    
    print(f"\n  Final predictions: mean={final_preds.mean():,.0f}, std={final_preds.std():,.0f}")
    
    return final_preds


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("   FINAL ENSEMBLE TRAINING WITH SEED AVERAGING")
    print("=" * 70)
    print(f"  LightGBM: {len(LGB_SEEDS)} seeds × {LGB_ITERATIONS:,} iterations")
    print(f"  CatBoost: {len(CAT_SEEDS)} seeds × {CAT_ITERATIONS:,} iterations")
    print(f"  XGBoost:  {len(XGB_SEEDS)} seeds × {XGB_ITERATIONS:,} iterations")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # Load data
    X_train, y_train, X_test, test_df, preprocessor = load_data()
    
    # Load best parameters from tuning
    lgb_params = joblib.load('models/lgb_best_params.pkl')
    cat_study = joblib.load('models/study_cat_fast.pkl')
    cat_params = cat_study.best_params
    xgb_study = joblib.load('models/study_xgb_power.pkl')
    xgb_params = xgb_study.best_params
    
    print("\n" + "=" * 70)
    print("BEST PARAMETERS FROM TUNING")
    print("=" * 70)
    print(f"  LightGBM: lr={lgb_params.get('learning_rate', 0.03):.4f}, "
          f"leaves={lgb_params.get('num_leaves', 197)}")
    print(f"  CatBoost: lr={cat_params.get('learning_rate', 0.03):.4f}, "
          f"depth={cat_params.get('depth', 8)}")
    print(f"  XGBoost:  lr={xgb_params.get('learning_rate', 0.02):.4f}, "
          f"depth={xgb_params.get('max_depth', 8)}")
    
    # ============================================================
    # PHASE 1: Cross-Validation with Seed Averaging
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: CROSS-VALIDATION (OOF Predictions)")
    print("=" * 70)
    
    oof_lgb, lgb_cv_models, lgb_oof_all = cv_with_seed_averaging(
        X_train, y_train, 'lgb', LGB_SEEDS, lgb_params)
    
    oof_cat, cat_cv_models, cat_oof_all = cv_with_seed_averaging(
        X_train, y_train, 'cat', CAT_SEEDS, cat_params)
    
    oof_xgb, xgb_cv_models, xgb_oof_all = cv_with_seed_averaging(
        X_train, y_train, 'xgb', XGB_SEEDS, xgb_params)
    
    # Save OOF predictions
    oof_data = {
        'lgb': oof_lgb, 'cat': oof_cat, 'xgb': oof_xgb,
        'lgb_all': lgb_oof_all, 'cat_all': cat_oof_all, 'xgb_all': xgb_oof_all,
        'y_true': y_train
    }
    joblib.dump(oof_data, f"{CHECKPOINT_DIR}/oof_predictions.pkl")
    
    # Find optimal weights
    weights, ensemble_cv = find_optimal_weights(oof_lgb, oof_cat, oof_xgb, y_train)
    
    # ============================================================
    # PHASE 2: Train Final Models on Full Data
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: TRAINING FINAL MODELS ON FULL DATA")
    print("=" * 70)
    
    lgb_final_models = train_final_models(X_train, y_train, 'lgb', LGB_SEEDS, lgb_params)
    cat_final_models = train_final_models(X_train, y_train, 'cat', CAT_SEEDS, cat_params)
    xgb_final_models = train_final_models(X_train, y_train, 'xgb', XGB_SEEDS, xgb_params)
    
    # Save all final models
    final_data = {
        'lgb_models': lgb_final_models,
        'cat_models': cat_final_models,
        'xgb_models': xgb_final_models,
        'weights': weights,
        'ensemble_cv': ensemble_cv,
        'preprocessor': preprocessor
    }
    joblib.dump(final_data, 'models/final_ensemble.pkl')
    print("\n  Saved models to models/final_ensemble.pkl")
    
    # ============================================================
    # PHASE 3: Generate Predictions
    # ============================================================
    final_preds = generate_predictions(X_test, lgb_final_models, cat_final_models, 
                                       xgb_final_models, weights)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'price': final_preds
    })
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_path = f'outputs/predictions/submission_final_{timestamp}.csv'
    submission.to_csv(submission_path, index=False)
    
    # Also save as latest
    submission.to_csv('outputs/predictions/submission_latest.csv', index=False)
    
    # ============================================================
    # SUMMARY
    # ============================================================
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    
    print("\n" + "=" * 70)
    print("   TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f} minutes")
    print(f"\n  Individual Model CVs (seed-averaged):")
    print(f"    LightGBM: {np.sqrt(mean_squared_error(y_train, oof_lgb)):,.2f}")
    print(f"    CatBoost: {np.sqrt(mean_squared_error(y_train, oof_cat)):,.2f}")
    print(f"    XGBoost:  {np.sqrt(mean_squared_error(y_train, oof_xgb)):,.2f}")
    print(f"\n  Ensemble CV RMSE: {ensemble_cv:,.2f}")
    print(f"  Estimated Leaderboard: {ensemble_cv * 0.81:,.0f}")
    print(f"\n  Submission saved to: {submission_path}")
    print("=" * 70)
    
    return ensemble_cv


if __name__ == '__main__':
    main()
