"""
EXTENSIVE HYPERPARAMETER TUNING FOR ALL MODELS
==============================================
Target: Sub-29,000 LB, 36,000 Ensemble CV

Current Best (LB 29,318):
- LGB: learning_rate=0.03, num_leaves=197, max_depth=10, min_child_samples=11,
       subsample=0.89, colsample_bytree=0.74, reg_alpha=0.81, reg_lambda=4.55
- CAT: learning_rate=0.03, depth=8, l2_leaf_reg=3.0, bagging_temperature=0.5,
       random_strength=0.5, border_count=128
- XGB: learning_rate=0.02, max_depth=8, min_child_weight=10, subsample=0.8,
       colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0

Strategy:
1. Use Optuna with TPE sampler (best for hyperparameter optimization)
2. 50 trials per model with early pruning
3. Search around the current best parameters
4. Save checkpoints after each model
5. Final training with seed averaging
"""

import sys
sys.path.insert(0, 'src')

import os
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')

from preprocessing import HousePricePreprocessor

# ============================================================
# CONFIGURATION
# ============================================================
N_FOLDS = 5
N_TRIALS = 50  # Extensive tuning
EARLY_STOPPING = 100
CHECKPOINT_DIR = 'models/tuning_checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Current best parameters (baseline)
BEST_LGB_PARAMS = {
    'learning_rate': 0.03,
    'num_leaves': 197,
    'max_depth': 10,
    'min_child_samples': 11,
    'subsample': 0.89,
    'colsample_bytree': 0.74,
    'reg_alpha': 0.81,
    'reg_lambda': 4.55
}

BEST_CAT_PARAMS = {
    'learning_rate': 0.03,
    'depth': 8,
    'l2_leaf_reg': 3.0,
    'bagging_temperature': 0.5,
    'random_strength': 0.5,
    'border_count': 128
}

BEST_XGB_PARAMS = {
    'learning_rate': 0.02,
    'max_depth': 8,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0
}


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
    
    # Remove outliers
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
    
    return X_train, y_train, X_test, test, preprocessor


def save_checkpoint(name, data):
    """Save checkpoint with timestamp."""
    path = f"{CHECKPOINT_DIR}/{name}.pkl"
    joblib.dump(data, path)
    print(f"  [Checkpoint saved: {path}]")


# ============================================================
# LIGHTGBM TUNING
# ============================================================
def tune_lightgbm(X, y, n_trials=50):
    """Tune LightGBM extensively with Optuna."""
    print("\n" + "=" * 70)
    print("TUNING LIGHTGBM (50 trials)")
    print(f"Baseline CV: 37,880 | Target: < 37,000")
    print("=" * 70)
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': 5000,
            # Search around best values with wider range
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 6, 14),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),
            'random_state': 42,
            'force_col_wise': True
        }
        
        oof = np.zeros(len(y))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, np.log1p(y_tr),
                     eval_set=[(X_val, np.log1p(y_val))],
                     callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                               lgb.log_evaluation(0)])
            
            oof[val_idx] = np.expm1(model.predict(X_val))
            
            # Pruning check after each fold
            intermediate_value = rmse(y[:val_idx[-1]+1][oof[:val_idx[-1]+1] > 0], 
                                      oof[:val_idx[-1]+1][oof[:val_idx[-1]+1] > 0])
            trial.report(intermediate_value, fold_idx)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return rmse(y, oof)
    
    # Create study with TPE sampler and median pruner
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    
    # Add baseline as first trial
    study.enqueue_trial(BEST_LGB_PARAMS)
    
    study.optimize(
        objective, 
        n_trials=n_trials, 
        show_progress_bar=True,
        n_jobs=1  # Sequential for stability
    )
    
    print(f"\nBest LGB CV: {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")
    
    # Save checkpoint
    save_checkpoint('lgb_tuning', {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'all_trials': [(t.params, t.value) for t in study.trials if t.value is not None]
    })
    
    return study.best_params, study.best_value


# ============================================================
# CATBOOST TUNING
# ============================================================
def tune_catboost(X, y, n_trials=50):
    """Tune CatBoost extensively with Optuna."""
    print("\n" + "=" * 70)
    print("TUNING CATBOOST (50 trials)")
    print(f"Baseline CV: 37,621 | Target: < 37,000")
    print("=" * 70)
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    def objective(trial):
        params = {
            'iterations': 5000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'depth': trial.suggest_int('depth', 6, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 64, 255),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
            'random_seed': 42,
            'verbose': False,
            'loss_function': 'RMSE',
            'early_stopping_rounds': EARLY_STOPPING
        }
        
        oof = np.zeros(len(y))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = CatBoostRegressor(**params)
            model.fit(X_tr, np.log1p(y_tr),
                     eval_set=(X_val, np.log1p(y_val)),
                     verbose=False)
            
            oof[val_idx] = np.expm1(model.predict(X_val))
            
            # Pruning
            trial.report(rmse(y[val_idx], oof[val_idx]), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return rmse(y, oof)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    
    # Add baseline
    study.enqueue_trial(BEST_CAT_PARAMS)
    
    study.optimize(
        objective, 
        n_trials=n_trials, 
        show_progress_bar=True,
        n_jobs=1
    )
    
    print(f"\nBest CAT CV: {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")
    
    save_checkpoint('cat_tuning', {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'all_trials': [(t.params, t.value) for t in study.trials if t.value is not None]
    })
    
    return study.best_params, study.best_value


# ============================================================
# XGBOOST TUNING
# ============================================================
def tune_xgboost(X, y, n_trials=50):
    """Tune XGBoost extensively with Optuna."""
    print("\n" + "=" * 70)
    print("TUNING XGBOOST (50 trials)")
    print(f"Baseline CV: 37,492 | Target: < 37,000")
    print("=" * 70)
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 5000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.06, log=True),
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'random_state': 42,
            'verbosity': 0,
            'early_stopping_rounds': EARLY_STOPPING
        }
        
        oof = np.zeros(len(y))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, np.log1p(y_tr),
                     eval_set=[(X_val, np.log1p(y_val))],
                     verbose=False)
            
            oof[val_idx] = np.expm1(model.predict(X_val))
            
            # Pruning
            trial.report(rmse(y[val_idx], oof[val_idx]), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return rmse(y, oof)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    
    # Add baseline
    study.enqueue_trial(BEST_XGB_PARAMS)
    
    study.optimize(
        objective, 
        n_trials=n_trials, 
        show_progress_bar=True,
        n_jobs=1
    )
    
    print(f"\nBest XGB CV: {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")
    
    save_checkpoint('xgb_tuning', {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'all_trials': [(t.params, t.value) for t in study.trials if t.value is not None]
    })
    
    return study.best_params, study.best_value


# ============================================================
# FINAL TRAINING WITH SEED AVERAGING
# ============================================================
def train_final_models(X, y, X_test, lgb_params, cat_params, xgb_params):
    """Train all models with seed averaging using best params."""
    print("\n" + "=" * 70)
    print("FINAL TRAINING WITH SEED AVERAGING")
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
    
    lgb_models = []
    cat_models = []
    xgb_models = []
    
    # ==================== LightGBM ====================
    print("\n--- LightGBM Training ---")
    for seed_idx, seed in enumerate(LGB_SEEDS):
        seed_oof = np.zeros(len(y))
        seed_test = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            model = lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                verbosity=-1,
                n_estimators=10000,
                random_state=seed,
                force_col_wise=True,
                **lgb_params
            )
            model.fit(
                X[train_idx], np.log1p(y[train_idx]),
                eval_set=[(X[val_idx], np.log1p(y[val_idx]))],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
            )
            seed_oof[val_idx] = np.expm1(model.predict(X[val_idx]))
            
            # Full model
            model_full = lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                verbosity=-1,
                n_estimators=model.best_iteration_,
                random_state=seed,
                force_col_wise=True,
                **lgb_params
            )
            model_full.fit(X, np.log1p(y))
            seed_test += np.expm1(model_full.predict(X_test)) / N_FOLDS
            lgb_models.append(model_full)
        
        oof_lgb += seed_oof / len(LGB_SEEDS)
        test_lgb += seed_test / len(LGB_SEEDS)
        print(f"  Seed {seed}: CV = {rmse(y, seed_oof):.2f}")
    
    lgb_cv = rmse(y, oof_lgb)
    print(f"LightGBM Final CV: {lgb_cv:.2f}")
    save_checkpoint('lgb_final_models', lgb_models)
    
    # ==================== CatBoost ====================
    print("\n--- CatBoost Training ---")
    for seed_idx, seed in enumerate(CAT_SEEDS):
        seed_oof = np.zeros(len(y))
        seed_test = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            model = CatBoostRegressor(
                iterations=10000,
                random_seed=seed,
                verbose=False,
                loss_function='RMSE',
                early_stopping_rounds=100,
                **cat_params
            )
            model.fit(
                X[train_idx], np.log1p(y[train_idx]),
                eval_set=(X[val_idx], np.log1p(y[val_idx])),
                verbose=False
            )
            seed_oof[val_idx] = np.expm1(model.predict(X[val_idx]))
            
            model_full = CatBoostRegressor(
                iterations=model.best_iteration_,
                random_seed=seed,
                verbose=False,
                loss_function='RMSE',
                **cat_params
            )
            model_full.fit(X, np.log1p(y), verbose=False)
            seed_test += np.expm1(model_full.predict(X_test)) / N_FOLDS
            cat_models.append(model_full)
        
        oof_cat += seed_oof / len(CAT_SEEDS)
        test_cat += seed_test / len(CAT_SEEDS)
        print(f"  Seed {seed}: CV = {rmse(y, seed_oof):.2f}")
    
    cat_cv = rmse(y, oof_cat)
    print(f"CatBoost Final CV: {cat_cv:.2f}")
    save_checkpoint('cat_final_models', cat_models)
    
    # ==================== XGBoost ====================
    print("\n--- XGBoost Training ---")
    for seed_idx, seed in enumerate(XGB_SEEDS):
        seed_oof = np.zeros(len(y))
        seed_test = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                n_estimators=10000,
                random_state=seed,
                verbosity=0,
                early_stopping_rounds=100,
                **xgb_params
            )
            model.fit(
                X[train_idx], np.log1p(y[train_idx]),
                eval_set=[(X[val_idx], np.log1p(y[val_idx]))],
                verbose=False
            )
            seed_oof[val_idx] = np.expm1(model.predict(X[val_idx]))
            
            model_full = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                n_estimators=model.best_iteration,
                random_state=seed,
                verbosity=0,
                **xgb_params
            )
            model_full.fit(X, np.log1p(y), verbose=False)
            seed_test += np.expm1(model_full.predict(X_test)) / N_FOLDS
            xgb_models.append(model_full)
        
        oof_xgb += seed_oof / len(XGB_SEEDS)
        test_xgb += seed_test / len(XGB_SEEDS)
        print(f"  Seed {seed}: CV = {rmse(y, seed_oof):.2f}")
    
    xgb_cv = rmse(y, oof_xgb)
    print(f"XGBoost Final CV: {xgb_cv:.2f}")
    save_checkpoint('xgb_final_models', xgb_models)
    
    return (oof_lgb, oof_cat, oof_xgb, 
            test_lgb, test_cat, test_xgb,
            lgb_models, cat_models, xgb_models)


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
    
    return result.x / result.x.sum()


def main():
    print("\n" + "=" * 70)
    print("EXTENSIVE HYPERPARAMETER TUNING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Target: Sub-29,000 LB | 36,000 Ensemble CV")
    print("=" * 70)
    
    # Load data
    X_train, y_train, X_test, test_df, preprocessor = load_data()
    
    # ==================== TUNING PHASE ====================
    
    # Tune LightGBM
    lgb_params, lgb_cv = tune_lightgbm(X_train, y_train, N_TRIALS)
    
    # Tune CatBoost
    cat_params, cat_cv = tune_catboost(X_train, y_train, N_TRIALS)
    
    # Tune XGBoost
    xgb_params, xgb_cv = tune_xgboost(X_train, y_train, N_TRIALS)
    
    # Save all tuned params
    save_checkpoint('all_tuned_params', {
        'lgb': lgb_params,
        'cat': cat_params,
        'xgb': xgb_params,
        'lgb_cv': lgb_cv,
        'cat_cv': cat_cv,
        'xgb_cv': xgb_cv
    })
    
    print("\n" + "=" * 70)
    print("TUNING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"LightGBM: {lgb_cv:.2f} (was 37,880)")
    print(f"CatBoost: {cat_cv:.2f} (was 37,621)")
    print(f"XGBoost:  {xgb_cv:.2f} (was 37,492)")
    
    # ==================== FINAL TRAINING ====================
    
    results = train_final_models(
        X_train, y_train, X_test,
        lgb_params, cat_params, xgb_params
    )
    
    oof_lgb, oof_cat, oof_xgb, test_lgb, test_cat, test_xgb, lgb_models, cat_models, xgb_models = results
    
    # Find optimal weights
    weights = find_optimal_weights(y_train, oof_lgb, oof_cat, oof_xgb)
    print(f"\nOptimal weights: LGB={weights[0]:.3f}, CAT={weights[1]:.3f}, XGB={weights[2]:.3f}")
    
    # Ensemble predictions
    oof_ensemble = weights[0] * oof_lgb + weights[1] * oof_cat + weights[2] * oof_xgb
    test_ensemble = weights[0] * test_lgb + weights[1] * test_cat + weights[2] * test_xgb
    
    ensemble_cv = rmse(y_train, oof_ensemble)
    
    # ==================== RESULTS ====================
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"LightGBM CV:  {rmse(y_train, oof_lgb):.2f} (was 37,880)")
    print(f"CatBoost CV:  {rmse(y_train, oof_cat):.2f} (was 37,621)")
    print(f"XGBoost CV:   {rmse(y_train, oof_xgb):.2f} (was 37,492)")
    print(f"Ensemble CV:  {ensemble_cv:.2f} (was 37,170 | target: 36,000)")
    print("=" * 70)
    
    # Save submission
    submission = pd.DataFrame({
        'id': test_df['_id'],
        'price': test_ensemble
    })
    submission.to_csv('outputs/predictions/submission_extensively_tuned.csv', index=False)
    print(f"\nSubmission saved: outputs/predictions/submission_extensively_tuned.csv")
    
    # Save all OOF predictions
    save_checkpoint('oof_predictions_tuned', {
        'lgb': oof_lgb,
        'cat': oof_cat,
        'xgb': oof_xgb,
        'y_true': y_train,
        'weights': weights,
        'ensemble_cv': ensemble_cv
    })
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Final comparison
    improvement = 37170 - ensemble_cv
    print(f"\n{'='*70}")
    print(f"IMPROVEMENT: {improvement:.0f} RMSE (CV)")
    if ensemble_cv < 36000:
        print("✓ TARGET ACHIEVED: Ensemble CV < 36,000!")
    else:
        print(f"Gap to target: {ensemble_cv - 36000:.0f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
