"""
FAST CatBoost & LightGBM Tuning (Smart Approach)
- 1,500 iterations for exploration (fast)
- 50 trials each (more exploration)
- Final training with 10,000 iterations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import lightgbm as lgb
import optuna
import joblib
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(__file__))
from preprocessing import HousePricePreprocessor

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data():
    """Load and preprocess data"""
    train = pd.read_csv('data/binaaz_train.csv')
    test = pd.read_csv('data/binaaz_test.csv')
    
    preprocessor = HousePricePreprocessor()
    
    X_train_df = preprocessor.extract_features(train)
    y_full = train['price']
    
    # Remove outliers
    q_low = y_full.quantile(0.01)
    q_high = y_full.quantile(0.99)
    mask = (y_full >= q_low) & (y_full <= q_high)
    
    X_train_df = X_train_df[mask].reset_index(drop=True)
    y_train = y_full[mask].reset_index(drop=True).values
    
    X_train = preprocessor.fit_transform(X_train_df)
    
    X_test_df = preprocessor.extract_features(test)
    X_test = preprocessor.transform(X_test_df)
    
    return X_train, y_train, X_test, test, preprocessor


def cv_score_catboost(params, X, y):
    """3-fold CV for speed during tuning"""
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(
            X_tr, np.log1p(y_tr),
            eval_set=(X_val, np.log1p(y_val)),
            early_stopping_rounds=50,
            verbose=0
        )
        
        pred = np.expm1(model.predict(X_val))
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        scores.append(rmse)
    
    return np.mean(scores)


def cv_score_lightgbm(params, X, y):
    """5-fold CV for stability (LightGBM needs more stable CV)"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, np.log1p(y_tr),
            eval_set=[(X_val, np.log1p(y_val))],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        
        pred = np.expm1(model.predict(X_val))
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        scores.append(rmse)
    
    return np.mean(scores)


def catboost_objective(trial, X, y):
    """Optuna objective for CatBoost - FAST with 1500 iterations"""
    params = {
        'iterations': 1500,  # Fast exploration
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'depth': trial.suggest_int('depth', 4, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'random_seed': 42,
        'verbose': 0
    }
    
    if params['grow_policy'] == 'Lossguide':
        params['max_leaves'] = trial.suggest_int('max_leaves', 31, 512)
    
    return cv_score_catboost(params, X, y)


def lightgbm_objective(trial, X, y):
    """Optuna objective for LightGBM - Ranges centered around known good params
    
    Known good params (38,500 CV):
    - learning_rate: 0.026
    - num_leaves: 198
    - max_depth: 10
    - subsample: 0.90
    - reg_alpha: 1.3
    - reg_lambda: 3.99
    """
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 2000,  # More iterations for better convergence
        # Centered around 0.026 (good value)
        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.04),
        # Centered around 198 (good value)
        'num_leaves': trial.suggest_int('num_leaves', 150, 250),
        # Centered around 10 (good value)
        'max_depth': trial.suggest_int('max_depth', 8, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        # Centered around 0.90 (good value) - HIGH subsample is key!
        'subsample': trial.suggest_float('subsample', 0.80, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        # Centered around 1.3 (good value) - LOW reg_alpha is key!
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 2.5),
        # Centered around 3.99 (good value)
        'reg_lambda': trial.suggest_float('reg_lambda', 2.0, 6.0),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3),
        'random_state': 42,
        'force_col_wise': True
    }
    
    return cv_score_lightgbm(params, X, y)


def progress_bar(current, total, best_score, elapsed_min):
    """Display progress bar"""
    pct = current / total * 100
    bar_len = 40
    filled = int(bar_len * current / total)
    bar = '█' * filled + '░' * (bar_len - filled)
    
    eta_min = (elapsed_min / current * (total - current)) if current > 0 else 0
    
    print(f'\r  [{bar}] {current}/{total} ({pct:.0f}%) | Best: {best_score:,.0f} | ETA: {eta_min:.0f}min', end='', flush=True)


def main():
    start_time = datetime.now()
    
    print("="*70)
    print("⚡ FAST TUNING (Smart Approach)")
    print("   1,500 iterations for tuning → 10,000 for final")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Load data
    print("\n📊 Loading data...")
    X_train, y_train, X_test, test_df, preprocessor = load_data()
    print(f"   Training samples: {len(X_train):,}")
    
    # =========================================================================
    # PHASE 1: CatBoost Fast Tuning (50 trials)
    # =========================================================================
    print("\n" + "="*70)
    print("🟠 PHASE 1: CatBoost Fast Tuning (50 trials, ~1 min each)")
    print("="*70)
    
    cat_study = optuna.create_study(direction='minimize', study_name='catboost_fast')
    cat_best = float('inf')
    cat_start = datetime.now()
    
    n_trials_cat = 50
    
    for i in range(n_trials_cat):
        cat_study.optimize(
            lambda trial: catboost_objective(trial, X_train, y_train),
            n_trials=1,
            show_progress_bar=False
        )
        
        if cat_study.best_value < cat_best:
            cat_best = cat_study.best_value
            joblib.dump(cat_study, 'models/study_cat_fast.pkl')
        
        elapsed = (datetime.now() - cat_start).total_seconds() / 60
        progress_bar(i + 1, n_trials_cat, cat_best, elapsed)
    
    print(f"\n\n   ✅ CatBoost Best CV: {cat_best:,.2f}")
    joblib.dump(cat_study, 'models/study_cat_fast.pkl')
    
    # =========================================================================
    # PHASE 2: LightGBM Fast Tuning (50 trials)
    # =========================================================================
    print("\n" + "="*70)
    print("🟢 PHASE 2: LightGBM Fast Tuning (50 trials, ~30 sec each)")
    print("="*70)
    
    lgb_study = optuna.create_study(direction='minimize', study_name='lightgbm_fast')
    lgb_best = float('inf')
    lgb_start = datetime.now()
    
    n_trials_lgb = 50
    
    for i in range(n_trials_lgb):
        lgb_study.optimize(
            lambda trial: lightgbm_objective(trial, X_train, y_train),
            n_trials=1,
            show_progress_bar=False
        )
        
        if lgb_study.best_value < lgb_best:
            lgb_best = lgb_study.best_value
            joblib.dump(lgb_study, 'models/study_lgb_fast.pkl')
        
        elapsed = (datetime.now() - lgb_start).total_seconds() / 60
        progress_bar(i + 1, n_trials_lgb, lgb_best, elapsed)
    
    print(f"\n\n   ✅ LightGBM Best CV: {lgb_best:,.2f}")
    joblib.dump(lgb_study, 'models/study_lgb_fast.pkl')
    
    # =========================================================================
    # PHASE 3: Final Training with Best Params (10,000 iterations)
    # =========================================================================
    print("\n" + "="*70)
    print("🏆 PHASE 3: Final Training with 10,000 iterations")
    print("="*70)
    
    # CatBoost final
    print("\n   Training CatBoost (10,000 iter, 5-fold CV)...")
    cat_best_params = cat_study.best_params.copy()
    cat_best_params['iterations'] = 10000
    cat_best_params['random_seed'] = 42
    cat_best_params['verbose'] = 0
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cat_oof = np.zeros(len(X_train))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        model = CatBoostRegressor(**cat_best_params)
        model.fit(
            X_train[train_idx], np.log1p(y_train[train_idx]),
            eval_set=(X_train[val_idx], np.log1p(y_train[val_idx])),
            early_stopping_rounds=100,
            verbose=0
        )
        cat_oof[val_idx] = np.expm1(model.predict(X_train[val_idx]))
        print(f"      Fold {fold+1}/5 complete")
    
    cat_final_cv = np.sqrt(mean_squared_error(y_train, cat_oof))
    print(f"   ✅ CatBoost Final CV: {cat_final_cv:,.2f}")
    
    # LightGBM final
    print("\n   Training LightGBM (10,000 iter, 5-fold CV)...")
    lgb_best_params = lgb_study.best_params.copy()
    lgb_best_params['n_estimators'] = 10000
    lgb_best_params['random_state'] = 42
    lgb_best_params['verbosity'] = -1
    lgb_best_params['force_col_wise'] = True
    
    lgb_oof = np.zeros(len(X_train))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        model = lgb.LGBMRegressor(**lgb_best_params)
        model.fit(
            X_train[train_idx], np.log1p(y_train[train_idx]),
            eval_set=[(X_train[val_idx], np.log1p(y_train[val_idx]))],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
        )
        lgb_oof[val_idx] = np.expm1(model.predict(X_train[val_idx]))
        print(f"      Fold {fold+1}/5 complete")
    
    lgb_final_cv = np.sqrt(mean_squared_error(y_train, lgb_oof))
    print(f"   ✅ LightGBM Final CV: {lgb_final_cv:,.2f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    print(f"\n   🟠 CatBoost:  {cat_final_cv:,.2f} (was ~38,000)")
    print(f"   🟢 LightGBM:  {lgb_final_cv:,.2f} (was ~38,500)")
    print(f"   🔵 XGBoost:   38,094 (from previous)")
    print(f"\n   Total time: {total_duration:.1f} minutes")
    print(f"\n   Saved: models/study_cat_fast.pkl")
    print(f"          models/study_lgb_fast.pkl")
    print("="*70)
    
    # Save best params for later use
    joblib.dump(cat_best_params, 'models/cat_best_params.pkl')
    joblib.dump(lgb_best_params, 'models/lgb_best_params.pkl')
    
    print("\n🎯 Next: Run combined ensemble with these new params!")


if __name__ == '__main__':
    main()
