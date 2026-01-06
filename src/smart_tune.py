"""
Smart Hyperparameter Tuning with Optuna
========================================
Features:
- Stratified K-Fold (binned prices)
- Progress bar with ETA
- Checkpointing after each trial
- Error handling and resume capability
- Target: CV RMSE ≤ 33,500

Usage:
    python src/smart_tune.py [--resume] [--n_trials 100]
"""

import sys
sys.path.insert(0, 'src')

import os
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from preprocessing_v3 import HousePricePreprocessorV3, load_data

# ============================================================
# CONFIGURATION
# ============================================================
CHECKPOINT_FILE = 'models/tuning_checkpoints/smart_tune_checkpoint.json'
STUDY_FILE = 'models/tuning_checkpoints/smart_tune_study.db'
N_FOLDS = 5
TARGET_RMSE = 33500
EARLY_STOPPING = 200

os.makedirs('models/tuning_checkpoints', exist_ok=True)


def load_and_prepare_data():
    """Load data with outlier removal and stratification bins"""
    print("=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)
    
    train_df, test_df = load_data()
    
    # Outlier removal (1st and 99th percentile)
    y_full = train_df['price']
    q_low = y_full.quantile(0.01)
    q_high = y_full.quantile(0.99)
    mask = (y_full >= q_low) & (y_full <= q_high)
    train_df = train_df[mask].reset_index(drop=True)
    
    print(f"  Samples after outlier removal: {len(train_df):,}")
    print(f"  Price range: {q_low:,.0f} - {q_high:,.0f} AZN")
    
    # Extract features
    preprocessor = HousePricePreprocessorV3()
    X_train = preprocessor.extract_features(train_df, target=train_df['price'], is_train=True)
    
    print(f"  Features: {X_train.shape[1]}")
    
    # Prepare data
    preprocessor.fit(X_train)
    X = preprocessor.transform(X_train)
    y = np.log1p(train_df['price'].values)
    
    # Create stratification bins (10 quantile bins for price)
    y_binned = pd.qcut(train_df['price'], q=10, labels=False, duplicates='drop')
    
    print(f"  Stratification bins: {y_binned.nunique()}")
    
    return X, y, y_binned.values, preprocessor.feature_cols, preprocessor


def progress_bar(current, total, start_time, prefix='', width=40):
    """Display progress bar with ETA"""
    filled = int(width * current / total)
    bar = '█' * filled + '░' * (width - filled)
    percent = 100 * current / total
    
    elapsed = time.time() - start_time
    if current > 0:
        eta_seconds = (elapsed / current) * (total - current)
        eta = str(timedelta(seconds=int(eta_seconds)))
    else:
        eta = "calculating..."
    
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    
    print(f"\r{prefix} |{bar}| {percent:5.1f}% [{current}/{total}] Elapsed: {elapsed_str} ETA: {eta}    ", end='', flush=True)


def evaluate_params(params, X, y, y_binned, feature_names):
    """Evaluate parameters with stratified CV"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(y))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_tr, y_tr, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, y_val, feature_name=feature_names, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=10000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)]
        )
        
        oof_preds[val_idx] = model.predict(X_val)
        
        # Calculate fold RMSE in original scale
        fold_rmse = np.sqrt(mean_squared_error(
            np.expm1(y_val), 
            np.expm1(oof_preds[val_idx])
        ))
        fold_scores.append(fold_rmse)
    
    # Overall RMSE
    overall_rmse = np.sqrt(mean_squared_error(np.expm1(y), np.expm1(oof_preds)))
    
    return overall_rmse, fold_scores


def save_checkpoint(trial_num, best_rmse, best_params, all_results):
    """Save checkpoint to disk"""
    checkpoint = {
        'trial_num': trial_num,
        'best_rmse': best_rmse,
        'best_params': best_params,
        'all_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint():
    """Load checkpoint if exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None


def create_objective(X, y, y_binned, feature_names, all_results, start_time, n_trials):
    """Create Optuna objective function"""
    
    def objective(trial):
        trial_start = time.time()
        trial_num = trial.number + 1
        
        # Smart parameter search space
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'n_jobs': -1,
            
            # Learning rate - try lower values for better generalization
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            
            # Tree structure
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            
            # Regularization
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            
            # Sampling
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            
            # Feature fraction
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        }
        
        try:
            rmse, fold_scores = evaluate_params(params, X, y, y_binned, feature_names)
            
            # Store result
            result = {
                'trial': trial_num,
                'rmse': rmse,
                'fold_scores': fold_scores,
                'params': {k: v for k, v in params.items() if k not in ['objective', 'metric', 'boosting_type', 'verbose', 'n_jobs']},
                'time': time.time() - trial_start
            }
            all_results.append(result)
            
            # Update progress
            progress_bar(trial_num, n_trials, start_time, prefix='Tuning')
            
            # Print result on new line
            fold_str = ', '.join([f'{s:,.0f}' for s in fold_scores])
            status = "✓ NEW BEST!" if rmse < min([r['rmse'] for r in all_results[:-1]], default=float('inf')) else ""
            print(f"\n  Trial {trial_num:3d}: RMSE = {rmse:,.0f} | Folds: [{fold_str}] | Std: {np.std(fold_scores):,.0f} {status}")
            
            # Save checkpoint after each trial
            best_result = min(all_results, key=lambda x: x['rmse'])
            save_checkpoint(trial_num, best_result['rmse'], best_result['params'], all_results)
            
            return rmse
            
        except Exception as e:
            print(f"\n  Trial {trial_num}: ERROR - {str(e)[:50]}")
            return float('inf')
    
    return objective


def run_tuning(n_trials=100, resume=False):
    """Run the hyperparameter tuning"""
    print("\n" + "=" * 70)
    print("SMART HYPERPARAMETER TUNING")
    print(f"Target: CV RMSE ≤ {TARGET_RMSE:,}")
    print("=" * 70)
    
    # Load data
    X, y, y_binned, feature_names, preprocessor = load_and_prepare_data()
    
    # Check for existing checkpoint
    all_results = []
    start_trial = 0
    
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            all_results = checkpoint['all_results']
            start_trial = checkpoint['trial_num']
            print(f"\n  Resuming from trial {start_trial + 1}")
            print(f"  Best RMSE so far: {checkpoint['best_rmse']:,.0f}")
    
    # Calculate remaining trials
    remaining_trials = n_trials - start_trial
    if remaining_trials <= 0:
        print(f"\n  Already completed {n_trials} trials!")
        best_result = min(all_results, key=lambda x: x['rmse'])
        print(f"\n  Best RMSE: {best_result['rmse']:,.0f}")
        return best_result
    
    print(f"\n  Running {remaining_trials} trials...")
    print("-" * 70)
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        study_name='lgb_tuning',
        storage=f'sqlite:///{STUDY_FILE}',
        load_if_exists=resume
    )
    
    # Create objective
    start_time = time.time()
    objective = create_objective(X, y, y_binned, feature_names, all_results, start_time, n_trials)
    
    # Run optimization
    try:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            show_progress_bar=False,
            catch=(Exception,)
        )
    except KeyboardInterrupt:
        print("\n\n  Interrupted! Saving checkpoint...")
        if all_results:
            best_result = min(all_results, key=lambda x: x['rmse'])
            save_checkpoint(len(all_results), best_result['rmse'], best_result['params'], all_results)
            print(f"  Checkpoint saved at trial {len(all_results)}")
    
    # Final results
    print("\n" + "=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)
    
    if all_results:
        best_result = min(all_results, key=lambda x: x['rmse'])
        
        print(f"\n  Best RMSE: {best_result['rmse']:,.0f}")
        print(f"  Fold scores: {[f'{s:,.0f}' for s in best_result['fold_scores']]}")
        print(f"  Target: {TARGET_RMSE:,} | Gap: {best_result['rmse'] - TARGET_RMSE:+,.0f}")
        
        print("\n  Best Parameters:")
        for k, v in best_result['params'].items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
        
        # Save final best params
        with open('models/tuning_checkpoints/best_lgb_params.json', 'w') as f:
            json.dump({
                'rmse': best_result['rmse'],
                'params': best_result['params'],
                'fold_scores': best_result['fold_scores']
            }, f, indent=2)
        
        print("\n  Best params saved to models/tuning_checkpoints/best_lgb_params.json")
        
        return best_result
    
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials')
    args = parser.parse_args()
    
    result = run_tuning(n_trials=args.n_trials, resume=args.resume)
    
    if result and result['rmse'] <= TARGET_RMSE:
        print(f"\n🎉 TARGET ACHIEVED! RMSE {result['rmse']:,.0f} ≤ {TARGET_RMSE:,}")
    elif result:
        print(f"\n⚠️  Target not reached. Best: {result['rmse']:,.0f}, Need: {TARGET_RMSE:,}")
