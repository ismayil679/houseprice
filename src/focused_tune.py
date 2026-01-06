"""
Focused Hyperparameter Tuning - Based on Working Baseline
==========================================================
Baseline: 35,975 RMSE with these params:
- learning_rate: 0.03
- num_leaves: 197
- max_depth: 10
- min_child_samples: 11
- subsample: 0.89
- colsample_bytree: 0.74
- reg_alpha: 0.81
- reg_lambda: 4.55

Target: ≤ 33,500 RMSE
"""

import sys
sys.path.insert(0, 'src')

import os
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.model_selection import KFold  # Same as train_v3.py
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from preprocessing_v3 import HousePricePreprocessorV3, load_data

# ============================================================
# CONFIGURATION
# ============================================================
CHECKPOINT_FILE = 'models/tuning_checkpoints/focused_tune_checkpoint.json'
N_FOLDS = 5
TARGET_RMSE = 33500
EARLY_STOPPING = 200

os.makedirs('models/tuning_checkpoints', exist_ok=True)

# Baseline params that gave 35,975 RMSE
BASELINE_PARAMS = {
    'learning_rate': 0.03,
    'num_leaves': 197,
    'max_depth': 10,
    'min_child_samples': 11,
    'subsample': 0.89,
    'colsample_bytree': 0.74,
    'reg_alpha': 0.81,
    'reg_lambda': 4.55,
}


def load_and_prepare_data():
    """Load data exactly as train_v3.py does"""
    print("=" * 70)
    print("LOADING DATA (same as train_v3.py)")
    print("=" * 70)
    
    train_df, test_df = load_data()
    
    # Outlier removal - SAME as train_v3.py
    y_full = train_df['price']
    q_low = y_full.quantile(0.01)
    q_high = y_full.quantile(0.99)
    mask = (y_full >= q_low) & (y_full <= q_high)
    train_df = train_df[mask].reset_index(drop=True)
    
    print(f"  Samples: {len(train_df):,}")
    print(f"  Price range: {q_low:,.0f} - {q_high:,.0f} AZN")
    
    # Extract features - SAME as train_v3.py
    preprocessor = HousePricePreprocessorV3()
    X_train = preprocessor.extract_features(train_df, target=train_df['price'], is_train=True)
    
    print(f"  Features: {X_train.shape[1]}")
    
    # Prepare data - SAME as train_v3.py
    preprocessor.fit(X_train)
    X = preprocessor.transform(X_train)
    y = np.log1p(train_df['price'].values)  # Log transform
    
    return X, y, preprocessor.feature_cols


def evaluate_lgb(params, X, y, feature_names):
    """Evaluate LightGBM with KFold - SAME as train_v3.py"""
    
    full_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'n_jobs': -1,
        **params
    }
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)  # SAME as train_v3.py
    
    oof_preds = np.zeros(len(y))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_tr, y_tr, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, y_val, feature_name=feature_names, reference=train_data)
        
        model = lgb.train(
            full_params,
            train_data,
            num_boost_round=10000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)]
        )
        
        oof_preds[val_idx] = model.predict(X_val)
        
        # RMSE in original scale - SAME as train_v3.py
        fold_rmse = np.sqrt(mean_squared_error(
            np.expm1(y_val), 
            np.expm1(oof_preds[val_idx])
        ))
        fold_scores.append(fold_rmse)
    
    overall_rmse = np.sqrt(mean_squared_error(np.expm1(y), np.expm1(oof_preds)))
    
    return overall_rmse, fold_scores


def save_checkpoint(results, best_idx):
    """Save checkpoint"""
    checkpoint = {
        'results': results,
        'best_idx': best_idx,
        'best_rmse': results[best_idx]['rmse'],
        'best_params': results[best_idx]['params'],
        'timestamp': datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def run_tuning(n_trials=50):
    """Run focused tuning around baseline params"""
    
    print("\n" + "=" * 70)
    print("FOCUSED HYPERPARAMETER TUNING")
    print(f"Baseline RMSE: 35,975 | Target: ≤ {TARGET_RMSE:,}")
    print("=" * 70)
    
    # Load data
    X, y, feature_names = load_and_prepare_data()
    
    # First verify baseline
    print("\n" + "-" * 70)
    print("VERIFYING BASELINE...")
    print("-" * 70)
    
    baseline_rmse, baseline_folds = evaluate_lgb(BASELINE_PARAMS, X, y, feature_names)
    print(f"  Baseline RMSE: {baseline_rmse:,.0f}")
    print(f"  Folds: {[f'{s:,.0f}' for s in baseline_folds]}")
    
    if baseline_rmse > 37000:
        print("\n  WARNING: Baseline worse than expected! Check preprocessing.")
        return None
    
    # Store results
    results = [{
        'trial': 0,
        'name': 'baseline',
        'rmse': baseline_rmse,
        'fold_scores': baseline_folds,
        'params': BASELINE_PARAMS.copy()
    }]
    best_idx = 0
    
    save_checkpoint(results, best_idx)
    
    # Tuning phase
    print("\n" + "-" * 70)
    print(f"TUNING ({n_trials} trials)")
    print("-" * 70)
    
    start_time = time.time()
    
    def create_trial_params(trial):
        """Generate params around baseline with focused search"""
        return {
            # Learning rate: try slightly lower for better generalization
            'learning_rate': trial.suggest_float('lr', 0.015, 0.05),
            
            # Tree structure: explore around baseline
            'num_leaves': trial.suggest_int('num_leaves', 127, 255),
            'max_depth': trial.suggest_int('max_depth', 8, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            
            # Regularization: baseline has good values, explore nearby
            'reg_alpha': trial.suggest_float('reg_alpha', 0.3, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 2.0, 8.0),
            
            # Sampling: stay close to baseline
            'subsample': trial.suggest_float('subsample', 0.8, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.85),
            
            # Additional params
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.1),
        }
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )
    
    for trial_num in range(1, n_trials + 1):
        trial = study.ask()
        params = create_trial_params(trial)
        
        trial_start = time.time()
        
        try:
            rmse, fold_scores = evaluate_lgb(params, X, y, feature_names)
            study.tell(trial, rmse)
            
            # Progress
            elapsed = time.time() - start_time
            eta = (elapsed / trial_num) * (n_trials - trial_num)
            eta_str = str(timedelta(seconds=int(eta)))
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            
            # Status
            is_best = rmse < results[best_idx]['rmse']
            status = "★ NEW BEST!" if is_best else ""
            
            print(f"  Trial {trial_num:2d}/{n_trials} | RMSE: {rmse:,.0f} | "
                  f"Folds: [{', '.join([f'{s:,.0f}' for s in fold_scores])}] | "
                  f"Time: {elapsed_str} | ETA: {eta_str} {status}")
            
            # Store result
            results.append({
                'trial': trial_num,
                'rmse': rmse,
                'fold_scores': fold_scores,
                'params': params
            })
            
            if is_best:
                best_idx = len(results) - 1
            
            # Save checkpoint
            save_checkpoint(results, best_idx)
            
        except Exception as e:
            print(f"  Trial {trial_num:2d}/{n_trials} | ERROR: {str(e)[:50]}")
            study.tell(trial, float('inf'))
    
    # Final summary
    print("\n" + "=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)
    
    best = results[best_idx]
    print(f"\n  Best RMSE: {best['rmse']:,.0f}")
    print(f"  Folds: {[f'{s:,.0f}' for s in best['fold_scores']]}")
    print(f"  Improvement: {baseline_rmse - best['rmse']:+,.0f} from baseline")
    print(f"  Gap to target: {best['rmse'] - TARGET_RMSE:+,.0f}")
    
    print("\n  Best Parameters:")
    for k, v in best['params'].items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    
    # Save final
    with open('models/tuning_checkpoints/best_focused_params.json', 'w') as f:
        json.dump(best, f, indent=2)
    
    print("\n  Saved to models/tuning_checkpoints/best_focused_params.json")
    
    return best


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=50)
    args = parser.parse_args()
    
    result = run_tuning(n_trials=args.n_trials)
    
    if result:
        if result['rmse'] <= TARGET_RMSE:
            print(f"\n🎉 TARGET ACHIEVED! {result['rmse']:,.0f} ≤ {TARGET_RMSE:,}")
        else:
            print(f"\n⚠️ Need more tuning. Best: {result['rmse']:,.0f}, Target: {TARGET_RMSE:,}")
