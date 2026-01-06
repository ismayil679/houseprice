"""
Model Training Pipeline for House Price Prediction
Uses XGBoost and CatBoost with RMSE validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from catboost import CatBoostRegressor
import joblib
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import load_data, preprocess_data


def calculate_metrics(y_true, y_pred, set_name=""):
    """Calculate and print regression metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    if set_name:
        print(f"\n{set_name} Metrics:")
        print(f"  RMSE: {rmse:,.2f} AZN")
        print(f"  MAE:  {mae:,.2f} AZN")
        print(f"  R²:   {r2:.4f}")
    
    return metrics


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    print("Parameters:", json.dumps(params, indent=2))
    
    # Train with early stopping
    params['early_stopping_rounds'] = 50
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score:.4f}")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    train_metrics = calculate_metrics(y_train, y_train_pred, "Training")
    val_metrics = calculate_metrics(y_val, y_val_pred, "Validation")
    
    return model, train_metrics, val_metrics


def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost model"""
    print("\n" + "="*60)
    print("TRAINING CATBOOST")
    print("="*60)
    
    # CatBoost parameters
    params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 3,
        'subsample': 0.8,
        'colsample_bylevel': 0.8,
        'min_data_in_leaf': 20,
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 50,
        'eval_metric': 'RMSE'
    }
    
    print("Parameters:", json.dumps(params, indent=2))
    
    # Train model
    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=100
    )
    
    print(f"\nBest iteration: {model.get_best_iteration()}")
    print(f"Best score: {model.get_best_score()['validation']['RMSE']:.4f}")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    train_metrics = calculate_metrics(y_train, y_train_pred, "Training")
    val_metrics = calculate_metrics(y_val, y_val_pred, "Validation")
    
    return model, train_metrics, val_metrics


def main():
    """Main training pipeline"""
    print("="*60)
    print("HOUSE PRICE PREDICTION - TRAINING PIPELINE")
    print("="*60)
    
    # Load and preprocess data
    train_df, test_df = load_data()
    X_full, y_full, X_test, train_ids, test_ids, preprocessor = preprocess_data(
        train_df, test_df
    )
    
    # Create train/validation split
    print("\n" + "="*60)
    print("CREATING TRAIN/VALIDATION SPLIT")
    print("="*60)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Validation set: {X_val.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    print(f"\nTarget statistics:")
    print(f"  Mean: {y_train.mean():,.2f} AZN")
    print(f"  Std: {y_train.std():,.2f} AZN")
    print(f"  Min: {y_train.min():,.2f} AZN")
    print(f"  Max: {y_train.max():,.2f} AZN")
    
    # Train models
    results = {}
    
    # XGBoost
    xgb_model, xgb_train_metrics, xgb_val_metrics = train_xgboost(
        X_train, y_train, X_val, y_val
    )
    results['xgboost'] = {
        'train': xgb_train_metrics,
        'val': xgb_val_metrics
    }
    
    # CatBoost
    cb_model, cb_train_metrics, cb_val_metrics = train_catboost(
        X_train, y_train, X_val, y_val
    )
    results['catboost'] = {
        'train': cb_train_metrics,
        'val': cb_val_metrics
    }
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON (VALIDATION SET)")
    print("="*60)
    
    comparison = pd.DataFrame({
        'Model': ['XGBoost', 'CatBoost'],
        'RMSE': [
            results['xgboost']['val']['rmse'],
            results['catboost']['val']['rmse']
        ],
        'MAE': [
            results['xgboost']['val']['mae'],
            results['catboost']['val']['mae']
        ],
        'R²': [
            results['xgboost']['val']['r2'],
            results['catboost']['val']['r2']
        ]
    })
    
    print(comparison.to_string(index=False))
    
    # Select best model based on RMSE
    best_model_name = 'xgboost' if results['xgboost']['val']['rmse'] < results['catboost']['val']['rmse'] else 'catboost'
    best_model = xgb_model if best_model_name == 'xgboost' else cb_model
    best_rmse = results[best_model_name]['val']['rmse']
    
    print(f"\n🏆 Best Model: {best_model_name.upper()}")
    print(f"   Validation RMSE: {best_rmse:,.2f} AZN")
    
    # Save models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    os.makedirs('models', exist_ok=True)
    
    # Save XGBoost
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    print("✓ XGBoost saved to models/xgboost_model.pkl")
    
    # Save CatBoost
    cb_model.save_model('models/catboost_model.cbm')
    print("✓ CatBoost saved to models/catboost_model.cbm")
    
    # Save best model reference
    best_model_info = {
        'best_model': best_model_name,
        'validation_rmse': best_rmse,
        'results': results
    }
    
    with open('models/best_model_info.json', 'w') as f:
        json.dump(best_model_info, f, indent=2)
    print("✓ Best model info saved to models/best_model_info.json")
    
    # Feature importance
    print("\n" + "="*60)
    print(f"FEATURE IMPORTANCE ({best_model_name.upper()})")
    print("="*60)
    
    if best_model_name == 'xgboost':
        importance = xgb_model.feature_importances_
    else:
        importance = cb_model.feature_importances_
    
    feature_names = [
        'area', 'room_count', 'floor', 'total_floors',
        'has_deed', 'has_mortgage', 'is_owner', 'is_agent',
        'lat', 'lon', 'dist_from_center', 'city_baki',
        'title_length', 'desc_length', 'is_new_building'
    ]
    
    feat_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(feat_importance.to_string(index=False))
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review validation metrics above")
    print("  2. Run evaluate.py to generate predictions on test set")
    print("  3. Submit predictions file to competition")
    
    return results, best_model_name


if __name__ == "__main__":
    results, best_model = main()
