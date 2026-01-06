"""
Model Evaluation and Test Set Prediction
Generates final predictions for competition submission
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
import joblib
import json
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import load_data, preprocess_data


def load_best_model():
    """Load the best trained model"""
    print("Loading best model...")
    
    # Load model info
    with open('models/best_model_info.json', 'r') as f:
        model_info = json.load(f)
    
    best_model_name = model_info['best_model']
    val_rmse = model_info['validation_rmse']
    
    print(f"Best model: {best_model_name.upper()}")
    print(f"Validation RMSE: {val_rmse:,.2f} AZN")
    
    # Load the model
    if best_model_name == 'xgboost':
        model = joblib.load('models/xgboost_model.pkl')
    elif best_model_name == 'catboost':
        model = CatBoostRegressor()
        model.load_model('models/catboost_model.cbm')
    else:
        raise ValueError(f"Unknown model: {best_model_name}")
    
    return model, best_model_name, val_rmse


def generate_predictions(model, X_test, test_ids):
    """Generate predictions for test set"""
    print("\nGenerating predictions on test set...")
    
    predictions = model.predict(X_test)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        '_id': test_ids,
        'price': predictions
    })
    
    # Ensure no negative predictions
    submission['price'] = submission['price'].clip(lower=0)
    
    print(f"\nPrediction statistics:")
    print(f"  Count: {len(predictions):,}")
    print(f"  Mean: {predictions.mean():,.2f} AZN")
    print(f"  Std: {predictions.std():,.2f} AZN")
    print(f"  Min: {predictions.min():,.2f} AZN")
    print(f"  Max: {predictions.max():,.2f} AZN")
    
    return submission


def save_predictions(submission, model_name):
    """Save predictions to CSV file"""
    os.makedirs('outputs/predictions', exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'outputs/predictions/submission_{model_name}_{timestamp}.csv'
    
    submission.to_csv(filename, index=False)
    print(f"\n✓ Predictions saved to: {filename}")
    
    # Also save a simple 'latest' version
    latest_file = 'outputs/predictions/submission_latest.csv'
    submission.to_csv(latest_file, index=False)
    print(f"✓ Latest predictions saved to: {latest_file}")
    
    return filename


def main():
    """Main evaluation pipeline"""
    print("="*60)
    print("HOUSE PRICE PREDICTION - EVALUATION")
    print("="*60)
    
    # Load and preprocess data
    train_df, test_df = load_data()
    X_full, y_full, X_test, train_ids, test_ids, preprocessor = preprocess_data(
        train_df, test_df
    )
    
    print(f"\nTest set size: {X_test.shape[0]:,} samples")
    
    # Load best model
    model, model_name, val_rmse = load_best_model()
    
    # Generate predictions
    submission = generate_predictions(model, X_test, test_ids)
    
    # Save predictions
    filename = save_predictions(submission, model_name)
    
    # Summary
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nModel used: {model_name.upper()}")
    print(f"Validation RMSE: {val_rmse:,.2f} AZN")
    print(f"Test predictions: {len(submission):,}")
    print(f"\nSubmission file ready: {filename}")
    print("\n📤 You can now submit this file to the competition!")
    
    return submission


if __name__ == "__main__":
    submission = main()
