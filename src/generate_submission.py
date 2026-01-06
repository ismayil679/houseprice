"""Quick script to generate submission from trained models"""
import pandas as pd
import numpy as np
import joblib
import sys
import os

sys.path.append(os.path.dirname(__file__))
from preprocessing import HousePricePreprocessor

print("Loading test data...")
test = pd.read_csv('data/binaaz_test.csv')

print("Loading models and preprocessor...")
models = joblib.load('models/fine_tuned_ensemble.pkl')
weights = joblib.load('models/fine_tuned_weights.pkl')
preprocessor = joblib.load('models/fine_tuned_preprocessor.pkl')

print("Preprocessing...")
X_test = preprocessor.extract_features(test)
X_test = preprocessor.transform(X_test)

print("Generating predictions...")
test_preds = []
for name, model in models:
    pred = np.expm1(model.predict(X_test))
    test_preds.append(pred)

test_preds_array = np.array(test_preds).T

# Simple average
simple_test = test_preds_array.mean(axis=1)

# Weighted
weighted_test = (test_preds_array * weights).sum(axis=1)

print(f"\nPrediction stats:")
print(f"  Simple mean:   {simple_test.mean():,.0f}")
print(f"  Weighted mean: {weighted_test.mean():,.0f}")

# Save submissions
submission = pd.DataFrame({
    'id': test['_id'],
    'price': weighted_test
})

submission.to_csv('outputs/predictions/submission_latest.csv', index=False)
print(f"\n✅ Submission saved: submission_latest.csv")
print(f"Expected leaderboard RMSE: ~31,690 (CV 37,283 × 0.85)")
