"""
Data Preprocessing Module for House Price Prediction
Handles feature extraction, cleaning, and transformation
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
import joblib
import os


class HousePricePreprocessor:
    """Preprocessor for housing price data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_medians = {}
        self.feature_cols = [
            'area', 'room_count', 'floor', 'total_floors',
            'has_deed', 'has_mortgage', 'is_owner', 'is_agent',
            'lat', 'lon', 'dist_from_center', 'city_baki',
            'title_length', 'desc_length', 'is_new_building'
        ]
    
    @staticmethod
    def extract_area(area_str):
        """Extract numeric area from string like '115 m²'"""
        if pd.isna(area_str):
            return np.nan
        try:
            match = re.search(r'(\d+\.?\d*)', str(area_str))
            if match:
                return float(match.group(1))
        except:
            return np.nan
        return np.nan
    
    @staticmethod
    def extract_floor_info(floor_str):
        """Extract floor and total floors from string like '5 / 9'"""
        if pd.isna(floor_str):
            return np.nan, np.nan
        try:
            parts = str(floor_str).split('/')
            if len(parts) == 2:
                floor = float(parts[0].strip())
                total_floors = float(parts[1].strip())
                return floor, total_floors
            elif len(parts) == 1:
                return float(parts[0].strip()), np.nan
        except:
            return np.nan, np.nan
        return np.nan, np.nan
    
    def extract_features(self, df):
        """Extract all features from raw dataframe"""
        df = df.copy()
        
        # Numeric features
        df['area'] = df['Sahə'].apply(self.extract_area)
        df[['floor', 'total_floors']] = df['Mərtəbə'].apply(
            lambda x: pd.Series(self.extract_floor_info(x))
        )
        df['room_count'] = df['Otaq sayı']
        
        # Binary features
        df['has_deed'] = (df['Kupça'] == 'var').astype(int)
        df['has_mortgage'] = (df['İpoteka'] == 'var').astype(int)
        df['is_owner'] = (df['poster_type'] == 'mülkiyyətçi').astype(int)
        df['is_agent'] = (df['poster_type'] == 'vasitəçi (agent)').astype(int)
        
        # Location features
        df['lat'] = df['latitude']
        df['lon'] = df['longitude']
        
        # Distance from Baku city center (40.4093, 49.8671)
        df['dist_from_center'] = np.sqrt(
            (df['lat'] - 40.4093)**2 + (df['lon'] - 49.8671)**2
        )
        
        # City encoding
        df['city_baki'] = (df['seher'] == 'baki').astype(int)
        
        # Text features
        df['title_length'] = df['title'].str.len()
        df['desc_length'] = df['description'].fillna('').str.len()
        df['is_new_building'] = df['title'].str.contains(
            'yeni tikili', case=False, na=False
        ).astype(int)
        
        return df[self.feature_cols]
    
    def fit(self, X, y=None):
        """Fit the preprocessor (calculate medians and fit scaler)"""
        # Calculate medians for missing value imputation
        for col in self.feature_cols:
            self.feature_medians[col] = X[col].median()
        
        # Fill missing values
        X_filled = X.copy()
        for col in self.feature_cols:
            X_filled[col].fillna(self.feature_medians[col], inplace=True)
        
        # Fit scaler
        self.scaler.fit(X_filled)
        
        return self
    
    def transform(self, X):
        """Transform features (impute and scale)"""
        X_transformed = X.copy()
        
        # Fill missing values using stored medians
        for col in self.feature_cols:
            median_val = self.feature_medians.get(col, X_transformed[col].median())
            X_transformed[col].fillna(median_val, inplace=True)
        
        # Scale features
        X_scaled = self.scaler.transform(X_transformed)
        
        return X_scaled
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)
    
    def save(self, filepath):
        """Save preprocessor to disk"""
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load preprocessor from disk"""
        return joblib.load(filepath)


def load_data(train_path='data/binaaz_train.csv', test_path='data/binaaz_test.csv'):
    """Load training and test data"""
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df


def preprocess_data(train_df, test_df):
    """
    Preprocess training and test data
    Returns: X_train, y_train, X_test, train_ids, test_ids, preprocessor
    """
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = HousePricePreprocessor()
    
    # Extract features
    print("Extracting features from training data...")
    X_train_raw = preprocessor.extract_features(train_df)
    y_train = train_df['price'].values
    train_ids = train_df['_id'].values
    
    print("Extracting features from test data...")
    X_test_raw = preprocessor.extract_features(test_df)
    test_ids = test_df['_id'].values
    
    print(f"\nExtracted features shape:")
    print(f"  Training: {X_train_raw.shape}")
    print(f"  Test: {X_test_raw.shape}")
    
    # Fit and transform
    print("\nFitting preprocessor on training data...")
    X_train = preprocessor.fit_transform(X_train_raw)
    
    print("Transforming test data...")
    X_test = preprocessor.transform(X_test_raw)
    
    print("\n✅ Preprocessing complete!")
    
    return X_train, y_train, X_test, train_ids, test_ids, preprocessor


if __name__ == "__main__":
    # Test preprocessing
    train_df, test_df = load_data()
    X_train, y_train, X_test, train_ids, test_ids, preprocessor = preprocess_data(
        train_df, test_df
    )
    
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # Save preprocessor
    os.makedirs('models', exist_ok=True)
    preprocessor.save('models/preprocessor.pkl')
