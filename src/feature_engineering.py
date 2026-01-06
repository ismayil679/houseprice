"""
Feature Engineering for Baku House Pricing
===========================================
Creates distance-based and location features using baku_coordinates.xlsx landmarks.

New Features:
- distance_to_city_center: Distance to Baku center (Fountain Square area)
- distance_to_nearest_metro: Distance to closest metro station
- metro_name: Name of nearest metro (for target encoding)
- distance_to_nearest_landmark: Distance to closest non-metro landmark
- num_landmarks_within_2km: Count of landmarks within 2km radius
- avg_distance_top5_landmarks: Average distance to 5 closest landmarks
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import pickle
import warnings
warnings.filterwarnings('ignore')

# Haversine formula for distance calculation
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine for numpy arrays."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


# Metro station name mapping: Azerbaijani -> Transliterated (landmarks file)
METRO_NAME_MAPPING = {
    # Azerbaijani name in training data -> Name in landmarks file
    "İçəri Şəhər": "Icheri Sheher",
    "Sahil": "Sahil",
    "28 May": "28 May",
    "Gənclik": "Ganjlik",
    "Nəriman Nərimanov": "Nariman Narimanov",
    "Bakmil": "Bakmil",
    "Ulduz": "Ulduz",
    "Koroğlu": "Koroglu",
    "Qara Qarayev": "Gara Garayev",
    "Neftçilər": "Neftchilar",
    "Xalqlar Dostluğu": "Halglar Doslugu",
    "Əhmədli": "Ahmedli",
    "Həzi Aslanov": "Hazi Aslanov",
    "Nizami": "Nizami Ganjavi",
    "Elmlər Akademiyası": "Elmlar Akademiyası",
    "İnşaatçılar": "Inshaatchilar",
    "20 Yanvar": "20 Yanvar",
    "Memar Əcəmi": "Memar Ajami",
    "Nəsimi": "Nasimi",
    "Azadlıq Prospekti": "Azadlig Prospekti",
    "Dərnəgül": "Darnagul",
    "Şah İsmayıl Xətai": "Shah Ismail Hatai",
    "Avtovağzal": "Avtovagzal",
}

# Baku city center (Fountain Square / İçərişəhər area)
BAKU_CENTER = (40.3667, 49.8352)

def load_landmarks():
    """Load and process landmarks from Excel file."""
    landmarks = pd.read_excel('baku_coordinates.xlsx')
    
    # Metro station landmarks (last 24 rows are metro stations)
    metro_names = list(METRO_NAME_MAPPING.values())
    metro_df = landmarks[landmarks['Title'].isin(metro_names)].copy()
    
    # Non-metro landmarks
    other_df = landmarks[~landmarks['Title'].isin(metro_names)].copy()
    
    return landmarks, metro_df, other_df


def extract_metro_from_locations(locations_str):
    """Extract metro station name from locations string."""
    if pd.isna(locations_str):
        return None
    
    parts = str(locations_str).split('\n')
    for p in parts:
        p = p.strip()
        if p.endswith(' m.'):
            return p.replace(' m.', '')
    return None


def create_distance_features(df, landmarks_df, metro_df, other_landmarks_df):
    """
    Create all distance-based features for a dataframe.
    OPTIMIZED: Uses matrix operations instead of row-by-row iteration.
    
    Parameters:
    -----------
    df : pd.DataFrame - Input data with 'latitude', 'longitude' columns
    landmarks_df : pd.DataFrame - All landmarks
    metro_df : pd.DataFrame - Metro station landmarks only
    other_landmarks_df : pd.DataFrame - Non-metro landmarks
    
    Returns:
    --------
    pd.DataFrame with new features added
    """
    df = df.copy()
    n = len(df)
    
    # Get coordinates as arrays and CLIP to Baku area
    # Baku coordinates: lat 39.5-41.0, lon 49.0-50.5
    prop_lats = df['latitude'].values.copy()
    prop_lons = df['longitude'].values.copy()
    
    # Clip outliers to Baku center
    baku_lat, baku_lon = BAKU_CENTER
    invalid_mask = (prop_lats < 39.5) | (prop_lats > 41.0) | \
                   (prop_lons < 49.0) | (prop_lons > 50.5) | \
                   np.isnan(prop_lats) | np.isnan(prop_lons)
    prop_lats[invalid_mask] = baku_lat
    prop_lons[invalid_mask] = baku_lon
    
    print(f"  Clipped {invalid_mask.sum()} invalid coordinates to Baku center")
    
    all_lats = landmarks_df['Latitude'].values
    all_lons = landmarks_df['Longitude'].values
    metro_lats = metro_df['Latitude'].values
    metro_lons = metro_df['Longitude'].values
    metro_names_list = metro_df['Title'].values
    
    # 1. Distance to city center (vectorized)
    df['distance_to_center'] = haversine_vectorized(
        prop_lats, prop_lons, BAKU_CENTER[0], BAKU_CENTER[1]
    )
    
    # 2. Extract metro name from locations column
    df['metro_name'] = df['locations'].apply(extract_metro_from_locations)
    
    # 3-6. Compute distance matrix: properties x landmarks
    # For efficiency, process in batches
    print("  Computing distance matrices...")
    
    batch_size = 5000
    nearest_metro_dist = np.zeros(n)
    nearest_metro_idx = np.zeros(n, dtype=int)
    nearest_landmark_dist = np.zeros(n)
    landmarks_within_2km = np.zeros(n, dtype=int)
    avg_dist_top5 = np.zeros(n)
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_lats = prop_lats[start:end]
        batch_lons = prop_lons[start:end]
        batch_n = end - start
        
        # Distance to all landmarks (batch_n x num_landmarks)
        dist_to_all = np.zeros((batch_n, len(all_lats)))
        for j in range(len(all_lats)):
            dist_to_all[:, j] = haversine_vectorized(batch_lats, batch_lons, all_lats[j], all_lons[j])
        
        # Distance to metros (batch_n x num_metros)
        dist_to_metros = np.zeros((batch_n, len(metro_lats)))
        for j in range(len(metro_lats)):
            dist_to_metros[:, j] = haversine_vectorized(batch_lats, batch_lons, metro_lats[j], metro_lons[j])
        
        # Compute features for this batch
        nearest_metro_dist[start:end] = dist_to_metros.min(axis=1)
        nearest_metro_idx[start:end] = dist_to_metros.argmin(axis=1)
        nearest_landmark_dist[start:end] = dist_to_all.min(axis=1)
        landmarks_within_2km[start:end] = (dist_to_all <= 2.0).sum(axis=1)
        
        # Top 5 average
        sorted_dists = np.sort(dist_to_all, axis=1)
        avg_dist_top5[start:end] = sorted_dists[:, :5].mean(axis=1)
        
        if start % 20000 == 0:
            print(f"    Processed {end}/{n} rows...")
    
    df['distance_to_nearest_metro'] = nearest_metro_dist
    df['nearest_metro_landmark'] = [metro_names_list[i] for i in nearest_metro_idx]
    df['distance_to_nearest_landmark'] = nearest_landmark_dist
    df['num_landmarks_within_2km'] = landmarks_within_2km
    df['avg_distance_top5_landmarks'] = avg_dist_top5
    
    # 7. Distance to Old City
    old_city = landmarks_df[landmarks_df['Title'].str.contains('Icheri|İçəri', case=False, na=False)]
    if len(old_city) > 0:
        old_city_lat = old_city['Latitude'].values[0]
        old_city_lon = old_city['Longitude'].values[0]
        df['distance_to_old_city'] = haversine_vectorized(
            prop_lats, prop_lons, old_city_lat, old_city_lon
        )
    
    return df


def main():
    """Test the feature engineering on training data."""
    print("Loading data...")
    train = pd.read_csv('data/binaaz_train.csv')
    test = pd.read_csv('data/binaaz_test.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    print("\nLoading landmarks...")
    landmarks_df, metro_df, other_df = load_landmarks()
    print(f"Total landmarks: {len(landmarks_df)}")
    print(f"Metro stations: {len(metro_df)}")
    print(f"Other landmarks: {len(other_df)}")
    
    print("\nCreating distance features for training data...")
    train_fe = create_distance_features(train, landmarks_df, metro_df, other_df)
    
    print("\nNew features summary:")
    new_cols = ['distance_to_center', 'distance_to_nearest_metro', 
                'distance_to_nearest_landmark', 'num_landmarks_within_2km',
                'avg_distance_top5_landmarks', 'distance_to_old_city', 'metro_name']
    
    for col in new_cols:
        if col in train_fe.columns:
            if train_fe[col].dtype in ['float64', 'int64']:
                print(f"  {col}: mean={train_fe[col].mean():.3f}, std={train_fe[col].std():.3f}")
            else:
                print(f"  {col}: {train_fe[col].nunique()} unique values, {train_fe[col].isna().sum()} missing")
    
    # Show correlation with price
    print("\nCorrelation with price:")
    for col in new_cols:
        if col in train_fe.columns and train_fe[col].dtype in ['float64', 'int64']:
            corr = train_fe[col].corr(train_fe['price'])
            print(f"  {col}: {corr:.4f}")
    
    # Save for inspection
    train_fe.to_csv('data/train_with_distance_features.csv', index=False)
    print("\nSaved to data/train_with_distance_features.csv")
    

if __name__ == "__main__":
    main()
