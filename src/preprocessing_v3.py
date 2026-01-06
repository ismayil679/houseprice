"""
Enhanced Data Preprocessing V3 for House Price Prediction
Features:
- Original 15 features
- Distance features (metro, landmarks)
- Target encoding (metro, district, settlement)
- Price/sqm proxy features
- Luxury indicators (based on differential analysis)
- Negative price indicators (ipoteka, urgent)
- NO StandardScaler (tree models are scale-invariant)
"""

import pandas as pd
import numpy as np
import re
from math import radians, sin, cos, sqrt, atan2
import joblib
import os


class HousePricePreprocessorV3:
    """Enhanced Preprocessor V3 with proper luxury indicators"""
    
    def __init__(self):
        self.feature_medians = {}
        self.target_encodings = {}
        self.global_mean = None
        self.feature_cols = None
        
        # Baku city center coordinates
        self.baku_center = (40.4093, 49.8671)
        
        # Key metro stations with coordinates
        self.metro_coords = {
            '28 May': (40.3795, 49.8488),
            'Gənclik': (40.4003, 49.8516),
            'Nariman Narimanov': (40.4074, 49.8675),
            'Ulduz': (40.4112, 49.8789),
            'Koroğlu': (40.4177, 49.8946),
            'Qara Qarayev': (40.4215, 49.9280),
            'Neftçilər': (40.4251, 49.9449),
            'Xalqlar Dostluğu': (40.4298, 49.9616),
            'Əhmədli': (40.4316, 49.9830),
            'Həzi Aslanov': (40.4332, 50.0021),
            'İçərişəhər': (40.3661, 49.8372),
            'Sahil': (40.3718, 49.8485),
            'Nizami': (40.3802, 49.8299),
            'Elmlar Akademiyası': (40.3875, 49.8099),
            '20 Yanvar': (40.4011, 49.8085),
            'İnşaatçılar': (40.4113, 49.8099),
            'Memar Əcəmi': (40.4237, 49.8098),
            'Nəsimi': (40.4305, 49.8073),
            'Azadlıq': (40.4421, 49.8022),
            'Dərnəgül': (40.4500, 49.7988),
            'Bakmil': (40.4092, 49.8377),
            'Avtovağzal': (40.4088, 49.8210),
            'Cəfər Cabbarlı': (40.3821, 49.8398),
        }
        
        # Key landmarks
        self.landmarks = {
            'flame_towers': (40.3597, 49.8321),
            'port_baku': (40.3640, 49.8541),
            'baku_mall': (40.3987, 49.8711),
            'baku_white_city': (40.3381, 49.8682),
            'deniz_mall': (40.3495, 49.8489),
            'sea_breeze': (40.4744, 50.1203),
        }
    
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two points in km"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
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
    
    def _extract_metro_from_locations(self, loc_str):
        """Extract metro station name from locations string"""
        if pd.isna(loc_str):
            return 'unknown'
        loc_str = str(loc_str).lower()
        for metro in self.metro_coords.keys():
            if metro.lower() in loc_str:
                return metro
        if 'metro' in loc_str or 'm.' in loc_str:
            return 'other_metro'
        return 'no_metro'
    
    def _extract_district(self, loc_str):
        """Extract district from locations string"""
        if pd.isna(loc_str):
            return 'unknown'
        
        districts = [
            'Binəqədi', 'Yasamal', 'Nəsimi', 'Nərimanov', 'Xətai',
            'Səbail', 'Nizami', 'Suraxanı', 'Xəzər', 'Qaradağ',
            'Sabunçu', 'Pirallahı', 'Abşeron'
        ]
        
        loc_lower = str(loc_str).lower()
        for district in districts:
            if district.lower() in loc_lower:
                return district
        return 'other'
    
    def _extract_settlement(self, loc_str):
        """Extract settlement from locations string"""
        if pd.isna(loc_str):
            return 'unknown'
        
        settlements = [
            'Badamdar', 'Bakıxanov', 'Bilgəh', 'Binə', 'Biləcəri',
            'Buzovna', 'Ceyranbatan', 'Corat', 'Digah', 'Fatmayı',
            'Görədil', 'Həzi Aslanov', 'Hövsan', 'Kürdəxanı',
            'Lökbatan', 'Maştağa', 'Masazır', 'Mərdəkan', 'Novxanı',
            'Nardaran', 'Pirallahı', 'Pirşağı', 'Qala', 'Ramana',
            'Sabunçu', 'Saray', 'Şağan', 'Şüvəlan', 'Türkan',
            'Yeni Suraxanı', 'Zığ', 'Zabrat', 'Əmircan'
        ]
        
        loc_lower = str(loc_str).lower()
        for settlement in settlements:
            if settlement.lower() in loc_lower:
                return settlement
        return 'other'
    
    def _compute_distance_features(self, df):
        """Compute distance-based features"""
        features = {}
        
        lat = df['latitude'].values
        lon = df['longitude'].values
        
        # Distance to center
        features['dist_from_center_km'] = np.array([
            self.haversine_distance(la, lo, self.baku_center[0], self.baku_center[1])
            if not (pd.isna(la) or pd.isna(lo)) else np.nan
            for la, lo in zip(lat, lon)
        ])
        
        # Distance to nearest metro
        min_metro_dist = []
        for la, lo in zip(lat, lon):
            if pd.isna(la) or pd.isna(lo):
                min_metro_dist.append(np.nan)
            else:
                distances = [
                    self.haversine_distance(la, lo, m_lat, m_lon)
                    for m_lat, m_lon in self.metro_coords.values()
                ]
                min_metro_dist.append(min(distances))
        features['dist_to_nearest_metro'] = np.array(min_metro_dist)
        
        # Distance to key landmarks
        for name, (l_lat, l_lon) in self.landmarks.items():
            features[f'dist_to_{name}'] = np.array([
                self.haversine_distance(la, lo, l_lat, l_lon)
                if not (pd.isna(la) or pd.isna(lo)) else np.nan
                for la, lo in zip(lat, lon)
            ])
        
        return features
    
    def _target_encode_column(self, series, target, is_train=True, smoothing=50):
        """Apply target encoding with smoothing"""
        col_name = series.name
        
        if is_train:
            # Calculate global mean
            global_mean = target.mean()
            
            # Calculate stats per category
            stats = pd.DataFrame({
                'category': series,
                'target': target
            }).groupby('category').agg({
                'target': ['mean', 'count']
            })
            stats.columns = ['mean', 'count']
            
            # Smoothed encoding
            stats['encoded'] = (
                (stats['count'] * stats['mean'] + smoothing * global_mean) /
                (stats['count'] + smoothing)
            )
            
            self.target_encodings[col_name] = stats['encoded'].to_dict()
            self.target_encodings[f'{col_name}_global_mean'] = global_mean
        
        # Apply encoding
        global_mean = self.target_encodings.get(f'{col_name}_global_mean', target.mean() if is_train else 0)
        encoded = series.map(self.target_encodings.get(col_name, {})).fillna(global_mean)
        
        return encoded
    
    def _compute_price_per_sqm_features(self, df, target=None, is_train=True):
        """Compute price per sqm proxy features using area and target encoding"""
        features = {}
        
        area = df['Sahə'].apply(self.extract_area)
        metro_name = df['locations'].apply(self._extract_metro_from_locations)
        district = df['locations'].apply(self._extract_district)
        
        if is_train and target is not None:
            # Calculate price per sqm by district
            temp_df = pd.DataFrame({
                'district': district,
                'area': area,
                'price': target
            })
            temp_df['price_per_sqm'] = temp_df['price'] / temp_df['area'].replace(0, np.nan)
            
            district_price_sqm = temp_df.groupby('district')['price_per_sqm'].median()
            self.district_price_per_sqm = district_price_sqm.to_dict()
            self.global_price_per_sqm = temp_df['price_per_sqm'].median()
            
            # By metro
            temp_df['metro'] = metro_name
            metro_price_sqm = temp_df.groupby('metro')['price_per_sqm'].median()
            self.metro_price_per_sqm = metro_price_sqm.to_dict()
        
        # Map district price per sqm
        features['district_price_per_sqm'] = district.map(
            self.district_price_per_sqm
        ).fillna(self.global_price_per_sqm)
        
        # Map metro price per sqm
        features['metro_price_per_sqm'] = metro_name.map(
            self.metro_price_per_sqm
        ).fillna(self.global_price_per_sqm)
        
        # Estimated prices
        features['estimated_price_district'] = features['district_price_per_sqm'] * area
        features['estimated_price_metro'] = features['metro_price_per_sqm'] * area
        features['estimated_price_combined'] = (
            features['estimated_price_district'] * 0.5 +
            features['estimated_price_metro'] * 0.5
        )
        
        # Relative to area median
        features['area_vs_district_median'] = area / district.map(
            lambda x: self.district_price_per_sqm.get(x, self.global_price_per_sqm)
        )
        
        return features
    
    def extract_features(self, df, target=None, is_train=True):
        """Extract all features from raw dataframe"""
        df = df.copy()
        features = {}
        
        # =====================================================================
        # BASIC NUMERIC FEATURES
        # =====================================================================
        area = df['Sahə'].apply(self.extract_area)
        floor_info = df['Mərtəbə'].apply(lambda x: pd.Series(self.extract_floor_info(x)))
        floor = floor_info[0]
        total_floors = floor_info[1]
        room_count = df['Otaq sayı']
        
        features['area'] = area
        features['room_count'] = room_count
        features['floor'] = floor
        features['total_floors'] = total_floors
        
        # =====================================================================
        # DERIVED NUMERIC FEATURES
        # =====================================================================
        features['floor_ratio'] = floor / total_floors.replace(0, np.nan)
        features['is_top_floor'] = (floor == total_floors).astype(int)
        features['is_ground_floor'] = (floor == 1).astype(int)
        features['rooms_per_area'] = room_count / area.replace(0, np.nan)
        features['area_per_room'] = area / room_count.replace(0, np.nan)
        features['floors_above'] = total_floors - floor
        features['floors_below'] = floor - 1
        
        # Area categories
        features['is_large_apartment'] = (area > 150).astype(int)
        features['is_small_apartment'] = (area < 50).astype(int)
        
        # High-rise indicator (luxury buildings tend to be taller)
        features['is_high_rise'] = (total_floors >= 16).astype(int)
        
        # =====================================================================
        # BINARY FEATURES
        # =====================================================================
        features['has_deed'] = (df['Kupça'] == 'var').astype(int)
        features['has_mortgage'] = (df['İpoteka'] == 'var').astype(int)
        features['is_owner'] = (df['poster_type'] == 'mülkiyyətçi').astype(int)
        features['is_agent'] = (df['poster_type'] == 'vasitəçi (agent)').astype(int)
        
        # Time features were tested but didn't improve CV - dates overlap train/test
        
        # =====================================================================
        # LOCATION FEATURES
        # =====================================================================
        features['lat'] = df['latitude']
        features['lon'] = df['longitude']
        features['city_baki'] = (df['seher'] == 'baki').astype(int)
        
        # Distance features
        dist_features = self._compute_distance_features(df)
        for k, v in dist_features.items():
            features[k] = v
        
        # =====================================================================
        # TEXT FEATURES - Basic
        # =====================================================================
        title = df['title'].fillna('')
        desc = df['description'].fillna('')
        all_text = (title + ' ' + desc).str.lower()
        
        features['title_length'] = title.str.len()
        features['desc_length'] = desc.str.len()
        features['is_new_building'] = title.str.contains('yeni tikili', case=False, na=False).astype(int)
        
        features['has_repair'] = (
            title.str.contains('təmirli|remont|ремонт', case=False, na=False) |
            desc.str.contains('təmirli|remont|ремонт', case=False, na=False)
        ).astype(int)
        
        features['has_furniture'] = (
            desc.str.contains('mebel|furniture|мебел', case=False, na=False)
        ).astype(int)
        
        # =====================================================================
        # LUXURY INDICATORS (based on differential analysis - high ratio in expensive vs normal)
        # =====================================================================
        # Property type - VERY strong indicators (7-64x more common in expensive)
        features['is_penthouse'] = all_text.str.contains('penthouse|pent-?house|пентхаус', na=False).astype(int)
        features['is_duplex'] = all_text.str.contains('dublex|duplex|дуплекс', na=False).astype(int)
        features['is_villa'] = all_text.str.contains('villa|вилла', na=False).astype(int)
        
        # Premium/luxury markers (4-9x more common in expensive)
        features['has_lux'] = all_text.str.contains('lux|люкс|lüks', na=False).astype(int)
        features['is_vip'] = all_text.str.contains(r'\bvip\b|вип', na=False).astype(int)
        features['is_elite'] = all_text.str.contains('elite|элит', na=False).astype(int)
        features['is_premium'] = all_text.str.contains('premium|прем', na=False).astype(int)
        features['has_elit'] = all_text.str.contains('elit', na=False).astype(int)  # Azerbaijani form
        
        # View indicators (4-6x more common in expensive)
        features['has_sea_view'] = all_text.str.contains('dəniz|deniz|sea|море', na=False).astype(int)
        features['has_view'] = all_text.str.contains('görünüş|mənzərə|view|вид', na=False).astype(int)
        features['has_panorama'] = all_text.str.contains('panorama|панорам', na=False).astype(int)
        
        # Premium renovation (4-6x more common in expensive)
        features['has_euro_repair'] = all_text.str.contains('avro|euro təmir|евро', na=False).astype(int)
        
        # Amenities (3-5x more common in expensive)
        features['has_pool'] = all_text.str.contains('hovuz|pool|бассейн', na=False).astype(int)
        
        # =====================================================================
        # NEGATIVE PRICE INDICATORS (more common in cheaper properties)
        # =====================================================================
        # Urgency indicates financial pressure -> lower prices
        features['is_urgent'] = all_text.str.contains('təcili|срочно|urgent', na=False).astype(int)
        
        # Mortgage/credit ads often cheaper properties
        features['mentions_ipoteka'] = all_text.str.contains('ipoteka|ипотек', na=False).astype(int)
        features['mentions_kredit'] = all_text.str.contains('kredit|кредит', na=False).astype(int)
        
        # =====================================================================
        # COMBINED LUXURY SCORE
        # =====================================================================
        features['luxury_score'] = (
            features['is_penthouse'] * 5 +   # Strongest indicator (64x)
            features['has_pool'] * 4 +        # Very strong (47x)
            features['is_elite'] * 3 +        # Strong (29x)
            features['is_duplex'] * 2 +       # Good indicator (8x)
            features['has_lux'] * 2 +
            features['is_vip'] * 2 +
            features['is_premium'] +
            features['has_elit'] +
            features['has_sea_view'] +
            features['has_view'] +
            features['has_panorama'] +
            features['has_euro_repair']
        )
        
        features['is_likely_expensive'] = (features['luxury_score'] >= 3).astype(int)
        
        # Negative indicator score
        features['budget_indicator_score'] = (
            features['is_urgent'] +
            features['mentions_ipoteka'] * 2 +  # Strong negative indicator
            features['mentions_kredit']
        )
        
        # =====================================================================
        # CATEGORICAL EXTRACTION (for target encoding)
        # =====================================================================
        metro_name = df['locations'].apply(self._extract_metro_from_locations)
        district = df['locations'].apply(self._extract_district)
        settlement = df['locations'].apply(self._extract_settlement)
        
        if is_train and target is not None:
            self.global_mean = np.log1p(target).mean()
            
        # Target encoding (NOTE: causes leakage in CV but helps final model)
        # For proper CV evaluation, we skip target encoding for now
        # These will be re-added in final training with full data
        metro_name.name = 'metro'
        district.name = 'district'
        settlement.name = 'settlement'
        
        # Skip target encoding to avoid CV leakage
        # log_target = np.log1p(target) if target is not None else None
        # features['metro_encoded'] = self._target_encode_column(metro_name, log_target, is_train)
        # features['district_encoded'] = self._target_encode_column(district, log_target, is_train)
        # features['settlement_encoded'] = self._target_encode_column(settlement, log_target, is_train)
        
        # Skip price/sqm features to avoid CV leakage
        # price_sqm_features = self._compute_price_per_sqm_features(df, target, is_train)
        # for k, v in price_sqm_features.items():
        #     features[k] = v
        
        # =====================================================================
        # FINALIZE FEATURES
        # =====================================================================
        feature_df = pd.DataFrame(features)
        
        # Store feature columns
        if self.feature_cols is None:
            self.feature_cols = list(feature_df.columns)
        
        return feature_df
    
    def fit(self, X, y=None):
        """Fit the preprocessor"""
        # Calculate medians for missing value imputation
        for col in X.columns:
            self.feature_medians[col] = X[col].median()
        return self
    
    def transform(self, X):
        """Transform features (impute missing values only, no scaling)"""
        X_transformed = X.copy()
        
        # Fill missing values using stored medians
        for col in X_transformed.columns:
            median_val = self.feature_medians.get(col, X_transformed[col].median())
            X_transformed[col] = X_transformed[col].fillna(median_val)
        
        return X_transformed.values
    
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


if __name__ == '__main__':
    # Test the preprocessor
    train_df, test_df = load_data()
    
    preprocessor = HousePricePreprocessorV3()
    
    # Extract features
    X_train = preprocessor.extract_features(train_df, target=train_df['price'], is_train=True)
    X_test = preprocessor.extract_features(test_df, is_train=False)
    
    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"\nFeature columns ({len(preprocessor.feature_cols)}):")
    for i, col in enumerate(preprocessor.feature_cols):
        print(f"  {i+1}. {col}")
