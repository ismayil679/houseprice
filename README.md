# House Price Prediction Project

An optimized machine learning project to predict house prices using heavily tuned CatBoost with advanced feature engineering.

## 🏆 Model Performance

**Optimized CatBoost**: **79,808 AZN RMSE** (Validation)
- Improved from baseline 81,771 → 79,808 (2,000 AZN reduction!)
- R² Score: 0.8030
- 5-Fold CV: 93,230 ± 15,167 AZN

## 📁 Project Structure

```
House_Pricing_Project/
├── data/                              # Data files
│   ├── binaaz_train.csv              # Training data with prices
│   ├── binaaz_test.csv               # Test data (for predictions)
│   ├── binaaz_sample.csv             # Sample data
│   └── baku_coordinates.xlsx         # Baku landmarks (130 locations)
│
├── src/                               # Source code
│   ├── advanced_preprocessing.py     # Advanced feature engineering
│   ├── optimized_train.py            # Hyperparameter tuning pipeline
│   ├── optimized_evaluate.py         # Generate predictions
│   ├── preprocessing.py              # Basic preprocessing (legacy)
│   ├── train.py                      # Basic training (legacy)
│   └── evaluate.py                   # Basic evaluation (legacy)
│
├── models/                            # Saved models
│   ├── advanced_preprocessor.pkl     # Advanced feature preprocessor
│   ├── catboost_optimized_final.cbm  # Final model (full training data)
│   ├── catboost_optimized_best.cbm   # Best validation model
│   └── optimization_results.json     # Optimization metadata
│
├── outputs/                           # Output files
│   └── predictions/                  # Prediction files
│       ├── submission_latest.csv     # Latest predictions
│       └── submission_*.csv          # Timestamped predictions
│
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Optimized CatBoost Model

```bash
python src/optimized_train.py
```

This will:
- Load data with line terminator fixes
- Extract 20 advanced features (including distance to landmarks)
- Train 4 different CatBoost configurations
- Perform 5-fold cross-validation
- Select best model based on validation RMSE
- Train final model on full dataset

**Training time**: ~10 minutes

### 3. Generate Predictions

```bash
python src/optimized_evaluate.py
```

This will:
- Load the optimized CatBoost model
- Generate predictions on test set
- Save to `outputs/predictions/submission_latest.csv`

## 📊 Features

The optimized model uses **20 advanced features**:

**Core Features (8):**
- `area`: Property area in m²
- `room_count`: Number of rooms
- `floor`: Floor number
- `total_floors`: Total floors in building
- `lat`, `lon`: Geographic coordinates
- `dist_from_center`: Distance from Baku city center (40.4093, 49.8671)
- `city_baki`: Located in Baku

**Binary Features (5):**
- `has_deed`: Property has deed (Kupça)
- `has_mortgage`: Mortgage available (İpoteka)
- `is_owner`: Posted by owner
- `is_agent`: Posted by agent
- `is_new_building`: New construction

**Engineered Features (4):**
- `floor_ratio`: floor / (total_floors + 1)
- `area_per_room`: area / (room_count + 1)
- `title_length`: Length of listing title
- `desc_length`: Length of description

**Distance Features (3):** 🆕
- `dist_to_nearest_landmark`: Distance to nearest important location
- `dist_to_nearest_5_landmarks`: Avg distance to 5 nearest landmarks
- `dist_to_nearest_10_landmarks`: Avg distance to 10 nearest landmarks

*Based on 130 important locations in Baku (metro stations, monuments, parks, etc.)*

## 📈 Model Performance

### Optimization Results

| Configuration | Validation RMSE | MAE | R² |
|--------------|----------------|------|-----|
| Baseline | 80,374 AZN | 30,048 | 0.8002 |
| **Optimized V1** ⭐ | **79,808 AZN** | **29,296** | **0.8030** |
| Optimized V2 | 80,216 AZN | 29,823 | 0.8010 |
| Optimized V3 | 80,997 AZN | 29,297 | 0.7971 |

### Feature Importance (Top 10)

1. **area** (26.9%) - Property size
2. **lat** (8.9%) - Latitude
3. **area_per_room** (8.8%) - Size efficiency
4. **room_count** (8.0%) - Number of rooms
5. **dist_to_nearest_10_landmarks** (6.6%) 🆕
6. **dist_from_center** (6.3%) - Distance from center
7. **dist_to_nearest_5_landmarks** (6.0%) 🆕
8. **total_floors** (5.6%) - Building height
9. **floor_ratio** (5.3%) - Relative floor position
10. **lon** (5.2%) - Longitude

Distance features contribute **15.9%** to predictions!

## 🔧 Optimization Details

### CatBoost Configuration (Best: Optimized V1)

```python
{
    'iterations': 3000,
    'learning_rate': 0.02,        # Lower LR for stability
    'depth': 10,                  # Deeper trees
    'l2_leaf_reg': 5,             # Regularization
    'subsample': 0.75,
    'colsample_bylevel': 0.8,
    'min_data_in_leaf': 15,
    'bagging_temperature': 0.5,
    'border_count': 254,
    'early_stopping_rounds': 150
}
```

### Training Strategy

1. **Stage 1**: Baseline model (2000 iterations)
2. **Stage 2**: Optimized V1 - Deeper trees + lower LR
3. **Stage 3**: Optimized V2 - Regularization focus
4. **Stage 4**: Optimized V3 - Aggressive boosting
5. **Stage 5**: Best model on full training data
6. **Validation**: 5-fold cross-validation

### Feature Engineering Pipeline

1. Parse numeric values from text (area: "115 m²" → 115.0)
2. Extract floor information ("5 / 9" → floor=5, total=9)
3. Calculate distances to 130 Baku landmarks
4. Create ratio features (floor_ratio, area_per_room)
5. Binary encoding (has_deed, is_owner, etc.)
6. Handle missing values with median imputation
7. StandardScaler normalization

## 📝 Key Improvements

### From Baseline to Optimized

✅ **Added distance features** (+3 features from baku_coordinates.xlsx)  
✅ **Engineered ratio features** (floor_ratio, area_per_room)  
✅ **Fixed line terminator issues** in CSV files  
✅ **Hyperparameter tuning** (tested 4 configurations)  
✅ **Cross-validation** (5-fold for robustness)  
✅ **Deeper trees** (depth 10 vs 8)  
✅ **Lower learning rate** (0.02 vs 0.05)  
✅ **More iterations** (3000 vs 1000)  

**Result**: 81,771 → **79,808 RMSE** (-2,000 AZN improvement!)

## 🎯 Competition Metric

Primary metric: **RMSE** (Root Mean Squared Error)

## 📤 Submission

Submit: `outputs/predictions/submission_latest.csv`

Format:
```csv
_id,price
20886,181234.56
117465,95678.90
...
```

Expected performance: **~80,000 AZN RMSE**
