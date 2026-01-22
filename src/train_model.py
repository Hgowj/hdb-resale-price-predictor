"""
HDB Resale Price Predictor - Model Training Script
This script trains the machine learning model and saves it for use in the web app.
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import time

def convert_remaining_lease(lease_str):
    """Convert remaining lease string to numeric years"""
    if pd.isna(lease_str):
        return 0
    if isinstance(lease_str, str):
        parts = lease_str.split()
        years = 0
        months = 0
        for i, part in enumerate(parts):
            if part == 'years' and i > 0:
                years = int(parts[i-1])
            elif part == 'months' and i > 0:
                months = int(parts[i-1])
        return years + months/12
    return float(lease_str)

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset"""
    print("Loading data...")
    data = pd.read_csv(file_path)
    print(f"Loaded {len(data)} records")
    
    # Filter data for 2020-2025 only
    print("Filtering data for 2020-2025...")
    data['date'] = pd.to_datetime(data['month'])
    data['year'] = data['date'].dt.year
    
    # Filter for years 2020-2025
    data_filtered = data[(data['year'] >= 2020) & (data['year'] <= 2025)]
    print(f"After filtering to 2020-2025: {len(data_filtered)} records ({len(data) - len(data_filtered)} removed)")
    
    # Show data distribution by year
    year_counts = data_filtered['year'].value_counts().sort_index()
    print("\nData distribution by year:")
    for year, count in year_counts.items():
        print(f"  {year}: {count:,} records")
    
    # Create target variable
    y = data_filtered['resale_price']
    
    # Select features
    features = ['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model', 'remaining_lease']
    X = data_filtered[features].copy()
    
    # Handle categorical variables
    categorical_features = ['town', 'flat_type', 'storey_range', 'flat_model']
    label_encoders = {}
    
    print("Encoding categorical variables...")
    for feature in categorical_features:
        if feature in X.columns:
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature].astype(str))
            label_encoders[feature] = le
            print(f"  {feature}: {len(le.classes_)} unique values")
    
    # Handle remaining_lease
    print("Processing remaining lease...")
    if 'remaining_lease' in X.columns:
        X['remaining_lease'] = X['remaining_lease'].apply(convert_remaining_lease)
    
    # Drop missing values
    original_size = len(X)
    X = X.dropna()
    y = y[X.index]
    print(f"After cleaning: {len(X)} records ({original_size - len(X)} removed)")
    
    return X, y, label_encoders, features

def train_model(X, y):
    """Train the Random Forest model"""
    print("Training model...")
    
    # Split for validation
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, test_size=0.2)
    
    # Create model
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=20,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=1,
        n_jobs=-1
    )
    
    # Train
    start_time = time.time()
    model.fit(train_X, train_y)
    training_time = time.time() - start_time
    
    # Evaluate
    val_predictions = model.predict(val_X)
    val_mae = mean_absolute_error(val_y, val_predictions)
    val_r2 = r2_score(val_y, val_predictions)
    
    print(f"Training completed in {training_time:.1f} seconds")
    print(f"Validation MAE: ${val_mae:,.0f}")
    print(f"R² Score: {val_r2:.3f}")
    
    # Train final model on all data
    print("Training final model on all data...")
    final_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=20,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=1,
        n_jobs=-1
    )
    final_model.fit(X, y)
    
    return final_model, val_mae, val_r2

def save_model(model, label_encoders, features, model_performance, save_path):
    """Save the trained model and associated data"""
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'features': features,
        'performance': model_performance,
        'feature_importance': dict(zip(features, model.feature_importances_))
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {save_path}")

def main():
    """Main training pipeline"""
    print("="*60)
    print("HDB RESALE PRICE PREDICTOR - MODEL TRAINING")
    print("="*60)
    
    # File paths - update this path to your new file
    data_path = r'C:\Users\65892\Documents\Personal Projects\Resale Flat Price Prediction\data\ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv'  # Your new file
    model_save_path = 'models/flat_price_model_2020_2025.pkl'
    
    try:
        # Load and preprocess data
        X, y, label_encoders, features = load_and_preprocess_data(data_path)
        
        # Train model
        model, val_mae, val_r2 = train_model(X, y)
        
        # Show feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print("-" * 30)
        for _, row in feature_importance.iterrows():
            print(f"{row['feature']:15}: {row['importance']:.3f} ({row['importance']*100:.1f}%)")
        
        # Save model
        model_performance = {
            'validation_mae': val_mae,
            'validation_r2': val_r2,
            'training_samples': len(X)
        }
        
        save_model(model, label_encoders, features, model_performance, model_save_path)
        
        print("="*60)
        print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print(f"✅ Trained on 2020-2025 data ({model_performance['training_samples']:,} samples)")
        print(f"✅ Model saved and ready for use in web app")
        print(f"✅ Validation MAE: ${val_mae:,.0f}")
        print(f"✅ R² Score: {val_r2:.3f}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise

if __name__ == "__main__":
    main()