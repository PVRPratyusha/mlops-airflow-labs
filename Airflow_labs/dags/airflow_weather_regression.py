"""
Airflow Lab 1 Alternative: Weather Temperature Prediction
A regression pipeline using real weather data from multiple cities
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import requests
import json

# Define paths
DATA_DIR = Path('/opt/airflow/dags/data')
MODEL_DIR = Path('/opt/airflow/dags/model')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def fetch_weather_data(**context):
    """
    Task 1: Fetch real weather data from multiple cities
    """
    print("=" * 70)
    print("TASK 1: Fetching Weather Data from Multiple Cities")
    print("=" * 70)
    
    # Multiple cities for diverse data
    cities = {
        'Boston': (42.36, -71.06),
        'New York': (40.71, -74.01),
        'Chicago': (41.88, -87.63),
        'Los Angeles': (34.05, -118.24),
        'Miami': (25.76, -80.19),
        'Seattle': (47.61, -122.33),
        'Denver': (39.74, -104.99),
        'Phoenix': (33.45, -112.07),
        'Portland': (45.52, -122.68),
        'San Francisco': (37.77, -122.42),
    }
    
    all_data = []
    
    for city_name, (lat, lon) in cities.items():
        print(f"  Fetching data for {city_name}...")
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,pressure_msl,cloud_cover',
            'temperature_unit': 'fahrenheit',
            'past_days': 7,  # Last 7 days
            'forecast_days': 0,
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            weather_data = response.json()
            
            hourly = weather_data['hourly']
            for i in range(len(hourly['time'])):
                all_data.append({
                    'city': city_name,
                    'latitude': lat,
                    'longitude': lon,
                    'timestamp': hourly['time'][i],
                    'temperature': hourly['temperature_2m'][i],
                    'humidity': hourly['relative_humidity_2m'][i],
                    'wind_speed': hourly['wind_speed_10m'][i],
                    'precipitation': hourly['precipitation'][i],
                    'pressure': hourly['pressure_msl'][i],
                    'cloud_cover': hourly['cloud_cover'][i],
                })
        except Exception as e:
            print(f"    Warning: Failed to fetch data for {city_name}: {e}")
            continue
    
    df = pd.DataFrame(all_data)
    
    print(f"\n✓ Weather data collected!")
    print(f"  Total samples: {len(df)}")
    print(f"  Cities: {df['city'].nunique()}")
    print(f"  Features: {list(df.columns)}")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Save raw data
    DATA_DIR.mkdir(exist_ok=True)
    raw_path = DATA_DIR / 'weather_data.csv'
    df.to_csv(raw_path, index=False)
    
    print(f"  Saved to: {raw_path}")
    
    return str(raw_path)

def preprocess_weather_data(**context):
    """
    Task 2: Preprocess and engineer features
    """
    print("=" * 70)
    print("TASK 2: Preprocessing Weather Data")
    print("=" * 70)
    
    raw_path = DATA_DIR / 'weather_data.csv'
    df = pd.read_csv(raw_path)
    
    print(f"Initial shape: {df.shape}")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Feature engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 20)).astype(int)
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"\nMissing values:\n{missing[missing > 0]}")
    
    # Drop rows with missing target (temperature)
    df = df.dropna(subset=['temperature'])
    
    # Fill missing values for features (if any)
    numeric_cols = ['humidity', 'wind_speed', 'precipitation', 'pressure', 'cloud_cover']
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    print(f"\n✓ Preprocessing complete!")
    print(f"  Final shape: {df.shape}")
    print(f"  Temperature range: {df['temperature'].min():.1f}°F - {df['temperature'].max():.1f}°F")
    
    # Statistics
    print(f"\nDataset Statistics:")
    print(df[['temperature', 'humidity', 'wind_speed', 'pressure']].describe())
    
    # Save preprocessed data
    processed_path = DATA_DIR / 'weather_processed.csv'
    df.to_csv(processed_path, index=False)
    
    print(f"\n  Saved to: {processed_path}")
    
    return str(processed_path)

def train_regression_models(**context):
    """
    Task 3: Train multiple regression models
    """
    print("=" * 70)
    print("TASK 3: Training Regression Models")
    print("=" * 70)
    
    processed_path = DATA_DIR / 'weather_processed.csv'
    df = pd.read_csv(processed_path)
    
    # Select features for prediction
    feature_cols = ['latitude', 'longitude', 'humidity', 'wind_speed', 
                   'precipitation', 'pressure', 'cloud_cover', 
                   'hour', 'day_of_week', 'month', 'is_weekend', 'is_daytime']
    
    X = df[feature_cols]
    y = df['temperature']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train Linear Regression
    print("\n" + "-" * 70)
    print("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    lr_train_score = lr_model.score(X_train, y_train)
    lr_test_score = lr_model.score(X_test, y_test)
    
    print(f"  Train R²: {lr_train_score:.4f}")
    print(f"  Test R²: {lr_test_score:.4f}")
    
    # Train Random Forest
    print("\n" + "-" * 70)
    print("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    rf_train_score = rf_model.score(X_train, y_train)
    rf_test_score = rf_model.score(X_test, y_test)
    
    print(f"  Train R²: {rf_train_score:.4f}")
    print(f"  Test R²: {rf_test_score:.4f}")
    
    # Feature importance from Random Forest
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Important Features:")
    print(feature_importance.head().to_string(index=False))
    
    # Save models
    MODEL_DIR.mkdir(exist_ok=True)
    
    lr_path = MODEL_DIR / 'linear_regression.pkl'
    with open(lr_path, 'wb') as f:
        pickle.dump(lr_model, f)
    
    rf_path = MODEL_DIR / 'random_forest.pkl'
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    
    # Save test data for evaluation
    test_data_path = MODEL_DIR / 'test_data.pkl'
    with open(test_data_path, 'wb') as f:
        pickle.dump({'X_test': X_test, 'y_test': y_test}, f)
    
    print(f"\n✓ Models trained and saved!")
    print(f"  Linear Regression: {lr_path}")
    print(f"  Random Forest: {rf_path}")
    
    return str(rf_path)

def evaluate_models(**context):
    """
    Task 4: Evaluate and compare models
    """
    print("=" * 70)
    print("TASK 4: Model Evaluation & Comparison")
    print("=" * 70)
    
    # Load models
    lr_path = MODEL_DIR / 'linear_regression.pkl'
    rf_path = MODEL_DIR / 'random_forest.pkl'
    test_data_path = MODEL_DIR / 'test_data.pkl'
    
    with open(lr_path, 'rb') as f:
        lr_model = pickle.load(f)
    
    with open(rf_path, 'rb') as f:
        rf_model = pickle.load(f)
    
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
        X_test = test_data['X_test']
        y_test = test_data['y_test']
    
    # Predictions
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    # Calculate metrics for Linear Regression
    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_rmse = np.sqrt(lr_mse)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)
    
    # Calculate metrics for Random Forest
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_rmse = np.sqrt(rf_mse)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    # Display comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    comparison = pd.DataFrame({
        'Metric': ['R² Score', 'RMSE (°F)', 'MAE (°F)', 'MSE'],
        'Linear Regression': [f'{lr_r2:.4f}', f'{lr_rmse:.2f}', f'{lr_mae:.2f}', f'{lr_mse:.2f}'],
        'Random Forest': [f'{rf_r2:.4f}', f'{rf_rmse:.2f}', f'{rf_mae:.2f}', f'{rf_mse:.2f}']
    })
    
    print(comparison.to_string(index=False))
    
    # Determine best model
    best_model = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
    best_r2 = max(rf_r2, lr_r2)
    best_rmse = rf_rmse if rf_r2 > lr_r2 else lr_rmse
    
    print("\n" + "=" * 70)
    print(f"🏆 BEST MODEL: {best_model}")
    print("=" * 70)
    print(f"  R² Score: {best_r2:.4f}")
    print(f"  RMSE: {best_rmse:.2f}°F")
    print(f"  Interpretation: Model explains {best_r2*100:.1f}% of temperature variance")
    
    # Sample predictions
    print("\n" + "-" * 70)
    print("SAMPLE PREDICTIONS (First 10)")
    print("-" * 70)
    
    sample_results = pd.DataFrame({
        'Actual': y_test.values[:10],
        'LR Pred': lr_pred[:10],
        'RF Pred': rf_pred[:10],
        'LR Error': np.abs(y_test.values[:10] - lr_pred[:10]),
        'RF Error': np.abs(y_test.values[:10] - rf_pred[:10])
    })
    
    print(sample_results.to_string(index=False))
    
    # Save evaluation results
    results = {
        'linear_regression': {
            'r2': lr_r2,
            'rmse': lr_rmse,
            'mae': lr_mae,
            'mse': lr_mse
        },
        'random_forest': {
            'r2': rf_r2,
            'rmse': rf_rmse,
            'mae': rf_mae,
            'mse': rf_mse
        },
        'best_model': best_model
    }
    
    results_path = MODEL_DIR / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Evaluation complete! Results saved to {results_path}")
    
    return best_model

# Define the DAG
with DAG(
    'Weather_Temperature_Prediction',
    default_args=default_args,
    description='Predict temperature using real weather data from multiple cities',
    schedule_interval=None,
    catchup=False,
    tags=['weather', 'regression', 'machine-learning'],
) as dag:
    
    task_fetch = PythonOperator(
        task_id='fetch_weather_data',
        python_callable=fetch_weather_data,
        provide_context=True,
    )
    
    task_preprocess = PythonOperator(
        task_id='preprocess_weather_data',
        python_callable=preprocess_weather_data,
        provide_context=True,
    )
    
    task_train = PythonOperator(
        task_id='train_regression_models',
        python_callable=train_regression_models,
        provide_context=True,
    )
    
    task_evaluate = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_models,
        provide_context=True,
    )
    
    # Define task dependencies
    task_fetch >> task_preprocess >> task_train >> task_evaluate
