"""Machine Learning functions for Airflow Lab 1 - K-means clustering"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle
from pathlib import Path

# Define paths
DATA_DIR = Path('/opt/airflow/dags/data')
MODEL_DIR = Path('/opt/airflow/dags/model')

def load_data(**context):
    """Task 1: Load data from CSV file"""
    print("=" * 60)
    print("TASK 1: Loading Data")
    print("=" * 60)
    
    file_path = DATA_DIR / 'file.csv'
    df = pd.read_csv(file_path)
    
    print(f"✓ Data loaded successfully!")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    return str(file_path)

def data_preprocessing(**context):
    """Task 2: Preprocess the data"""
    print("=" * 60)
    print("TASK 2: Data Preprocessing")
    print("=" * 60)
    
    file_path = DATA_DIR / 'file.csv'
    df = pd.read_csv(file_path)
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"\nMissing values:")
    print(missing)
    
    # Basic statistics
    print(f"\nData Statistics:")
    print(df.describe())
    
    # Handle missing values if any
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        print(f"✓ Dropped rows with missing values")
    
    print(f"✓ Preprocessing complete!")
    print(f"  Final shape: {df.shape}")
    
    return True

def build_save_model(**context):
    """Task 3: Build and save K-means clustering model"""
    print("=" * 60)
    print("TASK 3: Building K-means Model")
    print("=" * 60)
    
    # Load data
    file_path = DATA_DIR / 'file.csv'
    df = pd.read_csv(file_path)
    X = df.values
    
    # Build K-means model with 3 clusters
    n_clusters = 3
    print(f"\nTraining K-means with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / 'model.sav'
    
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans, f)
    
    print(f"✓ Model trained and saved!")
    print(f"  Model path: {model_path}")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    print(f"  Cluster centers shape: {kmeans.cluster_centers_.shape}")
    
    return str(model_path)

def load_model_elbow(**context):
    """Task 4: Load model and determine optimal clusters using elbow method"""
    print("=" * 60)
    print("TASK 4: Elbow Method for Optimal Clusters")
    print("=" * 60)
    
    # Load data
    file_path = DATA_DIR / 'file.csv'
    df = pd.read_csv(file_path)
    X = df.values
    
    # Calculate inertia for different cluster numbers
    print("\nCalculating inertia for k=1 to k=10...")
    inertias = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        print(f"  k={k}: inertia={kmeans.inertia_:.2f}")
    
    # Find elbow point (simple method - look for biggest drop)
    differences = np.diff(inertias)
    elbow_point = np.argmin(differences) + 2
    
    print("\n" + "=" * 60)
    print(f"📊 OPTIMAL NUMBER OF CLUSTERS: {elbow_point}")
    print("=" * 60)
    print(f"\nInertia values:")
    for k, inertia in zip(k_range, inertias):
        marker = " ← ELBOW" if k == elbow_point else ""
        print(f"  k={k}: {inertia:.2f}{marker}")
    
    # Load saved model
    model_path = MODEL_DIR / 'model.sav'
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    print(f"\n✓ Model loaded from: {model_path}")
    print(f"  Model has {loaded_model.n_clusters} clusters")
    
    return elbow_point
