"""
ML Data Preparation Script
Prepares whale data for LSTM training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def prepare_ml_data():
    print("=" * 60)
    print("ML DATA PREPARATION - DAY 1")
    print("=" * 60)
    print()
    
    # Load data
    try:
        df = pd.read_csv('data/whale_data.csv')
        print(f"Loaded {len(df)} whale transactions\n")
    except FileNotFoundError:
        print("ERROR: whale_data.csv not found!")
        return
    
    # Feature engineering
    print("Engineering features...")
    
    # Normalize value_usd (log scale for better ML performance)
    df['value_log'] = np.log10(df['value_usd'])
    
    # Time features
    df['timestamp'] = pd.to_datetime(df['date'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Select features for ML
    feature_columns = [
        'from_exchange',     # 0 or 1
        'to_exchange',       # 0 or 1
        'value_log',         # Log of USD value
        'fear_greed',        # 0-100
        'hour',              # 0-23
        'day_of_week',       # 0-6
    ]
    
    # Prepare X (features) and y (target)
    X = df[feature_columns].values
    y = (df['label'] == 'UP').astype(int).values  # 1 = UP, 0 = DOWN
    
    print(f"Features: {len(feature_columns)}")
    print(f"   - {', '.join(feature_columns)}")
    print(f"\nSamples: {len(X)}")
    print(f"   - UP: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"   - DOWN: {len(y) - y.sum()} ({(len(y) - y.sum())/len(y)*100:.1f}%)")
    
    # Train/Test split (80/20)
    print(f"\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Standardize features
    print(f"\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save processed data
    print(f"\nSaving processed data...")
    
    np.save('data/X_train.npy', X_train_scaled)
    np.save('data/X_test.npy', X_test_scaled)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Saved:")
    print("   - data/X_train.npy")
    print("   - data/X_test.npy")
    print("   - data/y_train.npy")
    print("   - data/y_test.npy")
    print("   - models/scaler.pkl")
    
    # Feature importance preview
    print("\n" + "=" * 60)
    print("FEATURE STATISTICS")
    print("=" * 60)
    
    feature_stats = pd.DataFrame(X_train, columns=feature_columns).describe()
    print(feature_stats)
    
    print("\n" + "=" * 60)
    print("ML DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print("\nNext step: python scripts/train.py (this will take 2-4h)")
    print("TIP: Run overnight for best results!")

if __name__ == "__main__":
    prepare_ml_data()

