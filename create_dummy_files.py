#import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
#import os

MODEL_FILENAME = 'fraud_rf_model.pkl'
STATUSES = ['submitted', 'accepted', 'rejected']
NUM_VENDORS = 100

def generate_random_timestamp(start_date, end_date):
    delta = end_date - start_date
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start_date + timedelta(seconds=random_seconds)

def generate_dataset(size):
    """Generates a synthetic dataset similar to fraud_detection_train.py's output."""
    data = {
        'customer_id': np.arange(1, size + 1),
        'transaction_id': [f"tx_{i}" for i in np.arange(1, size + 1)], # Added for queue unique ID
        'timestamp': [generate_random_timestamp(datetime(2022, 1, 1), datetime(2024, 1, 1)) for _ in range(size)],
        'status': np.random.choice(STATUSES, size=size),
        'vendor_id': np.random.randint(1, NUM_VENDORS + 1, size=size),
        'amount': np.round(np.random.uniform(10.0, 1000.0, size), 2),
        'fraudulent': np.random.choice([0, 1], size=size, p=[0.85, 0.15]) # Imbalanced fraud
    }
    return pd.DataFrame(data)

def preprocess_data(df):
    df_processed = df.copy()
    df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp']).astype(int) / 10**9
    le_status = LabelEncoder()
    # Fit encoder on all possible statuses to ensure consistent mapping
    le_status.fit(STATUSES)
    df_processed['status'] = le_status.transform(df_processed['status'])
    return df_processed, le_status

def train_and_save_model(df):
    X = df.drop(columns=['fraudulent', 'customer_id', 'transaction_id'], errors='ignore')
    y = df['fraudulent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILENAME)
    print(f"Dummy model saved to {MODEL_FILENAME}")

if __name__ == '__main__':
    print("Generating dummy model and synthetic transaction data...")

    # Generate training dataset and train model
    df_train = generate_dataset(5000)
    df_train_processed, _ = preprocess_data(df_train)
    train_and_save_model(df_train_processed)

    # Generate synthetic test data (transactions to be pushed to queue)
    df_test_transactions = generate_dataset(20) # Generate 20 dummy transactions
    # Keep only relevant columns for the queue (input to prediction)
    transactions_for_queue = df_test_transactions[['transaction_id', 'amount', 'timestamp', 'status', 'vendor_id']]
    transactions_for_queue.to_csv('synthetic_transactions.csv', index=False)
    print("Synthetic transactions saved to synthetic_transactions.csv")

    print("\nTo push these transactions to your queue service, you might use a script like:")
    print("import requests, json, pandas as pd")
    print("df = pd.read_csv('synthetic_transactions.csv')")
    print("headers = {'Authorization': 'your_agent_token_here', 'Content-Type': 'application/json'}")
    print("for index, row in df.iterrows():")
    print("    data = row.to_dict()")
    print("    data['timestamp'] = data['timestamp'].isoformat() # Convert timestamp to ISO format for JSON")
    print("    response = requests.post('http://localhost:7500/queues/transactions/push', headers=headers, json=data)")
    print("    print(f'Pushed {data[\"transaction_id\"]}: {response.status_code}')")