import requests
import json
import pandas as pd
from datetime import datetime
import os

# Load config to get tokens and queue URLs
try:
    import config
    AGENT_TOKEN = config.AGENT_TOKEN
    # Corrected: Ensure this is the base URL for transactions, without /pull or /push
    TRANSACTIONS_QUEUE_URL = config.TRANSACTIONS_QUEUE_URL
    print(f"Using TRANSACTIONS_QUEUE_URL: {TRANSACTIONS_QUEUE_URL}")
except ImportError:
    print("Error: config.py not found. Please create one.")
    exit(1)
except AttributeError as e:
    print(f"Error in config.py: Missing attribute {e}. Ensure AGENT_TOKEN and TRANSACTIONS_QUEUE_URL are defined.")
    exit(1)

# MODEL_FILENAME = 'fraud_rf_model.pkl'
# STATUSES = ['submitted', 'accepted', 'rejected']

headers = {
    'Authorization': AGENT_TOKEN,
    'Content-Type': 'application/json'
}

synthetic_data_file = 'synthetic_transactions.csv'

if not os.path.exists(synthetic_data_file):
    print(f"Error: '{synthetic_data_file}' not found. Please run create_dummy_files.py first.")
    exit(1)

try:
    df = pd.read_csv(synthetic_data_file)
except Exception as e:
    print(f"Error reading {synthetic_data_file}: {e}")
    exit(1)

print(f"Attempting to push {len(df)} transactions to the queue...")
for index, row in df.iterrows():
    data = row.to_dict()
    if isinstance(data['timestamp'], (pd.Timestamp, datetime)):
        data['timestamp'] = data['timestamp'].isoformat()
    
    #Append /push to the base URL
    response = requests.post(f"{TRANSACTIONS_QUEUE_URL}/push", headers=headers, json=data)
    print(f"Pushed {data.get('transaction_id', 'N/A')}: Status {response.status_code} - {response.text}")
    if response.status_code >= 400:
        print(f"Failed to push transaction {data.get('transaction_id', 'N/A')}. Check errors.")

print("Finished attempting to push transactions.")