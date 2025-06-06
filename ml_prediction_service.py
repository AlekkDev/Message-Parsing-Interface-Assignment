import json
import joblib
import requests
from mpi4py import MPI
import time
import pandas as pd
import numpy as np
import os

# --- MPI Initialization ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() # Total number of MPI processes

# --- Configuration
try:
    import config
    TRANSACTIONS_QUEUE_URL = config.TRANSACTIONS_QUEUE_URL
    RESULTS_QUEUE_URL = config.RESULTS_QUEUE_URL
    AGENT_TOKEN = config.AGENT_TOKEN
    ADMIN_TOKEN = config.ADMIN_TOKEN # Needed for creating queues if they don't exist
    MODEL_PATH = config.MODEL_PATH
    STATUSES = config.STATUSES # From fraud_detection_predict.py
except ImportError:
    print("Error: config.py not found. Please create one with TRANSACTIONS_QUEUE_URL, RESULTS_QUEUE_URL, AGENT_TOKEN, ADMIN_TOKEN, MODEL_PATH, and STATUSES.")
    exit(1)
except AttributeError as e:
    print(f"Error in config.py: Missing attribute {e}. Ensure all required variables are defined.")
    exit(1)

# Helper functions (mostly for queues)
def pull_from_queue(queue_url, token):
    headers = {'Authorization': token}
    try:
        response = requests.get(queue_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 400 and "Queue is empty" in response.text:
            return None
        print(f"[{comm.rank}]: Error pulling from queue {queue_url}: {e} - {response.text}")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"[{comm.rank}]: Connection error to queue service at {queue_url}: {e}. Is app.py running?")
        return None
    except json.JSONDecodeError:
        print(f"[{comm.rank}]: JSONDecodeError when pulling from {queue_url}: {response.text}")
        return None

def push_to_queue(queue_url, data, token):
    headers = {'Authorization': token, 'Content-Type': 'application/json'}
    try:
        response = requests.post(queue_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[{comm.rank}]: Error pushing to queue {queue_url}: {e} - {response.text}")
        return None

# ML Model Loading and Prediction
def load_ml_model(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file {filename} not found.")
    return joblib.load(filename)

def preprocess_single_transaction(transaction_data, statuses_map):
    """
    Preprocesses a single transaction dictionary into a DataFrame row for model prediction.
    Assumes transaction_data has 'amount', 'timestamp', 'status', 'vendor_id'.
    """
    df = pd.DataFrame([transaction_data])
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9

    # Use the consistent status mapping
    df['status'] = df['status'].map(statuses_map)

    # Ensure expected columns exist or fill missing with 0 or smth else
    # Ensure the feature columns match what the model was trained on
    expected_features = ['timestamp', 'status', 'vendor_id', 'amount']
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0 # Or a more appropriate default/imputation

    # Drop any columns that are not features used by the model
    X = df.drop(columns=['transaction_id', 'customer_id', 'timestamp'], errors='ignore') # Drop ID/original timestamp

    # Re-order columns to match training order
    feature_columns = ['timestamp', 'status', 'vendor_id', 'amount']
    X = df[['timestamp', 'status', 'vendor_id', 'amount']].copy()

    return X


# Master Process Logic (rank == 0)
if rank == 0:
    print(f"Master process (Rank {rank}) starting...") [cite: 2]

    # Load the pre-trained ML model
    model = None
    try:
        model = load_ml_model(MODEL_PATH) [cite: 5]
        print(f"Master: Model '{MODEL_PATH}' loaded successfully.")
    except FileNotFoundError:
        print(f"Master: Error: Model file '{MODEL_PATH}' not found. Please run create_dummy_files.py first.")
        exit(1)
    except Exception as e:
        print(f"Master: Error loading model: {e}")
        exit(1)

    # Number of worker processes available
    num_workers = size - 1
    if num_workers <= 0:
        print("Master: No worker processes available. Exiting.")
        exit(0)

    print(f"Master: Detected {num_workers} worker processes.")

    # Create queues if they don't exist (using admin token)
    # This is important for initial setup
    try:
        requests.post(f"http://localhost:7500/queues/transactions", headers={'Authorization': ADMIN_TOKEN})
        requests.post(f"http://localhost:7500/queues/results", headers={'Authorization': ADMIN_TOKEN})
        print("Master: Attempted to create 'transactions' and 'results' queues.")
    except Exception as e:
        print(f"Master: Could not confirm queue creation (they might already exist): {e}")


    # Main loop for managing tasks
    while True:
        # Read requests from the transactions queue
        # Read as many requests as processors available
        current_batch_requests = []
        for _ in range(num_workers):
            request_data = pull_from_queue(TRANSACTIONS_QUEUE_URL, AGENT_TOKEN) [cite: 6]
            if request_data:
                current_batch_requests.append(request_data)
            else:
                break # Queue is empty or an error occurred

        if not current_batch_requests:
            print("Master: Transactions queue empty. Blocking until messages are available...") [cite: 7]
            time.sleep(5) # Wait before checking again if queue is empty
            continue

        print(f"Master: Pulled {len(current_batch_requests)} requests from transactions queue.")

        # Distribute tasks to workers
        active_workers_for_batch = len(current_batch_requests)
        pending_results = [] # To store pending prediction results and original transaction IDs

        for i in range(active_workers_for_batch):
            task_data = current_batch_requests[i]
            # Send task to worker (rank i+1). Using an arbitrary tag like 11 for tasks.
            comm.send(task_data, dest=i + 1, tag=11) [cite: 9]
            print(f"Master: Sent task {task_data.get('transaction_id', 'N/A')} to worker {i+1}")

        # Gather results from workers
        for _ in range(active_workers_for_batch):
            # Receive results from any source worker (tag 22 for results)
            prediction_result = comm.recv(source=MPI.ANY_SOURCE, tag=22) [cite: 10]
            pending_results.append(prediction_result)
            print(f"Master: Received result from worker {prediction_result.get('worker_rank')}: {prediction_result.get('transaction_id')}")

        # Send final predictions to the results queue
        for res in pending_results:
            push_to_queue(RESULTS_QUEUE_URL, res, AGENT_TOKEN) [cite: 10]
            print(f"Master: Pushed prediction for transaction {res.get('transaction_id', 'N/A')} to results queue.")

        print("Master: Batch processing complete. Taking next batch...") [cite: 11]
        time.sleep(1) # Small delay to avoid busy-waiting

# Worker Process Logic (rank != 0)
else:
    print(f"Worker process (Rank {rank}) starting...")

    # Load the ML model once per worker
    model = None
    statuses_map = {status: idx for idx, status in enumerate(STATUSES)} # For preprocessing
    try:
        model = load_ml_model(MODEL_PATH) [cite: 5]
        print(f"Worker {rank}: Model '{MODEL_PATH}' loaded successfully.")
    except FileNotFoundError:
        print(f"Worker {rank}: Error: Model file '{MODEL_PATH}' not found. Please run create_dummy_files.py first.")
        exit(1)
    except Exception as e:
        print(f"Worker {rank}: Error loading model: {e}")
        exit(1)

    while True:
        try:
            # Wait for a task from the master
            task_data = comm.recv(source=0, tag=11) [cite: 9]
            transaction_id = task_data.get('transaction_id', 'N/A')
            print(f"Worker {rank}: Received task for transaction {transaction_id}")

            # Preprocess the received transaction data
            # ensure the transaction_data contains 'amount', 'timestamp', 'status', 'vendor_id'
            try:
                processed_features_df = preprocess_single_transaction(task_data, statuses_map)

                # Perform prediction
                prediction = model.predict(processed_features_df)[0]
                prediction_result = {
                    "transaction_id": transaction_id,
                    "prediction": int(prediction), # Convert numpy.int64 to standard int for JSON
                    "worker_rank": rank,
                    "original_amount": task_data.get('amount')
                }
            except Exception as e:
                print(f"Worker {rank}: Error during prediction for {transaction_id}: {e}")
                prediction_result = {
                    "transaction_id": transaction_id,
                    "prediction": -1, # Indicate error
                    "worker_rank": rank,
                    "error": str(e)
                }

            # Send the result back to the master (tag 22 for results)
            comm.send(prediction_result, dest=0, tag=22) [cite: 10]
            print(f"Worker {rank}: Sent result for transaction {transaction_id}")

        except Exception as e:
            # Handle potential communication errors or shutdown signals
            print(f"Worker {rank}: Error or shutdown signal received: {e}. Exiting.")
            break # Exit worker loop if master disconnects or error occurs