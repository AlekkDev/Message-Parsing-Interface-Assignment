import json
import joblib
import requests
from mpi4py import MPI
import time
import pandas as pd
import numpy as np
import os
from datetime import datetime # Import datetime for timestamp conversion check

#MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() # Total number of MPI processes
print(f"[{rank}]: MPI process initialized. Rank: {rank}, Total Size: {size}") 

#Configuration
try:
    import config
    TRANSACTIONS_QUEUE_URL = config.TRANSACTIONS_QUEUE_URL
    RESULTS_QUEUE_URL = config.RESULTS_QUEUE_URL
    AGENT_TOKEN = config.AGENT_TOKEN
    ADMIN_TOKEN = config.ADMIN_TOKEN # Needed for creating queues if they don't exist
    MODEL_PATH = config.MODEL_PATH
    STATUSES = config.STATUSES # From fraud_detection_predict.py
    # Get the default number of worker processors, default to 5 if not in config
    DEFAULT_WORKER_PROCESSORS = getattr(config, 'DEFAULT_WORKER_PROCESSORS', 5)
    print(f"[{rank}]: Configuration loaded successfully.") 
except ImportError:
    print(f"[{rank}]: Error: config.py not found. Please create one with TRANSACTIONS_QUEUE_URL, RESULTS_QUEUE_URL, AGENT_TOKEN, ADMIN_TOKEN, MODEL_PATH, STATUSES, and optionally DEFAULT_WORKER_PROCESSORS.")
    exit(1)
except AttributeError as e:
    print(f"[{rank}]: Error in config.py: Missing attribute {e}. Ensure all required variables are defined.")
    exit(1)

# Helper functions (mostly for queues)
def pull_from_queue(queue_url, token):
    print(f"[{comm.rank}]: Attempting to pull from queue: {queue_url}/pull")
    headers = {'Authorization': token}
    try:
        # Append /pull to the base URL
        response = requests.get(f"{queue_url}/pull", headers=headers)
        response.raise_for_status()
        print(f"[{comm.rank}]: Successfully pulled from queue {queue_url}/pull. Status: {response.status_code}")
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 400 and "Queue is empty" in response.text:
            print(f"[{comm.rank}]: Queue {queue_url}/pull is empty (Status 400).")
            return None
        print(f"[{comm.rank}]: HTTP Error pulling from queue {queue_url}/pull: {e} - {response.text}")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"[{comm.rank}]: Connection error to queue service at {queue_url}: {e}. Is app.py running and accessible?")
        return None
    except json.JSONDecodeError:
        print(f"[{comm.rank}]: JSONDecodeError when pulling from {queue_url}/pull: Expected JSON, got '{response.text}'")
        return None
    except Exception as e:
        print(f"[{comm.rank}]: An unexpected error occurred pulling from queue {queue_url}/pull: {e}")
        return None

def push_to_queue(queue_url, data, token):
    print(f"[{comm.rank}]: Attempting to push to queue: {queue_url}/push with data (transaction_id): {data.get('transaction_id', 'N/A')}")
    headers = {'Authorization': token, 'Content-Type': 'application/json'}
    try:
        response = requests.post(f"{queue_url}/push", headers=headers, json=data)
        response.raise_for_status()
        print(f"[{comm.rank}]: Successfully pushed to queue {queue_url}/push. Status: {response.status_code}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[{comm.rank}]: Error pushing to queue {queue_url}/push: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"[{comm.rank}]: Response text: {e.response.text}")
        return None
    except Exception as e:
        print(f"[{comm.rank}]: An unexpected error occurred pushing to queue {queue_url}/push: {e}")
        return None

# ML Model Loading and Prediction
def load_ml_model(filename):
    print(f"[{comm.rank}]: Loading ML model from {filename}...") 
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file {filename} not found.")
    model = joblib.load(filename)
    print(f"[{comm.rank}]: Model loaded successfully.") 
    return model

def preprocess_single_transaction(transaction_data, statuses_map):
    """
    Preprocesses a single transaction dictionary into a DataFrame row for model prediction.
    Assumes transaction_data has 'amount', 'timestamp', 'status', 'vendor_id'.
    """
    print(f"[{comm.rank}]: Preprocessing transaction ID: {transaction_data.get('transaction_id', 'N/A')}") 
    # Define the features the model expects and their order
    feature_columns_order = ['timestamp', 'status', 'vendor_id', 'amount']

    # Validate essential fields are present in the input transaction_data
    required_input_features = ['amount', 'timestamp', 'status', 'vendor_id']
    for field in required_input_features:
        if field not in transaction_data:
            raise ValueError(f"Missing required input feature for prediction: '{field}' in transaction ID {transaction_data.get('transaction_id', 'N/A')}")

    df = pd.DataFrame([transaction_data])

    # Convert timestamp. Handle potential string vs. datetime objects.
    # The transaction data pulled from the queue will have timestamp as a string.
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9
        print(f"[{comm.rank}]: Timestamp converted for {transaction_data.get('transaction_id', 'N/A')}") 
    except Exception as e:
        raise ValueError(f"Could not convert timestamp '{transaction_data.get('timestamp')}' for transaction {transaction_data.get('transaction_id', 'N/A')}: {e}")

    # Map status string to numerical encoding
    # Check if the status exists in the map
    if df['status'].iloc[0] not in statuses_map:
        raise ValueError(f"Unknown status '{df['status'].iloc[0]}' for transaction {transaction_data.get('transaction_id', 'N/A')}. Expected one of {list(statuses_map.keys())}")
    df['status'] = df['status'].map(statuses_map)
    print(f"[{comm.rank}]: Status mapped for {transaction_data.get('transaction_id', 'N/A')}") 

    # Ensure all expected feature columns exist and are in the correct order
    # It's safer to reconstruct X based on expected_features to avoid unexpected columns
    X = df[feature_columns_order].copy()
    print(f"[{comm.rank}]: Preprocessing complete for {transaction_data.get('transaction_id', 'N/A')}") 
    return X


# Master Process Logic (rank == 0)
if rank == 0:
    print(f"Master process (Rank {rank}) starting...")

    # Load the pre-trained ML model
    model = None
    try:
        model = load_ml_model(MODEL_PATH)
        print(f"Master: Model '{MODEL_PATH}' loaded successfully.")
    except FileNotFoundError:
        print(f"Master: Error: Model file '{MODEL_PATH}' not found. Please run create_dummy_files.py first.")
        # Broadcast a signal to workers to exit if master can't proceed
        for i in range(1, size):
            print(f"Master: Sending SHUTDOWN signal to worker {i} due to missing model.") 
            comm.send("SHUTDOWN", dest=i, tag=99) # Using a shutdown tag
        exit(1)
    except Exception as e:
        print(f"Master: Error loading model: {e}")
        for i in range(1, size):
            print(f"Master: Sending SHUTDOWN signal to worker {i} due to model load error.") 
            comm.send("SHUTDOWN", dest=i, tag=99)
        exit(1)

    # Number of worker processes available (total processes - 1 for master)
    # Use DEFAULT_WORKER_PROCESSORS from config, but ensure it doesn't exceed total MPI size - 1
    num_workers = min(DEFAULT_WORKER_PROCESSORS, size - 1)

    if num_workers <= 0:
        print("Master: No worker processes available (size - 1 <= 0). Exiting.")
        exit(0)

    print(f"Master: Configured for {DEFAULT_WORKER_PROCESSORS} worker processes. Actual available workers: {num_workers}. (Total MPI size: {size})")

    # Create queues if they don't exist (using admin token)
    queue_setup_headers = {'Authorization': ADMIN_TOKEN}

    for queue_name in ['transactions', 'results']:
        queue_full_url = f"http://{config.QUEUE_SERVICE_HOST}:{config.QUEUE_SERVICE_PORT}/queues/{queue_name}"
        print(f"Master: Attempting to create/confirm queue: {queue_name} at {queue_full_url}") 
        try:
            create_response = requests.post(queue_full_url, headers=queue_setup_headers)
            create_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            print(f"Master: Successfully created/confirmed '{queue_name}' queue. Response: {create_response.status_code} - {create_response.json()}")
        except requests.exceptions.HTTPError as e:
            if create_response.status_code == 400 and "Queue already exists" in create_response.text:
                print(f"Master: Queue '{queue_name}' already exists (status 400). Proceeding.")
            else:
                print(f"Master: HTTP Error creating/confirming '{queue_name}' queue: {e} - {create_response.text}")
                # You might want to exit here if queue creation is critical and failed unexpectedly
        except requests.exceptions.ConnectionError as e:
            print(f"Master: Connection error to queue service at {queue_full_url}. Is app.py running?")
            # This is critical, if queue service isn't running, master can't proceed
            for i in range(1, size):
                print(f"Master: Sending SHUTDOWN signal to worker {i} due to queue service connection error.") 
                comm.send("SHUTDOWN", dest=i, tag=99)
            exit(1)
        except Exception as e:
            print(f"Master: An unexpected error occurred during '{queue_name}' queue setup: {e}")
    print("Master: All necessary queues checked/created.") 

    # Main loop for managing tasks
    while True:
        print(f"\nMaster: Entering new batch processing cycle. Attempting to pull up to {num_workers} requests...") 
        current_batch_requests = []
        for _ in range(num_workers):
            request_data = pull_from_queue(TRANSACTIONS_QUEUE_URL, AGENT_TOKEN)
            if request_data:
                current_batch_requests.append(request_data)
                print(f"Master: Successfully pulled transaction {request_data.get('transaction_id', 'N/A')} from transactions queue.") 
            else:
                print(f"Master: No more requests found in queue or error on pull attempt. Pulled {len(current_batch_requests)} in this cycle.") 
                break # Queue is empty or an error occurred

        if not current_batch_requests:
            print("Master: Transactions queue empty. Blocking until messages are available (checking every 5 seconds)...")
            time.sleep(5) # Wait before checking again if queue is empty
            continue # Go to the top of the while loop to try pulling again

        print(f"Master: Pulled {len(current_batch_requests)} requests for this batch. Distributing tasks to workers...")

        # Distribute tasks to workers
        # Only send tasks to actual active workers based on how many requests were pulled
        for i in range(len(current_batch_requests)):
            task_data = current_batch_requests[i]
            # Send task to worker (rank i+1).
            # This assumes that if len(current_batch_requests) is less than num_workers,
            # only the necessary workers (1 to len(current_batch_requests)) receive tasks.
            # Other workers will remain in their comm.recv loop until a task or shutdown is sent.
            print(f"Master: Sending task {task_data.get('transaction_id', 'N/A')} to worker {i+1} (tag 11).") 
            comm.send(task_data, dest=i + 1, tag=11)
            print(f"Master: Sent task {task_data.get('transaction_id', 'N/A')} to worker {i+1} successfully.") 

        print(f"Master: All {len(current_batch_requests)} tasks sent. Now gathering results from workers...")

        # Gather results from workers
        pending_results = []
        for _ in range(len(current_batch_requests)):
            # Receive results from any source worker (tag 22 for results)
            print("Master: Waiting to receive a result from any worker (tag 22)...") 
            prediction_result = comm.recv(source=MPI.ANY_SOURCE, tag=22)
            pending_results.append(prediction_result)
            print(f"Master: Received result from worker {prediction_result.get('worker_rank')} for transaction {prediction_result.get('transaction_id', 'N/A')}.") 

        print("Master: All results gathered for this batch. Now pushing to results queue.")

        # Send final predictions to the results queue
        for res in pending_results:
            push_to_queue(RESULTS_QUEUE_URL, res, AGENT_TOKEN)
            print(f"Master: Pushed prediction for transaction {res.get('transaction_id', 'N/A')} to results queue.") 

        print("Master: Batch processing complete. Taking next batch in 1 second...")
        time.sleep(1) # Small delay to avoid busy-waiting


# Worker Process Logic (rank != 0)
else:
    print(f"Worker process (Rank {rank}) starting...")

    # Load the ML model once per worker
    model = None
    statuses_map = {status: idx for idx, status in enumerate(STATUSES)} # For preprocessing
    try:
        model = load_ml_model(MODEL_PATH)
        print(f"Worker {rank}: Model '{MODEL_PATH}' loaded successfully.")
    except FileNotFoundError:
        print(f"Worker {rank}: Error: Model file '{MODEL_PATH}' not found. Please ensure it's accessible. Worker exiting.")
        exit(1) # Worker cannot proceed without model
    except Exception as e:
        print(f"Worker {rank}: Error loading model: {e}. Worker exiting.")
        exit(1)

    while True:
        print(f"Worker {rank}: Waiting for task from master (tag 11 or SHUTDOWN tag 99)...") 
        try:
            # Wait for a task from the master
            # Also listen for a shutdown signal (tag 99)
            status = MPI.Status()
            task_data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            print(f"Worker {rank}: Received message from master with tag {status.Get_tag()}.") 

            if status.Get_tag() == 99 and task_data == "SHUTDOWN":
                print(f"Worker {rank}: Received shutdown signal. Exiting.")
                break # Exit the worker loop

            if not isinstance(task_data, dict): # Basic check if received data is actually a dictionary
                print(f"Worker {rank}: Received invalid task data (not a dict): {task_data}. Skipping.")
                continue

            transaction_id = task_data.get('transaction_id', 'N/A_NO_ID')
            print(f"Worker {rank}: Received task for transaction {transaction_id}.")

            prediction_result = {
                "transaction_id": transaction_id,
                "prediction": -1, # Default to -1 to indicate error/not processed
                "worker_rank": rank,
                "amount": task_data.get('amount'), # Ensure these are included for results queue
                "timestamp": task_data.get('timestamp') # Ensure these are included for results queue
            }

            try:
                # Preprocess the received transaction data
                print(f"Worker {rank}: Preprocessing transaction {transaction_id}...")
                processed_features_df = preprocess_single_transaction(task_data, statuses_map)

                # Perform prediction
                print(f"Worker {rank}: Performing prediction for transaction {transaction_id}...")
                prediction = model.predict(processed_features_df)[0]
                prediction_result["prediction"] = int(prediction) # Convert numpy.int64 to standard int for JSON
                print(f"Worker {rank}: Prediction for {transaction_id} is: {prediction_result['prediction']}") 

            except ValueError as e:
                print(f"Worker {rank}: !!! INPUT DATA ERROR for {transaction_id}: {e}")
                prediction_result["error"] = str(e)
            except Exception as e:
                print(f"Worker {rank}: !!! UNEXPECTED ERROR during prediction for {transaction_id}: {e}")
                prediction_result["error"] = str(e)

            # Send the result back to the master (tag 22 for results)
            print(f"Worker {rank}: Sending result for transaction {transaction_id} back to master (tag 22).")
            comm.send(prediction_result, dest=0, tag=22)
            print(f"Worker {rank}: Sent result for transaction {transaction_id} successfully.") 

        except Exception as e:
            # General catch-all for unexpected MPI communication errors, though specific checks are better
            print(f"Worker {rank}: General error or unexpected communication issue: {e}. Worker exiting.")
            break # Exit the worker loop