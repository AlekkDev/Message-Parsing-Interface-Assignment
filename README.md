# MPI ML Prediction Service

## Overview

We developed a high-performance, distributed ML prediction service. It processes transaction fraud detection requests using MPI, integrating with a queue service for asynchronous communication.

## Project Structure

* **`app.py`**: Flask-based queue service (Assignment 3). Handles `/push`, `/pull`, and queue creation.
* **`config.py`**: System configuration: tokens, queue URLs, model path, and status mappings. **Update tokens.**
* **`ml_prediction_service.py`**: Core MPI application.
    * **Master (Rank 0)**: Pulls transaction batches from queue, distributes tasks to workers via MPI, gathers results, and pushes predictions to results queue.
    * **Workers (Rank > 0)**: Load ML model, receive tasks, preprocess data, perform prediction, and return results to master via MPI.
* **`create_dummy_files.py`**: Generates `fraud_rf_model.pkl` (dummy model) and `synthetic_transactions.csv` (test data).
* **`create_queues.py`**: Helper to programmatically create queues in `app.py`.
* **`push.py`**: Client to push `synthetic_transactions.csv` to the `transactions` queue.
* **`pull.py`**: Client to pull results from the `results` queue.

## Getting Started


### Installation

#### Dependencies
```bash
pip install mpi4py scikit-learn pandas requests joblib Flask
```
## Running the Service
1. Generate Test data and model:
   ```bash
   python create_dummy_files.py
   ```
2. Start the Flask queue service (Terminal 1):
   ```bash
   python app.py
   ```
3. Initialize Queues (Terminal 2):
   ```bash
   python create_queues.py
   ```
4. Push transactions to the queue (Terminal 2):
   ```bash
    python push.py
    ```
5. Start the MPI service (Terminal 3):
    ```bash
    mpiexec -n 5 python ml_prediction_service.py
    ```
6. Pull results from the queue (Terminal 4):
    ```bash
    python pull.py
    ```

