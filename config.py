# config.py
MAX_QUEUE_LENGTH = 100
PERSISTENCE_INTERVAL = 300
ADMIN_TOKEN = "ADMIN_TOKEN"
AGENT_TOKEN = "AGENT_OR_ADMIN_TOKEN"

# URLs for Flask app.py (from Assignment 3 queue service)
QUEUE_SERVICE_HOST = "localhost"
QUEUE_SERVICE_PORT = 7500
TRANSACTIONS_QUEUE_URL = f"http://{QUEUE_SERVICE_HOST}:{QUEUE_SERVICE_PORT}/queues/transactions/pull"
RESULTS_QUEUE_URL = f"http://{QUEUE_SERVICE_HOST}:{QUEUE_SERVICE_PORT}/queues/results/push"

# pre-trained ML model
MODEL_PATH = "fraud_rf_model.pkl"

# Statuses used for encoding
STATUSES = ['submitted', 'accepted', 'rejected']