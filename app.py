from flask import Flask, request, jsonify
import logging
# from datetime import datetime
import os
import json
from threading import RLock
# import pytz
# from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# Create FLASK app
app = Flask(__name__)
app.config.from_pyfile('config.py') # load config from config.py

queues = {} # Dictionary to hold queues
queue_lock = RLock() # Lock for queues, Prevents two users from modifying the same queue simultaneously
PERSISTENCE_FILE = "queue_data.json" # File to save data
REQUIRED_FIELDS = ['transaction_id', 'amount', 'timestamp'] #
REQUIRED_CONFIGS = ['MAX_QUEUE_LENGTH', 'PERSISTENCE_INTERVAL', 'ADMIN_TOKEN', 'AGENT_TOKEN']


# LOGGING
# Creates log file that records: WHEN message was sent, the QUEUE NAME, and the MESSAGE
logging.basicConfig(
    filename='message_queue.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@app.before_request
def log_request():
    # Log incoming requests
    logging.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
    logging.info(f"Headers: {dict(request.headers)}")
    if request.data:
        logging.info(f"Body: {request.data.decode('utf-8')}")

@app.after_request
def log_response(response):
    # Log outgoing responses
    logging.info(f"Response: {response.status} - {response.data.decode('utf-8')}")
    return response

# DATA SAVING AND LOADING

def save_state():
    # Saves queues to a file
    with queue_lock:
        with open(PERSISTENCE_FILE, 'w') as f:
            json.dump(queues, f)

def load_state():
    # Loads queues from a file
    global queues
    if os.path.exists(PERSISTENCE_FILE):
        with open(PERSISTENCE_FILE, 'r') as f:
            queues = json.load(f)

# AUTHENTICATION CHECK
def check_auth(required_role):
    #Check if request has proper authorization
    token = request.headers.get('Authorization')

    if required_role == 'admin': # Check if the user is an admin
        return token == app.config['ADMIN_TOKEN'] #
    elif required_role == 'agent_or_admin': # Check if the user is an agent or admin
        return token in [app.config['ADMIN_TOKEN'], app.config['AGENT_TOKEN']]
    return False

# Load initial state
load_state()

# QUEUE ENDPOINTS
@app.route('/queues', methods=['GET'])
def list_queues():
    # List all queues
    if not check_auth("agent_or_admin"):
        return jsonify({"error": "Unauthorized: Agent or Admin access required"}), 401
    return jsonify(list(queues.keys()))
@app.route('/queues/<queue_name>', methods=['POST'])
def create_queue(queue_name):
    # Create a new queue
    if not check_auth('admin'):
        return jsonify({"error": "Unauthorized: Admin access required"}), 403
    # print("Access granted")
    with queue_lock:
        # print("Creating queue:", queue_name)
        if queue_name in queues:
            return jsonify({"error": "Queue already exists"}), 400
        queues[queue_name] = []
        save_state()
    return jsonify({"message": f"Queue {queue_name} created"}), 201

@app.route('/queues/<queue_name>', methods=['DELETE'])
def delete_queue(queue_name):
    if not check_auth('admin'):
        return jsonify({"error": "Unauthorized: Admin access required"}), 403
    with queue_lock:
        if queue_name not in queues:
            return jsonify({"error": "Queue not found"}), 404
        del queues[queue_name]
        save_state()
    return jsonify({"message": f"Queue {queue_name} deleted"}), 200

# MESSAGE ENDPOINTS
@app.route('/queues/<queue_name>/push', methods=['POST'])
def push_message(queue_name):
    if not check_auth('agent_or_admin'):
        return jsonify({"error": "Unauthorized: Agent or Admin access required"}), 401
    data = request.get_json()

    if not all(field in data for field in REQUIRED_FIELDS):
        return jsonify({"error": f"Missing required fields: {REQUIRED_FIELDS}"}), 400

    if not data:
        return jsonify({"error": "No data provided"}), 400

    with queue_lock:
        if queue_name not in queues:
            return jsonify({"error": "Queue not found"}), 404

        if len(queues[queue_name]) >= app.config['MAX_QUEUE_LENGTH']:
            return jsonify({"error": "Queue is full"}), 400

        queues[queue_name].append(data)
        save_state()
        return jsonify({"message":"Message added to queue"}), 201
@app.route('/queues/<queue_name>/pull', methods=['GET'])
def pull_message(queue_name):
    # Check if the user is an agent or admin
    if not check_auth('agent_or_admin'):
        return jsonify({"error": "Unauthorized: Agent or Admin access required"}), 401

    with queue_lock:
        # Check if the queue exists
        if queue_name not in queues:
            return jsonify({"error": "Queue not found"}), 404
        # Check if the queue is empty
        if not queues[queue_name]:
            return jsonify({"error": "Queue is empty"}), 400
        # Pop the first message from the queue
        message = queues[queue_name].pop(0)

        save_state() # Save the state after popping the message
        return jsonify(message), 200
    
# SCHEDULER FOR AUTO-SAVING
# scheduler = BackgroundScheduler(timezone=pytz.utc)  # needed to fix error by specifying timezone
# scheduler.add_job(save_state, 'interval', seconds=app.config['PERSISTENCE_INTERVAL'])
# scheduler.start()

# SHUTDOWN - used to ensure proper cleanup
@atexit.register
def shutdown():
    # scheduler.shutdown()
    save_state()


if __name__ == '__main__':
    app.run(port=7500)