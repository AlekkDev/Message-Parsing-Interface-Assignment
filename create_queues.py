import requests
import json
import config # Assuming config.py is in the same directory

# Load tokens and base URL from config
ADMIN_TOKEN = config.ADMIN_TOKEN
QUEUE_SERVICE_HOST = config.QUEUE_SERVICE_HOST
QUEUE_SERVICE_PORT = config.QUEUE_SERVICE_PORT

BASE_QUEUE_URL = f"http://{QUEUE_SERVICE_HOST}:{QUEUE_SERVICE_PORT}/queues"
HEADERS = {'Authorization': ADMIN_TOKEN, 'Content-Type': 'application/json'}

queues_to_create = ['transactions', 'results']

print(f"Attempting to create queues via {BASE_QUEUE_URL}...")

for queue_name in queues_to_create:
    url = f"{BASE_QUEUE_URL}/{queue_name}"
    print(f"\nSending POST request to {url} to create queue '{queue_name}'...")
    try:
        response = requests.post(url, headers=HEADERS)
        if response.status_code == 201:
            print(f"SUCCESS: Queue '{queue_name}' created. Response: {response.json()}")
        elif response.status_code == 400 and "Queue already exists" in response.text:
            print(f"INFO: Queue '{queue_name}' already exists. Skipping creation. Response: {response.json()}")
        else:
            print(f"ERROR: Failed to create queue '{queue_name}'. Status: {response.status_code}, Response: {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to the queue service at {BASE_QUEUE_URL}. Is app.py running?")
        break # Exit if connection fails for one, likely fails for others
    except Exception as e:
        print(f"AN UNEXPECTED ERROR occurred: {e}")

print("\nQueue creation attempt complete.")