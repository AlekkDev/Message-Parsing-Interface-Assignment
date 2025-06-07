import requests
import json
import time # Added for sleep

# Load your config to get tokens and queue URLs
try:
    import config
    AGENT_TOKEN = config.AGENT_TOKEN
    RESULTS_QUEUE_URL = config.RESULTS_QUEUE_URL
    print(f"Using RESULTS_QUEUE_URL: {RESULTS_QUEUE_URL}")
except ImportError:
    print("Error: config.py not found. Please create one.")
    exit(1)
except AttributeError as e:
    print(f"Error in config.py: Missing attribute {e}. Ensure AGENT_TOKEN and RESULTS_QUEUE_URL are defined.")
    exit(1)

headers = {'Authorization': AGENT_TOKEN}

print("Attempting to pull results from the queue...")
pulled_count = 0
max_pulls = 50 # Limit for demonstration, prevent infinite loop if queue never empties

while pulled_count < max_pulls:
    response = requests.get(f"{RESULTS_QUEUE_URL}/pull", headers=headers)
    if response.status_code == 200:
        result = response.json()
        print(f"Pulled result: {json.dumps(result, indent=2)}")
        pulled_count += 1
    elif response.status_code == 400 and "Queue is empty" in response.text:
        print("Results queue is empty. Stopping pull attempts.")
        break
    else:
        print(f"Error pulling from results queue: Status {response.status_code} - {response.text}")
        break
    # Small delay to avoid hammering the queue, especially if it's empty
    time.sleep(0.1)

print(f"Finished pulling results. Total pulled: {pulled_count}")