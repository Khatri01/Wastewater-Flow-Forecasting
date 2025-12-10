import os
import requests
from google.cloud import bigquery
from google.cloud import secretmanager
from datetime import datetime, timedelta, timezone

# Initialize Google Cloud clients
bigquery_client = bigquery.Client()
secret_client = secretmanager.SecretManagerServiceClient()

# --- Configuration ---
PROJECT_ID = os.environ.get('GCP_PROJECT')
DATASET_ID = "wastewater_data"
TABLE_ID = "sensor_readings"
TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# Reference to the secret in Secret Manager
SECRET_NAME = "smartcover-api-key"
SECRET_VERSION_REF = f"projects/{PROJECT_ID}/secrets/{SECRET_NAME}/versions/latest"

# Static configuration
LOCATION_NAME = "Waste 3"
CONVERSION_FACTOR_GPM_TO_CFS = 448.8
LOCATION_LIST_URL = "https://www.mysmartcover.com/api/locations/list.php"
HISTORICAL_DATA_URL = "https://www.mysmartcover.com/api/locations/data.php"

def fetch_and_store_data(_event, _context):
    """
    Triggered by Cloud Scheduler. Fetches data and stores it in BigQuery.
    """
    # --- Step 1: Securely fetch the API token ---
    print("Fetching API token from Secret Manager...")
    response = secret_client.access_secret_version(request={"name": SECRET_VERSION_REF})
    api_token = response.payload.data.decode("UTF-8")
    headers = {'Authorization': f'Bearer {api_token}'}
    
    # --- Step 2: Find the numeric ID for the location ---
    location_id = None
    try:
        print(f"Searching for location ID for '{LOCATION_NAME}'...")
        response = requests.get(LOCATION_LIST_URL, headers=headers, timeout=30)
        response.raise_for_status()
        locations = response.json().get('locations', [])
        for loc in locations:
            [cite_start]if loc.get('description') == LOCATION_NAME: # [cite: 162]
                [cite_start]location_id = loc.get('id') # [cite: 162]
                break
        if not location_id:
            print(f"Error: Could not find location '{LOCATION_NAME}'.")
            return "Location not found", 404
        print(f"Success! Found Location ID: {location_id}")

    except requests.exceptions.RequestException as e:
        print(f"Error calling Location List API: {e}")
        return "API error", 500

    # --- Step 3: Fetch historical flow data ---
    try:
        print(f"Fetching recent flow data for location ID {location_id}...")
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=90) # Use a 90-min window for safety
        
        params = {
            'location': location_id,
            [cite_start]'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'), # [cite: 53]
            [cite_start]'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'), # [cite: 53]
            [cite_start]'data_type': 6  # data_type 6 is "Flow" [cite: 388]
        }

        response = requests.get(HISTORICAL_DATA_URL, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        [cite_start]api_data_wrapper = response.json().get('data', []) # The API returns a list containing a single list of data points [cite: 228]
        historical_data = []
        if api_data_wrapper:
            historical_data = api_data_wrapper[0]

        if not historical_data:
            print("Success! API call worked, but no new data was found in the lookback window.")
            return "Success, no new data", 200
            
        # --- Step 4: Prepare and insert data into BigQuery ---
        rows_to_insert = []
        for entry in historical_data:
            if entry and len(entry) == 2:
                timestamp_str, flow_gpm = entry
                flow_cfs = float(flow_gpm) / CONVERSION_FACTOR_GPM_TO_CFS
                rows_to_insert.append({
                    "timestamp": timestamp_str,
                    "sensor_id": str(location_id),
                    "flow_data": flow_cfs
                })
        
        if not rows_to_insert:
            print("Data points were found but were invalid. Nothing to insert.")
            return "Success, no valid data", 200

        print(f"Inserting {len(rows_to_insert)} new rows into BigQuery...")
        errors = bigquery_client.insert_rows_json(TABLE_REF, rows_to_insert)

        if not errors:
            print("Successfully inserted new data.")
            return "Success", 200
        else:
            print(f"BigQuery insertion errors: {errors}")
            return "Error inserting data", 500

    except requests.exceptions.RequestException as e:
        print(f"Error calling Historical Data API: {e}")
        return "API error", 500
    