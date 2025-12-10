import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb # Import XGBoost
from google.cloud import bigquery, storage
from datetime import datetime, timezone
import google.auth

# --- Configuration ---
BUCKET_NAME = 'wastewater-models-bucket' 

def get_project_id():
    """Gets the Project ID from the environment or auth credentials."""
    pid = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if pid:
        return pid
    # Fallback to Application Default Credentials
    _, pid = google.auth.default()
    return pid

PROJECT_ID = get_project_id()
if not PROJECT_ID:
    raise RuntimeError("No GCP project ID detected.")

SENSOR_ID = '46799'
N_LAGS = 144  # 24h lookback (144 * 10 min)
N_OUTPUTS = 18 # 3h horizon (18 * 10 min)
MODEL_FILE = "xgboost_models.joblib" # The list of 18 models

# BigQuery & GCS Clients
bq_client = bigquery.Client()
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def load_artifacts():
    """Downloads the list of XGBoost models from GCS."""
    print("Downloading XGBoost models...")
    bucket.blob(MODEL_FILE).download_to_filename(MODEL_FILE)
    models = joblib.load(MODEL_FILE)
    print(f"Loaded {len(models)} XGBoost models.")
    return models

def get_inference_data():
    """Gets the last 144 data points for the lookback window."""
    print("Fetching data from BigQuery...")
    # Fetches the 288 most recent data points to be robust to gaps
    query = f"""
        SELECT timestamp, flow_data
        FROM `{PROJECT_ID}.wastewater_data.sensor_readings_clean`
        WHERE sensor_id = '{SENSOR_ID}'
        ORDER BY timestamp DESC
        LIMIT {N_LAGS * 2}
    """
    df = bq_client.query(query).to_dataframe()
    
    if len(df) < N_LAGS: 
        print(f"Error: Not enough data for lookback. Need {N_LAGS}, got {len(df)}.")
        return None
        
    df = df.sort_values(by='timestamp', ascending=True)
    df['Datetime'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.set_index('Datetime').asfreq('10min') 
    df['flow_data'] = df['flow_data'].interpolate(limit_direction='both') 
    
    return df.tail(N_LAGS) # Return exactly 144 rows

def feature_engineer(df):
    """Replicates the feature engineering from XGBoost notebook."""
    print("Engineering features...")
    flow_lags = df['flow_data'].values # Shape (144,)
    
    last_time = df.index[-1]
    forecast_start_time = last_time + pd.Timedelta(minutes=10)
    
    h = forecast_start_time.hour
    dow = forecast_start_time.dayofweek
    
    time_features = [
        np.sin(2 * np.pi * h / 24),
        np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7)
    ]
    
    X = np.hstack([flow_lags, time_features]).astype('float32')
    X = X.reshape(1, -1) # Shape (1, 148)
    
    future_times = pd.date_range(start=forecast_start_time, periods=N_OUTPUTS, freq='10min')
    
    return X, future_times, last_time

# --- vvv THIS FUNCTION IS NOW FIXED AND SIMPLIFIED vvv ---
def save_predictions(predictions, future_times, sensor_id, last_time):
    """Saves the XGBoost forecast to BigQuery by inserting new rows."""
    print(f"Saving {len(predictions)} XGB predictions to BigQuery...")
    job_time_utc = datetime.now(timezone.utc)
    
    last_data_point_time_str = pd.Timestamp(last_time, tz=timezone.utc).isoformat()

    rows_to_insert = []
    for i in range(len(future_times)):
        rows_to_insert.append({
            'prediction_time': job_time_utc.isoformat(), 
            'forecast_time': pd.Timestamp(future_times[i], tz=timezone.utc).isoformat(), 
            'sensor_id': sensor_id,
            'gru_prediction': None, # This model is XGB, so this is NULL
            'rf_prediction': None,  # This is also NULL
            'xgboost_prediction': float(predictions[i]), # Save to the XGBoost column
            'last_data_point_time': last_data_point_time_str
        })
    
    table_id = f"{PROJECT_ID}.wastewater_data.sensor_predictions"
    errors = bq_client.insert_rows_json(table_id, rows_to_insert) # Simple insertion
    
    if not errors:
        print(f"Successfully saved {len(rows_to_insert)} predictions.")
    else:
        print(f"Encountered errors while inserting rows: {errors}")
# --- ^^^ THIS FUNCTION IS NOW FIXED AND SIMPLIFIED ^^^ ---


# --- Main execution ---
def main():
    try:
        models = load_artifacts() # This is now a list of 18 models
        df_raw = get_inference_data()
        
        if df_raw is None:
            print("Skipping prediction, not enough data.")
            return

        X, future_times, last_time = feature_engineer(df_raw)

        # Get the prediction: one from each model in the list
        prediction = [model.predict(X)[0] for model in models]
        
        save_predictions(prediction, future_times, SENSOR_ID, last_time)
        print("XGBoost Prediction job complete.")
        
    except Exception as e:
        print(f"Job failed: {e}")
        raise e

if __name__ == "__main__":
    main()