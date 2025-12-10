import os
import pandas as pd
import numpy as np
import torch
import joblib
from google.cloud import bigquery, storage
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import google.auth
from gru_app_model import GRU_AttnSeq2Seq

# --- Configuration ---
BUCKET_NAME = 'wastewater-models-bucket' 

def get_project_id():
    pid = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if pid:
        return pid
    _, pid = google.auth.default()
    return pid

PROJECT_ID = get_project_id()
if not PROJECT_ID:
    raise RuntimeError("No GCP project ID detected.")

SENSOR_ID = '46799'
LOOKBACK = 144
HORIZON = 18

SCALER_FILE = "flow_data_scaler.joblib"
BASELINE_FILE = "daily_baseline.csv"
MODEL_FILE = "gru_flow_model.pth"

bq_client = bigquery.Client()
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def load_artifacts():
    print("Downloading artifacts...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bucket.blob(SCALER_FILE).download_to_filename(SCALER_FILE)
    scaler = joblib.load(SCALER_FILE)

    bucket.blob(BASELINE_FILE).download_to_filename(BASELINE_FILE)
    baseline_medians = pd.read_csv(BASELINE_FILE, index_col=0, header=None).squeeze("columns")
    
    bucket.blob(MODEL_FILE).download_to_filename(MODEL_FILE)
    input_dim = 7
    model = GRU_AttnSeq2Seq(input_dim=input_dim, horizon=HORIZON).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    
    print("Artifacts loaded.")
    return scaler, baseline_medians, model, device

def get_inference_data():
    """Gets the recent clean data to build the 24-hour lookback window."""
    print("Fetching data from BigQuery...")
    # Fetches the 288 most recent data points, making it resilient to gaps.
    query = f"""
        SELECT timestamp, flow_data
        FROM `{PROJECT_ID}.wastewater_data.sensor_readings_clean`
        WHERE sensor_id = '{SENSOR_ID}'
        ORDER BY timestamp DESC
        LIMIT 288
    """
    df = bq_client.query(query).to_dataframe()
    if df.empty or len(df) < 288:
        print(f"Not enough data in the *entire table* for a lookback. Need 288, got {len(df)}.")
        return None
        
    # The data was fetched in descending order, so we must reverse it
    df = df.sort_values(by='timestamp', ascending=True)
        
    df['Datetime'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    # Re-grid to 10min (this creates NaNs if there were gaps) and interpolate
    df = df.set_index('Datetime').asfreq('10min') 
    df['flow_data'] = df['flow_data'].interpolate(limit_direction='both') # Fill small gaps
    return df

def feature_engineer(df, scaler, baseline_medians):
    """Replicates the feature engineering from the training notebook."""
    print("Engineering features...")
    tod = (df.index.hour * 60 + df.index.minute) // 10
    df['Baseline'] = baseline_medians.reindex(tod).values
    
    df['resid'] = df['flow_data'] - df['Baseline']
    df['resid_diff'] = df['resid'].diff().fillna(0.0)
    
    h = df.index.hour + df.index.minute / 60.0
    df['sin_h'] = np.sin(2*np.pi*h/24.0).astype('float32')
    df['cos_h'] = np.cos(2*np.pi*h/24.0).astype('float32')
    dow = df.index.dayofweek
    df['sin_d'] = np.sin(2*np.pi*dow/7.0).astype('float32')
    df['cos_d'] = np.cos(2*np.pi*dow/7.0).astype('float32')
    
    df['resid_prevday'] = df['resid'].shift(24 * 6)
    df = df.dropna(subset=['resid_prevday'])
    
    df[['resid_z', 'resid_diff_z']] = scaler.transform(df[['resid', 'resid_diff']].values)
    return df

def create_windows(df, baseline_medians, device):
    """Creates the exact input windows the model expects."""
    print("Creating windows...")
    features_df = df.tail(LOOKBACK)[['resid_z', 'resid_diff_z', 'resid_prevday', 'sin_h', 'cos_h', 'sin_d', 'cos_d']]
    X = features_df.astype('float32').values
    
    last_time = df.index[-1] 
    future_times = pd.date_range(start=last_time, periods=HORIZON + 1, freq='10min')[1:]
    
    future_df = pd.DataFrame(index=future_times)
    h = future_df.index.hour + future_df.index.minute / 60.0
    future_df['sin_h'] = np.sin(2*np.pi*h/24.0).astype('float32')
    future_df['cos_h'] = np.cos(2*np.pi*h/24.0).astype('float32')
    dow = future_df.index.dayofweek
    future_df['sin_d'] = np.sin(2*np.pi*dow/7.0).astype('float32')
    future_df['cos_d'] = np.cos(2*np.pi*dow/7.0).astype('float32')
    F = future_df.astype('float32').values

    future_tod = (future_times.hour * 60 + future_times.minute) // 10
    BL = baseline_medians.reindex(future_tod).values.astype('float32')
    
    X = torch.from_numpy(X).unsqueeze(0).to(device)
    F = torch.from_numpy(F).unsqueeze(0).to(device)
    
    return X, F, BL, future_times, last_time

# --- vvv THIS IS THE MISSING FUNCTION vvv ---
def inverse_transform(y_pred_resid_z, BL, scaler):
    """Converts the model's residual output back to the final flow (cfs)."""
    # y_pred_resid_z is shape (1, 18, 1)
    # Squeeze the last dim to match notebook logic -> (1, 18)
    y_squeezed = y_pred_resid_z.squeeze(-1) 
    
    resid = y_squeezed * scaler.scale_[0] + scaler.mean_[0]
    final_flow = resid + BL # (1, 18) + (18,) -> broadcasts to (1, 18)
    return final_flow
# --- ^^^ END OF MISSING FUNCTION ^^^ ---

def save_predictions(predictions, future_times, sensor_id, last_time):
    """Saves the final forecast to BigQuery."""
    print(f"Saving {len(future_times)} predictions to BigQuery...")
    job_time_utc = datetime.now(timezone.utc)
    predictions_row = predictions[0] 

    last_data_point_time_str = pd.Timestamp(last_time, tz=timezone.utc).isoformat()

    rows_to_insert = []
    for i in range(len(future_times)):
        rows_to_insert.append({
            'prediction_time': job_time_utc.isoformat(), 
            'forecast_time': pd.Timestamp(future_times[i], tz=timezone.utc).isoformat(), 
            'sensor_id': sensor_id,
            'gru_prediction': float(predictions_row[i]), 
            'lstm_prediction': None,
            'cnn_prediction': None,
            'rf_prediction': None,
            'xgboost_prediction': None,
            'ensemble_prediction': None,
            'last_data_point_time': last_data_point_time_str
        })
    
    table_id = f"{PROJECT_ID}.wastewater_data.sensor_predictions"
    errors = bq_client.insert_rows_json(table_id, rows_to_insert)
    if not errors:
        print(f"Successfully saved {len(rows_to_insert)} predictions.")
    else:
        print(f"Encountered errors while inserting rows: {errors}")

# --- Main execution ---
def main():
    try:
        scaler, baseline_medians, model, device = load_artifacts()
        df_raw = get_inference_data()
        
        min_rows_needed = LOOKBACK + (24 * 6)
        if df_raw is None or len(df_raw) < min_rows_needed:
            print(f"Not enough data to build features/lookback. Need {min_rows_needed} rows, got {len(df_raw)}. Skipping prediction.")
            return

        df_features = feature_engineer(df_raw, scaler, baseline_medians)
        
        if len(df_features) < LOOKBACK:
            print(f"Not enough data after feature engineering. Need {LOOKBACK} rows, got {len(df_features)}. Skipping prediction.")
            return

        X, F, BL, future_times, last_time = create_windows(df_features, baseline_medians, device)

        with torch.no_grad():
            y_pred_z = model(X, F, None, 0.0).cpu().numpy()
        
        # This call will now work because the function is defined
        final_prediction = inverse_transform(y_pred_z, BL, scaler) 
        
        save_predictions(final_prediction, future_times, SENSOR_ID, last_time)
        print("Prediction job complete.")
        
    except Exception as e:
        print(f"Job failed: {e}")
        raise e 

if __name__ == "__main__":
    main()