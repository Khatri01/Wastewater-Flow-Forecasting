import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import joblib
from google.cloud import bigquery, storage
from datetime import datetime, timezone
import google.auth

# --- Configuration ---
BUCKET_NAME = 'wastewater-models-bucket' 

def get_project_id():
    pid = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if pid: return pid
    _, pid = google.auth.default()
    return pid

PROJECT_ID = get_project_id()
if not PROJECT_ID:
    raise RuntimeError("No GCP project ID detected.")

SENSOR_ID = '46799'
LOOKBACK = 144
HORIZON = 18
MODEL_FILE = "CNN_flow_model.pth" 
SCALER_FILE = "flow_data_scaler.joblib"
BASELINE_FILE = "daily_baseline.csv"

bq_client = bigquery.Client()
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# --- CNN Model Definition ---
class CNN_Forecaster(nn.Module):
    def __init__(self, input_dim, horizon=18, channels=64):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=channels, kernel_size=7, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(channels)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=7, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(channels)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 144, 128), 
            nn.ReLU(),
            nn.Linear(128, horizon)
        )

    def forward(self, x_hist, future_feats=None, y_teacher=None, tf_ratio=0.5):
        x = x_hist.permute(0, 2, 1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.head(x)

def load_artifacts():
    print("Downloading artifacts...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bucket.blob(SCALER_FILE).download_to_filename(SCALER_FILE)
    scaler = joblib.load(SCALER_FILE)

    bucket.blob(BASELINE_FILE).download_to_filename(BASELINE_FILE)
    baseline_medians = pd.read_csv(BASELINE_FILE, index_col=0, header=None).squeeze("columns")
    
    bucket.blob(MODEL_FILE).download_to_filename(MODEL_FILE)
    
    model = CNN_Forecaster(input_dim=7, horizon=HORIZON).to(device)
    
    # Robust model loading
    try:
        checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    model.eval()
    print("Artifacts loaded.")
    return scaler, baseline_medians, model, device

def get_inference_data():
    print("Fetching data from BigQuery...")
    query = f"""
        SELECT timestamp, flow_data
        FROM `{PROJECT_ID}.wastewater_data.sensor_readings_clean`
        WHERE sensor_id = '{SENSOR_ID}'
        ORDER BY timestamp DESC
        LIMIT {LOOKBACK * 2}
    """
    df = bq_client.query(query).to_dataframe()
    
    # Warn but don't fail yet if data is slightly short (interpolation might fix it)
    if len(df) < LOOKBACK * 2:
        print(f"Warning: Fetched {len(df)} rows. Ideal is {LOOKBACK*2}.")
        
    df = df.sort_values(by='timestamp', ascending=True)
    df['Datetime'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.set_index('Datetime').asfreq('10min')
    df['flow_data'] = df['flow_data'].interpolate(limit_direction='both')
    
    # FIX: Return the FULL dataframe, do not use .tail() here
    return df 

def feature_engineer(df, scaler, baseline_medians):
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
    print("Creating windows...")
    features_df = df.tail(LOOKBACK)[['resid_z', 'resid_diff_z', 'resid_prevday', 'sin_h', 'cos_h', 'sin_d', 'cos_d']]
    X = features_df.astype('float32').values
    
    last_time = df.index[-1]
    future_times = pd.date_range(start=last_time, periods=HORIZON + 1, freq='10min')[1:]
    
    future_tod = (future_times.hour * 60 + future_times.minute) // 10
    BL = baseline_medians.reindex(future_tod).values.astype('float32')
    
    X = torch.from_numpy(X).unsqueeze(0).to(device)
    
    return X, None, BL, future_times, last_time

def inverse_transform(y_pred_resid_z, BL, scaler):
    resid = y_pred_resid_z * scaler.scale_[0] + scaler.mean_[0]
    final_flow = resid + BL
    return final_flow

def save_predictions(predictions, future_times, sensor_id, last_time):
    print(f"Saving {len(predictions)} CNN predictions...")
    job_time_utc = datetime.now(timezone.utc)
    predictions_row = predictions[0] 
    last_data_point_time_str = pd.Timestamp(last_time, tz=timezone.utc).isoformat()

    rows_to_insert = []
    for i in range(len(future_times)):
        rows_to_insert.append({
            'prediction_time': job_time_utc.isoformat(), 
            'forecast_time': pd.Timestamp(future_times[i], tz=timezone.utc).isoformat(), 
            'sensor_id': sensor_id,
            'gru_prediction': None,
            'rf_prediction': None,
            'xgboost_prediction': None,
            'lstm_prediction': None,
            'cnn_prediction': float(predictions_row[i]), 
            'last_data_point_time': last_data_point_time_str
        })
    
    table_id = f"{PROJECT_ID}.wastewater_data.sensor_predictions"
    errors = bq_client.insert_rows_json(table_id, rows_to_insert)
    if not errors:
        print(f"Successfully saved {len(rows_to_insert)} predictions.")
    else:
        print(f"Encountered errors: {errors}")

def main():
    try:
        scaler, baseline_medians, model, device = load_artifacts()
        df_raw = get_inference_data()
        
        # Check for sufficient data
        min_raw_needed = LOOKBACK * 2 - 12 
        if df_raw is None or len(df_raw) < min_raw_needed:
            print(f"Skipping: Need at least {min_raw_needed} raw rows, got {len(df_raw) if df_raw is not None else 0}.")
            return

        df_features = feature_engineer(df_raw, scaler, baseline_medians)
        
        # After feature engineering (dropping lag rows), we need LOOKBACK rows
        if len(df_features) < LOOKBACK:
            print(f"Skipping: Not enough data after feature engineering. Got {len(df_features)}, need {LOOKBACK}.")
            return

        X, _, BL, future_times, last_time = create_windows(df_features, baseline_medians, device)

        with torch.no_grad():
            y_pred_z = model(X).cpu().numpy()
        
        final_prediction = inverse_transform(y_pred_z, BL, scaler)
        save_predictions(final_prediction, future_times, SENSOR_ID, last_time)
        print("CNN Prediction job complete.")
        
    except Exception as e:
        print(f"Job failed: {e}")
        raise e 

if __name__ == "__main__":
    main()