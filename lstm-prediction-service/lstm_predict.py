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
LOOKBACK = 144  # 24h lookback
HORIZON = 18    # 3h horizon
MODEL_FILE = "lstm_flow_model.pth" 
SCALER_FILE = "flow_data_scaler.joblib"
BASELINE_FILE = "daily_baseline.csv"

bq_client = bigquery.Client()
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# --- LSTM Model Definition ---
class AttnSeq2Seq(nn.Module):
    def __init__(self, input_dim, enc_h=384, dec_h=384, horizon=18):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.LSTM(input_dim, enc_h, num_layers=2, batch_first=True, dropout=0.1)
        self.Wa = nn.Linear(enc_h, dec_h, bias=False)
        self.dec_in = nn.Linear(1 + 4 + enc_h, dec_h)
        self.decoder = nn.LSTM(dec_h, dec_h, num_layers=1, batch_first=True)
        self.out = nn.Linear(dec_h, 1)

    def attend(self, dec_h_t, enc_out):
        proj = self.Wa(enc_out)
        scores = torch.einsum('bld,bd->bl', proj, dec_h_t)
        alpha = torch.softmax(scores, dim=1)
        ctx = torch.einsum('bl,bld->bd', alpha, enc_out)
        return ctx

    def forward(self, x, future_feats, y_teacher=None, tf_ratio=0.5):
        B = x.size(0)
        enc_out, (h, c) = self.encoder(x)
        dec_h = h[-1]
        dec_c = torch.zeros_like(dec_h).unsqueeze(0)
        dec_state = (dec_h.unsqueeze(0), dec_c)

        y_prev = torch.zeros(B, 1, device=x.device)
        outs = []

        for t in range(self.horizon):
            ctx = self.attend(dec_state[0].squeeze(0), enc_out)
            
            feat_t = future_feats[:, t]
            if feat_t.dim() == 3: feat_t = feat_t.squeeze(1)
            feat_t = feat_t.reshape(B, -1)
            
            y_prev = y_prev.reshape(B, 1)
            ctx    = ctx.reshape(B, -1)

            din = torch.cat([y_prev, feat_t, ctx], dim=1)
            din = torch.relu(self.dec_in(din)).unsqueeze(1)
            dec_out, dec_state = self.decoder(din, dec_state)
            y_t = self.out(dec_out).squeeze(1)

            outs.append(y_t.unsqueeze(1))
            if self.training and (y_teacher is not None) and (np.random.rand() < tf_ratio):
                y_prev = y_teacher[:, t].unsqueeze(1)
            else:
                y_prev = y_t.detach().unsqueeze(1)

        return torch.cat(outs, dim=1)

def load_artifacts():
    print("Downloading artifacts...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bucket.blob(SCALER_FILE).download_to_filename(SCALER_FILE)
    scaler = joblib.load(SCALER_FILE)

    bucket.blob(BASELINE_FILE).download_to_filename(BASELINE_FILE)
    baseline_medians = pd.read_csv(BASELINE_FILE, index_col=0, header=None).squeeze("columns")
    
    bucket.blob(MODEL_FILE).download_to_filename(MODEL_FILE)
    
    # Safe load with dictionary support
    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)
    
    # Use defaults if metadata is missing
    input_dim = checkpoint.get('input_dim', 7) 
    enc_h = checkpoint.get('enc_h', 384)
    dec_h = checkpoint.get('dec_h', 384)
    horizon = checkpoint.get('horizon', HORIZON)
    
    model = AttnSeq2Seq(input_dim=input_dim, enc_h=enc_h, dec_h=dec_h, horizon=horizon).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Artifacts loaded.")
    return scaler, baseline_medians, model, device

def get_inference_data():
    print("Fetching data from BigQuery...")
    # Fetch 48 hours (288 rows) to allow for 24h lag + 24h input window
    query = f"""
        SELECT timestamp, flow_data
        FROM `{PROJECT_ID}.wastewater_data.sensor_readings_clean`
        WHERE sensor_id = '{SENSOR_ID}'
        ORDER BY timestamp DESC
        LIMIT {LOOKBACK * 2}
    """
    df = bq_client.query(query).to_dataframe()
    
    # Check if we have enough data BEFORE interpolation
    if len(df) < LOOKBACK * 2:
         # We might still be able to run if it's close, but let's warn
         print(f"Warning: Fetched {len(df)} rows. Ideal is {LOOKBACK*2}.")

    df = df.sort_values(by='timestamp', ascending=True)
    df['Datetime'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.set_index('Datetime').asfreq('10min')
    df['flow_data'] = df['flow_data'].interpolate(limit_direction='both')
    
    # FIX: Do NOT call .tail(LOOKBACK) here. Return the full 48h dataset.
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
    
    # Create the 24-hour lag feature
    df['resid_prevday'] = df['resid'].shift(24 * 6)
    
    # Drop the first 24 hours (which are now NaNs due to the shift)
    df = df.dropna(subset=['resid_prevday'])
    
    df[['resid_z', 'resid_diff_z']] = scaler.transform(df[['resid', 'resid_diff']].values)
    return df

def create_windows(df, baseline_medians, device):
    print("Creating windows...")
    # We should now have exactly 144 rows left after feature engineering
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

def inverse_transform(y_pred_resid_z, BL, scaler):
    y_squeezed = y_pred_resid_z.squeeze(-1) 
    resid = y_squeezed * scaler.scale_[0] + scaler.mean_[0]
    final_flow = resid + BL
    return final_flow

def save_predictions(predictions, future_times, sensor_id, last_time):
    print(f"Saving {len(predictions)} LSTM predictions...")
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
            'lstm_prediction': float(predictions_row[i]),
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
        
        # Check if we have enough raw data (allowing for some missing rows before interpolation)
        min_raw_needed = LOOKBACK * 2 - 12 # Allow missing 2 hours
        if df_raw is None or len(df_raw) < min_raw_needed:
             # The previous error was here because df_raw was already truncated to 144
            print(f"Skipping: Need at least {min_raw_needed} raw rows, got {len(df_raw) if df_raw is not None else 0}.")
            return

        df_features = feature_engineer(df_raw, scaler, baseline_medians)
        
        # After dropping the first 24h (lag), we should have at least 144 rows left
        if len(df_features) < LOOKBACK:
            print(f"Skipping: Not enough data after feature engineering. Got {len(df_features)}, need {LOOKBACK}.")
            return

        X, F, BL, future_times, last_time = create_windows(df_features, baseline_medians, device)

        with torch.no_grad():
            y_pred_z = model(X, F, None, 0.0).cpu().numpy()
        
        final_prediction = inverse_transform(y_pred_z, BL, scaler)
        save_predictions(final_prediction, future_times, SENSOR_ID, last_time)
        print("LSTM Prediction job complete.")
        
    except Exception as e:
        print(f"Job failed: {e}")
        raise e 

if __name__ == "__main__":
    main()