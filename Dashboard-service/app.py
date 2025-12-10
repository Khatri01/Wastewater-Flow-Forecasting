import os
import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
from google.cloud import bigquery
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from zoneinfo import ZoneInfo
import google.auth
from datetime import datetime, timedelta
import pytz

# Initialize the Dash app
app = dash.Dash(__name__, requests_pathname_prefix=os.environ.get("SCRIPT_NAME", "/"))
server = app.server

# Initialize BigQuery client
client = bigquery.Client()


# --- Configuration ---
def get_project_id():
    pid = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if pid:
        return pid
    _, pid = google.auth.default()
    return pid


PROJECT_ID = get_project_id()
if not PROJECT_ID:
    raise RuntimeError("No GCP project ID detected.")

CLEAN_TABLE_REF = f"{PROJECT_ID}.wastewater_data.sensor_readings_clean"
PRED_TABLE_REF = f"{PROJECT_ID}.wastewater_data.sensor_predictions"
STATS_TABLE_REF = f"{PROJECT_ID}.wastewater_data.diurnal_stats_clean"
SENSOR_ID_TO_DISPLAY = "46799"
LOCAL_TIMEZONE_STR = "America/Chicago"
LOCAL_TIMEZONE = ZoneInfo(LOCAL_TIMEZONE_STR)

# Sensor coordinates
SENSOR_LAT = 27.713097
SENSOR_LON = -97.326119

# --- MODEL WEIGHTS (Rank-Based) ---
MODEL_WEIGHTS = {
    "rf_prediction": 0.333,
    "xgboost_prediction": 0.267,
    "cnn_prediction": 0.067,
    "gru_prediction": 0.200,
    "lstm_prediction": 0.133,
}

# --- STYLING CONSTANTS ---
COLORS = {
    "background": "#1e1e1e",
    "card": "#252526",
    "text": "#ffffff",
    "accent": "#636EFA",
    "inactive": "#3e3e42",
    "diurnal_mean": "#FF00FF",
    "diurnal_ci": "rgba(255, 0, 255, 0.25)",
}

BTN_STYLE = {
    "padding": "10px 20px",
    "margin": "0 5px",
    "borderRadius": "5px",
    "cursor": "pointer",
    "fontSize": "16px",
    "fontWeight": "bold",
    "transition": "all 0.3s ease",
}
BTN_ACTIVE = {
    **BTN_STYLE,
    "backgroundColor": COLORS["accent"],
    "color": "white",
    "border": "none",
    "boxShadow": "0 4px 6px rgba(0,0,0,0.3)",
}
BTN_INACTIVE = {
    **BTN_STYLE,
    "backgroundColor": COLORS["inactive"],
    "color": "#aaaaaa",
    "border": "1px solid #555",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card"],
    "borderRadius": "12px",
    "padding": "15px",
    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.3)",
    "margin": "10px",
}


def load_data_from_bigquery(lookback_hours):
    print(f"Fetching data from BigQuery (Last {lookback_hours} Hours)...")

    sql_latest_times = f"""
        SELECT 'GRU' as model_type, MAX(prediction_time) as max_time, MAX(last_data_point_time) as handoff_time
            FROM `{PRED_TABLE_REF}` WHERE sensor_id = '{SENSOR_ID_TO_DISPLAY}' AND gru_prediction IS NOT NULL
        UNION ALL
        SELECT 'RF' as model_type, MAX(prediction_time) as max_time, MAX(last_data_point_time) as handoff_time
            FROM `{PRED_TABLE_REF}` WHERE sensor_id = '{SENSOR_ID_TO_DISPLAY}' AND rf_prediction IS NOT NULL
        UNION ALL
        SELECT 'XGB' as model_type, MAX(prediction_time) as max_time, MAX(last_data_point_time) as handoff_time
            FROM `{PRED_TABLE_REF}` WHERE sensor_id = '{SENSOR_ID_TO_DISPLAY}' AND xgboost_prediction IS NOT NULL
        UNION ALL
        SELECT 'LSTM' as model_type, MAX(prediction_time) as max_time, MAX(last_data_point_time) as handoff_time
            FROM `{PRED_TABLE_REF}` WHERE sensor_id = '{SENSOR_ID_TO_DISPLAY}' AND lstm_prediction IS NOT NULL
        UNION ALL
        SELECT 'CNN' as model_type, MAX(prediction_time) as max_time, MAX(last_data_point_time) as handoff_time
            FROM `{PRED_TABLE_REF}` WHERE sensor_id = '{SENSOR_ID_TO_DISPLAY}' AND cnn_prediction IS NOT NULL
    """

    handoff_time_local = None

    try:
        latest_times_df = client.query(sql_latest_times).to_dataframe()

        if not latest_times_df.empty and latest_times_df["handoff_time"].notna().any():
            last_obs_time = latest_times_df["handoff_time"].min().to_pydatetime()
            handoff_time_local = last_obs_time.astimezone(LOCAL_TIMEZONE)
        else:
            last_obs_time = datetime.now(pytz.utc)

        sql_observed = f"""
            SELECT timestamp, flow_data FROM `{CLEAN_TABLE_REF}`
            WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {lookback_hours} HOUR)
            AND timestamp <= @last_obs_time AND sensor_id = '{SENSOR_ID_TO_DISPLAY}'
            ORDER BY timestamp ASC
        """
        job_config_obs = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "last_obs_time", "TIMESTAMP", last_obs_time
                )
            ]
        )
        df_obs = client.query(sql_observed, job_config=job_config_obs).to_dataframe()

        sql_predicted = f"""
            WITH latest_times AS ({sql_latest_times})
            SELECT
                t.forecast_time,
                t.gru_prediction,
                t.rf_prediction,
                t.xgboost_prediction,
                t.lstm_prediction,
                t.cnn_prediction
            FROM `{PRED_TABLE_REF}` t
            JOIN latest_times lp ON t.prediction_time = lp.max_time
            WHERE t.sensor_id = '{SENSOR_ID_TO_DISPLAY}'
            ORDER BY t.forecast_time ASC
        """

        df_pred_all = client.query(sql_predicted).to_dataframe()
        df_pred = df_pred_all.groupby("forecast_time").first().reset_index()

        # --- WEIGHTED ENSEMBLE CALCULATION ---
        df_pred["Weighted Average"] = 0.0
        total_weight = 0.0
        for col, weight in MODEL_WEIGHTS.items():
            if col in df_pred.columns:
                if not df_pred[col].isna().all():
                    df_pred["Weighted Average"] += df_pred[col].fillna(0) * weight
                    total_weight += weight

        if total_weight > 0:
            df_pred["Weighted Average"] /= total_weight
        else:
            df_pred["Weighted Average"] = None

        # Prepare Dataframes
        df_obs["timestamp_local"] = df_obs["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
        df_obs["type"] = "Observed Flow"
        df_obs = df_obs.rename(columns={"flow_data": "flow"})

        df_pred["timestamp_local"] = df_pred["forecast_time"].dt.tz_convert(
            LOCAL_TIMEZONE
        )

        model_cols = list(MODEL_WEIGHTS.keys())
        df_pred_long = df_pred.melt(
            id_vars=["timestamp_local"],
            value_vars=model_cols + ["Weighted Average"],
            var_name="type",
            value_name="flow",
        )
        df_pred_long["type"] = df_pred_long["type"].replace(
            {
                "gru_prediction": "GRU",
                "rf_prediction": "Random Forest",
                "xgboost_prediction": "XGBoost",
                "lstm_prediction": "LSTM",
                "cnn_prediction": "1D-CNN",
            }
        )

        # Stitching Logic
        if not df_obs.empty:
            last_point = df_obs.iloc[-1]
            bridge_data = []
            for model_type in df_pred_long["type"].unique():
                bridge_data.append(
                    {
                        "timestamp_local": last_point["timestamp_local"],
                        "flow": last_point["flow"],
                        "type": model_type,
                    }
                )
            if bridge_data:
                df_bridge = pd.DataFrame(bridge_data)
                df_pred_long = pd.concat(
                    [pd.DataFrame(bridge_data), df_pred_long], ignore_index=True
                )
                df_pred_long = df_pred_long.sort_values(by="timestamp_local")

        df_combined = pd.concat(
            [
                df_obs[["timestamp_local", "flow", "type"]],
                df_pred_long[["timestamp_local", "flow", "type"]],
            ],
            ignore_index=True,
        )

        # --- MERGE STATS (SNAP TO GRID) ---
        sql_diurnal = f"""SELECT ten_min_index, Mean_Flow_cfs, Lower_95_CI, Upper_95_CI FROM `{STATS_TABLE_REF}`"""
        df_diurnal = client.query(sql_diurnal).to_dataframe()

        def get_ten_min_index(dt):
            if pd.isna(dt):
                return None
            rounded = dt + timedelta(minutes=5)
            rounded = rounded - timedelta(
                minutes=rounded.minute % 10,
                seconds=rounded.second,
                microseconds=rounded.microsecond,
            )
            return int(rounded.hour * 6 + rounded.minute / 10)

        df_combined["ten_min_index"] = df_combined["timestamp_local"].apply(
            get_ten_min_index
        )
        df_combined = pd.merge(df_combined, df_diurnal, on="ten_min_index", how="left")

        return df_combined, handoff_time_local

    except Exception as e:
        print(f"Error querying BigQuery: {e}")
        return pd.DataFrame(), None


def create_map():
    fig = go.Figure(
        go.Scattermapbox(
            lat=[SENSOR_LAT],
            lon=[SENSOR_LON],
            mode="markers+text",
            marker=go.scattermapbox.Marker(size=25, color="rgb(239, 85, 59)"),
            text=[f"<b>Sensor {SENSOR_ID_TO_DISPLAY}</b>"],
            textposition="bottom right",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Location: Sensor {SENSOR_ID_TO_DISPLAY}",
            font=dict(color="black", size=16),
        ),
        title_x=0.5,
        hovermode="closest",
        mapbox=dict(
            style="open-street-map",
            bearing=0,
            center=go.layout.mapbox.Center(lat=SENSOR_LAT, lon=SENSOR_LON),
            pitch=0,
            zoom=15,
        ),
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        paper_bgcolor="white",
    )
    return fig


# --- Dashboard Layout ---
app.layout = html.Div(
    style={
        "backgroundColor": COLORS["background"],
        "minHeight": "100vh",
        "fontFamily": "sans-serif",
    },
    children=[
        dcc.Store(id="time-window-store", data=48),
        html.Div(
            style={
                "textAlign": "center",
                "padding": "20px",
                "borderBottom": "1px solid #333",
            },
            children=[
                html.H1(
                    "Wastewater Flow at TAMU-CC",
                    style={
                        "color": COLORS["text"],
                        "margin": "0",
                        "fontWeight": "300",
                        "letterSpacing": "1px",
                    },
                ),
                html.P(
                    "Real-time Sensor Data & AI Forecasting (Weighted Average)",
                    style={"color": "#aaaaaa", "marginTop": "5px", "fontSize": "14px"},
                ),
            ],
        ),
        html.Div(
            style={"textAlign": "center", "padding": "20px"},
            children=[
                html.Span(
                    "Select Time Range:",
                    style={
                        "color": "#cccccc",
                        "marginRight": "15px",
                        "fontSize": "18px",
                    },
                ),
                html.Button(
                    "Last 24 Hours", id="btn-24h", n_clicks=0, style=BTN_INACTIVE
                ),
                html.Button(
                    "Last 48 Hours", id="btn-48h", n_clicks=0, style=BTN_ACTIVE
                ),
            ],
        ),
        # --- NEW: Checkbox Filters ---
        html.Div(
            style={"textAlign": "center", "paddingBottom": "15px"},
            children=[
                html.Span(
                    "Select Data Series:",
                    style={
                        "color": "#cccccc",
                        "marginRight": "15px",
                        "fontWeight": "bold",
                    },
                ),
                dcc.Checklist(
                    id="model-selector",
                    options=[
                        {"label": " Observed Flow", "value": "Observed Flow"},
                        {"label": " Weighted Average", "value": "Weighted Average"},
                        {"label": " Random Forest", "value": "Random Forest"},
                        {"label": " XGBoost", "value": "XGBoost"},
                        {"label": " 1D-CNN", "value": "1D-CNN"},
                        {"label": " GRU", "value": "GRU"},
                        {"label": " LSTM", "value": "LSTM"},
                        {"label": " Historical Stats (Mean + CI)", "value": "Stats"},
                    ],
                    value=[
                        "Observed Flow",
                        "Weighted Average",
                        "Stats",
                    ],  # Default selection
                    inline=True,
                    style={"color": "white", "fontSize": "14px"},
                    labelStyle={"marginRight": "20px", "cursor": "pointer"},
                ),
            ],
        ),
        html.Div(
            style={"display": "flex", "flexWrap": "wrap", "padding": "0 20px"},
            children=[
                html.Div(
                    style={"flex": "3", "minWidth": "600px"},
                    children=[
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                dcc.Graph(
                                    id="live-flow-chart",
                                    style={"height": "65vh"},
                                    config={"displayModeBar": False},
                                )
                            ],
                        )
                    ],
                ),
                html.Div(
                    style={"flex": "1", "minWidth": "300px"},
                    children=[
                        html.Div(
                            style={**CARD_STYLE, "backgroundColor": "#ffffff"},
                            children=[
                                dcc.Graph(
                                    id="sensor-map",
                                    figure=create_map(),
                                    style={"height": "65vh"},
                                    config={"displayModeBar": False},
                                )
                            ],
                        )
                    ],
                ),
            ],
        ),
        html.Div(
            style={
                "textAlign": "center",
                "color": "#555",
                "padding": "20px",
                "fontSize": "12px",
            },
            children=[
                "Data updates automatically every 30 minutes. | Models: GRU, RF, XGBoost, LSTM, 1D-CNN"
            ],
        ),
        dcc.Interval(id="interval-component", interval=30 * 60 * 1000, n_intervals=0),
    ],
)


@app.callback(
    [
        Output("time-window-store", "data"),
        Output("btn-24h", "style"),
        Output("btn-48h", "style"),
    ],
    [Input("btn-24h", "n_clicks"), Input("btn-48h", "n_clicks")],
    [State("time-window-store", "data")],
)
def update_button_styles(btn24, btn48, current_val):
    trigger = ctx.triggered_id
    hours = 48
    if trigger == "btn-24h":
        hours = 24
    elif trigger == "btn-48h":
        hours = 48
    elif current_val:
        hours = current_val
    return (
        hours,
        (BTN_ACTIVE if hours == 24 else BTN_INACTIVE),
        (BTN_ACTIVE if hours == 48 else BTN_INACTIVE),
    )


@app.callback(
    Output("live-flow-chart", "figure"),
    [
        Input("interval-component", "n_intervals"),
        Input("time-window-store", "data"),
        Input("model-selector", "value"),
    ],
)
def update_graph_live(n, lookback_hours, selected_models):
    if lookback_hours is None:
        lookback_hours = 48
    df_combined, handoff_time = load_data_from_bigquery(lookback_hours)
    if df_combined.empty:
        return dash.no_update

    fig = go.Figure()

    color_map = {
        "Observed Flow": "#636EFA",
        "Weighted Average": "#FECB52",
        "Random Forest": "#00CC96",
        "XGBoost": "#AB63FA",
        "1D-CNN": "#00D084",
        "GRU": "#EF553B",
        "LSTM": "#FFA15A",
    }

    df_hist_ci = df_combined.drop_duplicates(subset=["timestamp_local"]).sort_values(
        by="timestamp_local"
    )

    # --- FILTER STATS: Only show for FORECAST period ---
    if handoff_time is not None:
        start_plot = handoff_time - timedelta(minutes=10)
        df_stats_plot = df_hist_ci[df_hist_ci["timestamp_local"] >= start_plot].copy()
    else:
        df_stats_plot = pd.DataFrame()

    # Plot Stats (if selected)
    if "Stats" in selected_models and not df_stats_plot.empty:
        fig.add_trace(
            go.Scatter(
                x=df_stats_plot["timestamp_local"],
                y=df_stats_plot["Upper_95_CI"],
                mode="lines",
                name="95% CI",
                line=dict(width=0, color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_stats_plot["timestamp_local"],
                y=df_stats_plot["Lower_95_CI"],
                mode="lines",
                name="95% CI",
                line=dict(width=0, color="rgba(255,255,255,0)"),
                fill="tonexty",
                fillcolor=COLORS["diurnal_ci"],
                hoverinfo="skip",
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_stats_plot["timestamp_local"],
                y=df_stats_plot["Mean_Flow_cfs"],
                mode="lines",
                name="Historical Mean",
                line=dict(dash="dot", width=2, color=COLORS["diurnal_mean"]),
                opacity=1.0,
            )
        )

    # Plot Selected Models
    for trace_name in df_combined["type"].unique():
        if trace_name in selected_models:
            df_trace = df_combined[df_combined["type"] == trace_name]
            line_width = (
                4
                if trace_name == "Weighted Average"
                else 3 if trace_name == "Observed Flow" else 2
            )
            opacity = (
                1.0 if trace_name in ["Observed Flow", "Weighted Average"] else 0.8
            )

            fig.add_trace(
                go.Scatter(
                    x=df_trace["timestamp_local"],
                    y=df_trace["flow"],
                    mode="lines",
                    name=trace_name,
                    line=dict(
                        color=color_map.get(trace_name, "#FFFFFF"), width=line_width
                    ),
                    opacity=opacity,
                    showlegend=True,
                )
            )

    fig.update_layout(
        title=f"Flow Rate vs. Time (Last {lookback_hours} Hours)",
        xaxis_title="Date and Time",
        yaxis_title="Flow (cfs)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_x=0.5,
        xaxis_dtick=3600000,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin={"r": 10, "t": 50, "l": 10, "b": 50},
    )
    if handoff_time is not None:
        try:
            forecast_end = df_combined[df_combined["type"] != "Observed Flow"][
                "timestamp_local"
            ].max()
            fig.add_vrect(
                x0=handoff_time,
                x1=forecast_end,
                fillcolor="white",
                opacity=0.15,
                line_width=0,
                annotation_text="Forecast Window",
                annotation_position="top left",
            )
        except Exception:
            pass
    return fig


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
