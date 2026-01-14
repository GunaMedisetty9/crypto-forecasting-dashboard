import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Crypto Forecasting Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

PALETTE_OKABE_ITO = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#000000"]
PALETTE_EXTRA = ["#5778A4", "#E49444", "#D1615D", "#85B6B2", "#6A9F58", "#E7CA60", "#A87C9F", "#F1A2A9", "#967662", "#B8B0AC"]

MODEL_COLORS = {
    "ARIMA":   "#0072B2",
    "SARIMA":  "#56B4E9",
    "Prophet": "#E69F00",
    "LSTM":    "#009E73",
}

# ===== FIXED THEME CSS =====
if st.session_state.theme == "dark":
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(180deg, #0e1117 0%, #1a1d29 100%) !important;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }

        [data-testid="stToolbar"] {
            background: transparent !important;
        }

        [data-testid="stToolbar"] button {
            background-color: rgba(166, 124, 82, 0.3) !important;
            border: 1px solid #8B7355 !important;
            color: #D4C4A8 !important;
            border-radius: 6px !important;
        }

        [data-testid="stToolbar"] button:hover {
            background-color: #A67C52 !important;
            border-color: #C9B99B !important;
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(180deg, #1e2130 0%, #2a2d3a 100%) !important;
            border-right: 2px solid #8B7355 !important;
        }

        [data-testid="stSidebar"] {
            color: #D4C4A8 !important;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #C9B99B !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] label {
            color: #C9B99B !important;
            font-weight: 500 !important;
            font-size: 1rem !important;
        }

        [data-testid="collapsedControl"] {
            background-color: #C9B99B !important;
            border: 2px solid #A67C52 !important;
        }

        div[data-testid="stSidebarCollapseButton"] {
            background-color: #C9B99B !important;
            border: 2px solid #A67C52 !important;
            color: #0e1117 !important;
        }

        .main-header {
            font-size: 2.2rem !important;
            color: #C9B99B !important;
            text-align: center;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
        }

        .sub-header {
            font-size: 1.2rem !important;
            color: #A67C52 !important;
            text-align: center;
            font-weight: 400 !important;
        }

        h2, h3 {
            color: #C9B99B !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #8B7355 !important;
            padding-bottom: 0.5rem !important;
        }

        .stMarkdown {
            color: #D4C4A8 !important;
        }

        [data-testid="stMetricValue"] {
            color: #C9B99B !important;
            font-size: 1.8rem !important;
            font-weight: bold !important;
        }

        [data-testid="stMetricLabel"] {
            color: #B8956A !important;
            font-weight: 500 !important;
        }

        [data-testid="stMetricDelta"] {
            color: #B8B76D !important;
            font-weight: 600 !important;
        }

        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #1e2130 0%, #262730 100%) !important;
            border: 2px solid #8B7355 !important;
            border-radius: 10px !important;
            padding: 1rem !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
        }

        div[data-testid="metric-container"]:hover {
            border-color: #C9B99B !important;
            box-shadow: 0 6px 12px rgba(201, 185, 155, 0.2) !important;
        }

        .stSelectbox div, .stSelectbox label {
            color: #C9B99B !important;
            font-weight: 500 !important;
        }

        .stCheckbox label {
            color: #D4C4A8 !important;
            font-weight: 500 !important;
        }

        .stCheckbox label:hover {
            color: #C9B99B !important;
        }

        .stDownloadButton button {
            background: linear-gradient(135deg, #262730 0%, #1e2130 100%) !important;
            color: #D4C4A8 !important;
            border: 2px solid #8B7355 !important;
            padding: 0.75rem 1rem !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            width: 100% !important;
            text-align: center !important;
            transition: all 0.3s ease !important;
        }

        .stDownloadButton button:hover {
            background: linear-gradient(135deg, #A67C52 0%, #8B7355 100%) !important;
            transform: translateY(-2px) !important;
            color: #0e1117 !important;
            box-shadow: 0 4px 12px rgba(166, 124, 82, 0.4) !important;
        }

        .stSuccess {
            background: linear-gradient(135deg, rgba(184, 183, 109, 0.2) 0%, rgba(184, 183, 109, 0.1) 100%) !important;
            color: #B8B76D !important;
            border-left: 4px solid #B8B76D !important;
            border-radius: 8px !important;
            padding: 1rem !important;
        }

        .stInfo {
            background: linear-gradient(135deg, rgba(166, 124, 82, 0.2) 0%, rgba(166, 124, 82, 0.1) 100%) !important;
            color: #D4C4A8 !important;
            border-left: 4px solid #A67C52 !important;
            border-radius: 8px !important;
        }

        .stWarning {
            background: linear-gradient(135deg, rgba(184, 149, 106, 0.2) 0%, rgba(184, 149, 106, 0.1) 100%) !important;
            color: #D4C4A8 !important;
            border-left: 4px solid #B8956A !important;
            border-radius: 8px !important;
        }

        .stError {
            background: linear-gradient(135deg, rgba(166, 124, 82, 0.3) 0%, rgba(166, 124, 82, 0.15) 100%) !important;
            color: #D4C4A8 !important;
            border-left: 4px solid #A67C52 !important;
            border-radius: 8px !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            background-color: #1e2130 !important;
            border-radius: 10px !important;
            padding: 0.5rem !important;
        }

        .stTabs [data-baseweb="tab"] {
            color: #8B7355 !important;
            font-weight: 500 !important;
            padding: 0.75rem 1.5rem !important;
        }

        .stTabs [aria-selected="true"] {
            color: #C9B99B !important;
            border-bottom: 3px solid #D4C4A8 !important;
            font-weight: 600 !important;
        }

        .stTabs div[data-baseweb="tab-highlight"] {
            background-color: transparent !important;
        }

        .dataframe {
            color: #D4C4A8 !important;
            background-color: #1e2130 !important;
            border: 2px solid #8B7355 !important;
            border-radius: 8px !important;
        }

        .dataframe thead tr th {
            background-color: #262730 !important;
            color: #C9B99B !important;
            border: 1px solid #8B7355 !important;
            font-weight: 600 !important;
            padding: 0.75rem !important;
        }

        .dataframe tbody tr td {
            background-color: #1e2130 !important;
            color: #D4C4A8 !important;
            border: 1px solid #8B7355 !important;
            padding: 0.75rem !important;
        }

        .dataframe tbody tr:hover td {
            background-color: #262730 !important;
            color: #C9B99B !important;
        }

        hr {
            border: none !important;
            height: 2px !important;
            background: linear-gradient(90deg, transparent 0%, #8B7355 50%, transparent 100%) !important;
            margin: 2rem 0 !important;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #1e2130;
        }
        ::-webkit-scrollbar-thumb {
            background: #8B7355;
            border-radius: 5px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #A67C52;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%) !important;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ===== CRYPTO LIST =====
CRYPTO_LIST = {
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "SOL-USD": "SOL-USD",
    "ADA-USD": "ADA-USD",
    "XRP-USD": "XRP-USD",
    "DOGE-USD": "DOGE-USD",
    "DOT-USD": "DOT-USD",
    "AVAX-USD": "AVAX-USD",
}

@st.cache_data
def load_data():
    try:
        btc_data = pd.read_csv("BTC_USD_data.csv", index_col=0, parse_dates=True)
        eth_data = pd.read_csv("ETH_USD_data.csv", index_col=0, parse_dates=True)
        predictions_data = joblib.load("predictions_forecasts.pkl")
        train_data = joblib.load("train_data.pkl")
        test_data = joblib.load("test_data.pkl")
        return {
            "BTC-USD": btc_data,
            "ETH-USD": eth_data,
        }, predictions_data, train_data, test_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def ensure_columns(data):
    if "Returns" not in data.columns:
        data["Returns"] = data["Close"].pct_change()
    if "Volatility" not in data.columns:
        data["Volatility"] = data["Returns"].rolling(window=30).std() * np.sqrt(252) * 100
    if "MA7" not in data.columns:
        data["MA7"] = data["Close"].rolling(window=7).mean()
    if "MA30" not in data.columns:
        data["MA30"] = data["Close"].rolling(window=30).mean()
    if "MA50" not in data.columns:
        data["MA50"] = data["Close"].rolling(window=50).mean()
    if "MA200" not in data.columns:
        data["MA200"] = data["Close"].rolling(window=200).mean()
    if "RSI" not in data.columns:
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))
    if "MACD" not in data.columns:
        exp1 = data["Close"].ewm(span=12, adjust=False).mean()
        exp2 = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = exp1 - exp2
        data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
        data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]
    if "BB_Middle" not in data.columns:
        data["BB_Middle"] = data["Close"].rolling(window=20).mean()
        data["BB_Std"] = data["Close"].rolling(window=20).std()
        data["BB_Upper"] = data["BB_Middle"] + data["BB_Std"] * 2
        data["BB_Lower"] = data["BB_Middle"] - data["BB_Std"] * 2
    return data

@st.cache_data(ttl=3600)
def load_live_data(ticker):
    try:
        data = yf.download(ticker, period="2y", progress=False)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = ensure_columns(data)
            return data
        return None
    except:
        return None

def filter_by_time_range(df: pd.DataFrame, time_range: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.sort_index()
    end = df.index.max()
    if time_range == "All":
        return df
    if time_range == "1Y":
        start = end - pd.DateOffset(years=1)
    elif time_range == "3Y":
        start = end - pd.DateOffset(years=3)
    elif time_range == "5Y":
        start = end - pd.DateOffset(years=5)
    else:
        return df
    return df.loc[df.index >= start]

data_dict, predictions_data, train_data, test_data = load_data()

st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("---")

selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", options=list(CRYPTO_LIST.keys()), index=0)
crypto_name = CRYPTO_LIST[selected_crypto]

if selected_crypto in ["BTC-USD", "ETH-USD"]:
    data = data_dict[selected_crypto]
    data = ensure_columns(data)
    has_predictions = True
else:
    data = load_live_data(selected_crypto)
    has_predictions = False

if data is None:
    st.error(f"Could not load data for {crypto_name}")
    st.stop()

selected_model = st.sidebar.selectbox("Select Forecasting Model", options=["LSTM", "ARIMA", "SARIMA", "Prophet", "All Models"] if has_predictions else ["Live Data Only"], index=0)
show_technical = st.sidebar.checkbox("Show Technical Indicators", value=True)
show_forecast = st.sidebar.checkbox("Show Future Forecast", value=True) if has_predictions else False

st.sidebar.markdown("---")
time_range = st.sidebar.radio("Time Range", options=["1Y", "3Y", "5Y", "All"], index=2, horizontal=True)

if has_predictions:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Performance**")
    metrics_df = pd.DataFrame(predictions_data["all_metrics"][selected_crypto])
    best_model = metrics_df.sort_values("RMSE").iloc[0]
    st.sidebar.metric("Best Model", best_model["Model"])
    st.sidebar.metric("RMSE", f"{best_model['RMSE']:.2f}")
    st.sidebar.metric("R² Score", f"{best_model['R² Score']:.4f}")
    st.sidebar.metric("MAPE", f"{best_model['MAPE']:.2f}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Export Report**")
pdf_buffer = BytesIO()
csv_data = data.to_csv()
st.sidebar.download_button(label="Download CSV Data", data=csv_data, file_name=f"{crypto_name}_data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", key="csv_dl")

plot_data = filter_by_time_range(data, time_range)

col1, col2, col3, col4, col5 = st.columns(5)

current_price = data["Close"].iloc[-1]
price_change_1d = (data["Close"].iloc[-1] - data["Close"].iloc[-2]) / data["Close"].iloc[-2] * 100
current_rsi = data["RSI"].iloc[-1] if "RSI" in data.columns else 50
current_vol = data["Volatility"].iloc[-1] if "Volatility" in data.columns else 0
current_volume = data["Volume"].iloc[-1]

with col1:
    st.metric("Current Price", f"${current_price:.2f}", f"{price_change_1d:.2f}%")
with col2:
    st.metric("7-Day MA", f"${data['MA7'].iloc[-1]:.2f}" if "MA7" in data.columns else "NA")
with col3:
    rsi_delta = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
    st.metric("RSI", f"{current_rsi:.2f}", rsi_delta)
with col4:
    st.metric("Volatility (30D)", f"{current_vol:.2f}%")
with col5:
    st.metric("Volume", f"{current_volume/1e9:.2f}B")

st.markdown("---")
theme_emoji = "🌙" if st.session_state.theme == "dark" else "☀️"
st.success(f"{theme_emoji} Dashboard v6.2 Final - {crypto_name} | {selected_model}")

st.markdown("## 📈 Price Analysis & Predictions")

chart_template = "plotly_dark" if st.session_state.theme == "dark" else "plotly_white"
bg_color = "#0e1117" if st.session_state.theme == "dark" else "#ffffff"
text_color = "#D4C4A8" if st.session_state.theme == "dark" else "#262730"
legend_font_color = "#C9B99B" if st.session_state.theme == "dark" else "#1f2937"

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

recent_data = plot_data.copy()

fig.add_trace(go.Candlestick(
    x=recent_data.index,
    open=recent_data["Open"] if "Open" in recent_data.columns else recent_data["Close"],
    high=recent_data["High"] if "High" in recent_data.columns else recent_data["Close"],
    low=recent_data["Low"] if "Low" in recent_data.columns else recent_data["Close"],
    close=recent_data["Close"],
    name="OHLC",
    increasing_line_color=MODEL_COLORS["LSTM"],
    decreasing_line_color="#D55E00",
), row=1, col=1)

if show_technical:
    if "MA7" in data.columns:
        fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data["MA7"], mode="lines", name="MA7", line=dict(color="#0072B2", width=2), legendgroup="indicators"), row=1, col=1)
    if "MA30" in data.columns:
        fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data["MA30"], mode="lines", name="MA30", line=dict(color="#56B4E9", width=2, dash="dash"), legendgroup="indicators"), row=1, col=1)

if show_forecast and has_predictions:
    try:
        if selected_model == "All Models":
            model_colors = {"lstm": MODEL_COLORS["LSTM"], "arima": MODEL_COLORS["ARIMA"], "sarima": MODEL_COLORS["SARIMA"], "prophet": MODEL_COLORS["Prophet"]}
            for model in ["lstm", "arima", "sarima", "prophet"]:
                pred_key = f"{model}_predictions"
                if pred_key in predictions_data and selected_crypto in predictions_data[pred_key]:
                    preds = predictions_data[pred_key][selected_crypto]
                    if isinstance(preds, pd.DataFrame) and "predictions" in preds.columns:
                        fig.add_trace(go.Scatter(x=preds.index, y=preds["predictions"], mode="lines", name=f"{model.upper()} Forecast", line=dict(color=model_colors[model], width=2.5, dash="dot"), legendgroup="forecasts"), row=1, col=1)
        else:
            model_lower = selected_model.lower()
            pred_key = f"{model_lower}_predictions"
            if pred_key in predictions_data and selected_crypto in predictions_data[pred_key]:
                preds = predictions_data[pred_key][selected_crypto]
                if isinstance(preds, pd.DataFrame) and "predictions" in preds.columns:
                    fig.add_trace(go.Scatter(x=preds.index, y=preds["predictions"], mode="lines", name=f"{selected_model} Forecast", line=dict(color="#B8B76D", width=3, dash="dot"), legendgroup="forecasts"), row=1, col=1)
    except Exception as e:
        st.warning(f"Could not load predictions: {str(e)}")

if len(recent_data) > 365:
    volume_data = recent_data["Volume"].resample("W").sum()
    weekly_close = recent_data["Close"].resample("W").last()
    up_mask = weekly_close >= weekly_close.shift(1)
    vol_up = volume_data.where(up_mask, 0)
    vol_down = volume_data.where(~up_mask, 0)
    
    fig.add_trace(go.Bar(x=volume_data.index, y=vol_up, name="Volume Up", marker_color="#009E73", opacity=0.7, legendgroup="volume"), row=2, col=1)
    fig.add_trace(go.Bar(x=volume_data.index, y=vol_down, name="Volume Down", marker_color="#D55E00", opacity=0.7, legendgroup="volume"), row=2, col=1)
else:
    up_mask = recent_data["Close"] >= recent_data["Close"].shift(1)
    vol_up = recent_data["Volume"].where(up_mask, 0)
    vol_down = recent_data["Volume"].where(~up_mask, 0)
    
    fig.add_trace(go.Bar(x=recent_data.index, y=vol_up, name="Volume Up", marker_color="#009E73", opacity=0.7, legendgroup="volume"), row=2, col=1)
    fig.add_trace(go.Bar(x=recent_data.index, y=vol_down, name="Volume Down", marker_color="#D55E00", opacity=0.7, legendgroup="volume"), row=2, col=1)

fig.update_layout(
    template=chart_template,
    plot_bgcolor=bg_color,
    paper_bgcolor=bg_color,
    font=dict(color=text_color, size=12),
    height=650,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.01,
        bgcolor="rgba(0, 0, 0, 0)",
        bordercolor="rgba(0, 0, 0, 0)",
        borderwidth=0,
        font=dict(color=legend_font_color, size=12, family="Arial")
    ),
    showlegend=True
)

fig.update_yaxes(title_text="Price (USD)", row=1, col=1, title_font=dict(color=text_color))
fig.update_yaxes(title_text="Volume", row=2, col=1, title_font=dict(color=text_color))
fig.update_xaxes(title_text="Date", row=2, col=1, title_font=dict(color=text_color))

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("💡 **Data Source:** Yahoo Finance | **Last Updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
