
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(
    page_title="Crypto Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
# ===== Color palettes (must be defined BEFORE charts use them) =====
PALETTE_OKABE_ITO = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#000000"]
PALETTE_EXTRA = ["#5778A4", "#E49444", "#D1615D", "#85B6B2", "#6A9F58", "#E7CA60", "#A87C9F", "#F1A2A9", "#967662", "#B8B0AC"]

MODEL_COLORS = {
    "ARIMA":   "#0072B2",
    "SARIMA":  "#56B4E9",
    "Prophet": "#E69F00",  # amber
    "LSTM":    "#009E73",
}
# ============================================================================
# CSS (SAME AS v6.1)
# ============================================================================

if st.session_state.theme == 'dark':
    st.markdown("""
    <style>
        * {transition: background-color 0.5s ease, color 0.5s ease, border-color 0.5s ease !important;}
        .stApp, .stApp > header, [data-testid="stHeader"] {
            background: linear-gradient(180deg, #0e1117 0%, #1a1d29 100%) !important;
        }
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(180deg, #1e2130 0%, #2a2d3a 100%) !important;
            border-right: 2px solid #8B7355 !important;
        }
        [data-testid="stSidebar"] * {color: #D4C4A8 !important;}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #C9B99B !important; font-weight: 600 !important;
        }
        [data-testid="stSidebar"] label {
            color: #C9B99B !important; font-weight: 500 !important; font-size: 1rem !important;
        }
        [data-testid="collapsedControl"] {
            background-color: #C9B99B !important;
            border: 2px solid #A67C52 !important;
            color: #0e1117 !important;
            box-shadow: 0 4px 12px rgba(201, 185, 155, 0.4) !important;
        }
        [data-testid="collapsedControl"]:hover {
            background-color: #A67C52 !important;
            border-color: #C9B99B !important;
            box-shadow: 0 6px 16px rgba(201, 185, 155, 0.6) !important;
        }
        [data-testid="collapsedControl"] svg {
            fill: #0e1117 !important; stroke: #0e1117 !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div {
            background-color: #262730 !important;
            color: #D4C4A8 !important;
            border: 2px solid #8B7355 !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
            background-color: #262730 !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
            background-color: #262730 !important; color: #D4C4A8 !important;
        }
        [data-testid="stSidebar"] div[role="listbox"] {
            background-color: #262730 !important; border: 2px solid #8B7355 !important;
        }
        [data-testid="stSidebar"] div[role="option"] {
            background-color: #262730 !important; color: #D4C4A8 !important; padding: 0.75rem !important;
        }
        [data-testid="stSidebar"] div[role="option"]:hover {
            background-color: #3a3d4a !important; color: #C9B99B !important;
        }
        .main-header {
            font-size: 2.2rem !important; color: #C9B99B !important; text-align: center;
            font-weight: 700 !important; margin-bottom: 0.5rem !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
        }
        .sub-header {
            font-size: 1.2rem !important; color: #A67C52 !important;
            text-align: center; font-weight: 400 !important;
        }
        h2, h3 {
            color: #C9B99B !important; font-weight: 600 !important;
            border-bottom: 2px solid #8B7355 !important; padding-bottom: 0.5rem !important;
        }
        .stMarkdown {color: #D4C4A8 !important;}
        [data-testid="stMetricValue"] {
            color: #C9B99B !important; font-size: 1.8rem !important; font-weight: bold !important;
        }
        [data-testid="stMetricLabel"] {color: #B8956A !important; font-weight: 500 !important;}
        [data-testid="stMetricDelta"] {color: #B8B76D !important; font-weight: 600 !important;}
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #1e2130 0%, #262730 100%) !important;
            border: 2px solid #8B7355 !important; border-radius: 10px !important;
            padding: 1rem !important; box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
        }
        div[data-testid="metric-container"]:hover {
            border-color: #C9B99B !important;
            box-shadow: 0 6px 12px rgba(201, 185, 155, 0.2) !important;
        }
        [data-testid="column"]:last-child {
            position: fixed !important; top: 0.8rem !important; right: 0.8rem !important;
            z-index: 999999 !important; width: 60px !important;
        }
        [data-testid="column"]:last-child .stButton > button {
            background: linear-gradient(135deg, #A67C52 0%, #8B7355 100%) !important;
            color: #0e1117 !important; border: 3px solid #C9B99B !important;
            padding: 0 !important; border-radius: 50% !important; font-size: 1.8rem !important;
            width: 60px !important; height: 60px !important;
            display: flex !important; align-items: center !important; justify-content: center !important;
            box-shadow: 0 4px 20px rgba(166, 124, 82, 0.5) !important;
            transition: all 0.3s ease !important; margin: 0 !important;
        }
        [data-testid="column"]:last-child .stButton > button:hover {
            transform: scale(1.15) rotate(180deg) !important;
            box-shadow: 0 6px 30px rgba(166, 124, 82, 0.7) !important;
        }
        .stSelectbox > div > div {
            background-color: #262730 !important; color: #D4C4A8 !important;
            border: 2px solid #8B7355 !important;
        }
        .stSelectbox label {color: #C9B99B !important; font-weight: 500 !important;}
        .stCheckbox > label {color: #D4C4A8 !important; font-weight: 500 !important;}
        .stCheckbox > label:hover {color: #C9B99B !important;}
        .stDownloadButton > button {
            background: linear-gradient(135deg, #262730 0%, #1e2130 100%) !important;
            color: #D4C4A8 !important; border: 2px solid #8B7355 !important;
            padding: 0.75rem 1rem !important; border-radius: 8px !important;
            font-weight: 600 !important; width: 100% !important; text-align: center !important;
            transition: all 0.3s ease !important;
        }
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #A67C52 0%, #8B7355 100%) !important;
            transform: translateY(-2px) !important; color: #0e1117 !important;
            box-shadow: 0 4px 12px rgba(166, 124, 82, 0.4) !important;
        }
        .stSuccess {
            background: linear-gradient(135deg, rgba(184, 183, 109, 0.2) 0%, rgba(184, 183, 109, 0.1) 100%) !important;
            color: #B8B76D !important; border-left: 4px solid #B8B76D !important;
            border-radius: 8px !important; padding: 1rem !important;
        }
        .stInfo {
            background: linear-gradient(135deg, rgba(166, 124, 82, 0.2) 0%, rgba(166, 124, 82, 0.1) 100%) !important;
            color: #D4C4A8 !important; border-left: 4px solid #A67C52 !important; border-radius: 8px !important;
        }
        .stWarning {
            background: linear-gradient(135deg, rgba(184, 149, 106, 0.2) 0%, rgba(184, 149, 106, 0.1) 100%) !important;
            color: #D4C4A8 !important; border-left: 4px solid #B8956A !important; border-radius: 8px !important;
        }
        .stError {
            background: linear-gradient(135deg, rgba(166, 124, 82, 0.3) 0%, rgba(166, 124, 82, 0.15) 100%) !important;
            color: #D4C4A8 !important; border-left: 4px solid #A67C52 !important; border-radius: 8px !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1e2130 !important; border-radius: 10px !important; padding: 0.5rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #8B7355 !important; font-weight: 500 !important; padding: 0.75rem 1.5rem !important;
        }
        .stTabs [aria-selected="true"] {
            color: #C9B99B !important; border-bottom: 3px solid #A67C52 !important; font-weight: 600 !important;
        }
        .dataframe {
            color: #D4C4A8 !important; background-color: #1e2130 !important;
            border: 2px solid #8B7355 !important; border-radius: 8px !important;
        }
        .dataframe thead tr th {
            background-color: #262730 !important; color: #C9B99B !important;
            border: 1px solid #8B7355 !important; font-weight: 600 !important; padding: 0.75rem !important;
        }
        .dataframe tbody tr td {
            background-color: #1e2130 !important; color: #D4C4A8 !important;
            border: 1px solid #8B7355 !important; padding: 0.75rem !important;
        }
        .dataframe tbody tr:hover td {
            background-color: #262730 !important; color: #C9B99B !important;
        }
        table {
            color: #D4C4A8 !important; background-color: #1e2130 !important;
            border: 2px solid #8B7355 !important; border-radius: 8px !important;
        }
        table thead tr th {
            background-color: #262730 !important; color: #C9B99B !important;
            border: 1px solid #8B7355 !important; font-weight: 600 !important; padding: 0.75rem !important;
        }
        table tbody tr td {
            background-color: #1e2130 !important; color: #D4C4A8 !important;
            border: 1px solid #8B7355 !important; padding: 0.75rem !important;
        }
        table tbody tr:hover td {background-color: #262730 !important;}
        hr {
            border: none !important; height: 2px !important;
            background: linear-gradient(90deg, transparent 0%, #8B7355 50%, transparent 100%) !important;
            margin: 2rem 0 !important;
        }
        #MainMenu, footer, header {visibility: hidden;}
        ::-webkit-scrollbar {width: 10px; height: 10px;}
        ::-webkit-scrollbar-track {background: #1e2130;}
        ::-webkit-scrollbar-thumb {background: #8B7355; border-radius: 5px;}
        ::-webkit-scrollbar-thumb:hover {background: #A67C52;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        * {transition: background-color 0.5s ease, color 0.5s ease !important;}
        .stApp {background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%) !important;}
        .main-header {font-size: 2.2rem !important; color: #1f77b4 !important; text-align: center; font-weight: 700 !important;}
        .sub-header {font-size: 1.2rem !important; color: #666 !important; text-align: center;}
        [data-testid="collapsedControl"] {
            background-color: #1f77b4 !important; border: 2px solid #0d47a1 !important; color: white !important;
        }
        [data-testid="collapsedControl"]:hover {background-color: #0d47a1 !important;}
        [data-testid="collapsedControl"] svg {fill: white !important;}
        [data-testid="column"]:last-child {
            position: fixed !important; top: 0.8rem !important; right: 0.8rem !important;
            z-index: 999999 !important; width: 60px !important;
        }
        [data-testid="column"]:last-child .stButton > button {
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
            color: #1f2937 !important; border: 3px solid #f59e0b !important;
            padding: 0 !important; border-radius: 50% !important; font-size: 1.8rem !important;
            width: 60px !important; height: 60px !important;
            display: flex !important; align-items: center !important; justify-content: center !important;
            box-shadow: 0 4px 20px rgba(245, 158, 11, 0.5) !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="column"]:last-child .stButton > button:hover {
            transform: scale(1.15) rotate(180deg) !important;
        }
        #MainMenu, footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

col1, col2 = st.columns([20, 1])
with col1:
    st.markdown('<div class="main-header">📈 Cryptocurrency Market Forecasting Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">🤖 Advanced Time Series Analysis with ARIMA, SARIMA, Prophet & LSTM</div>', unsafe_allow_html=True)
with col2:
    toggle_symbol = "☀️" if st.session_state.theme == 'dark' else "🌙"
    st.button(toggle_symbol, on_click=toggle_theme, key="theme_toggle", help="Toggle Theme")

st.markdown("---")

# ============================================================================
# DATA LOADING
# ============================================================================

CRYPTO_LIST = {
    'BTC-USD': 'BTC-USD',
    'ETH-USD': 'ETH-USD',
    'SOL-USD': 'SOL-USD',
    'ADA-USD': 'ADA-USD',
    'XRP-USD': 'XRP-USD',
    'DOGE-USD': 'DOGE-USD',
    'DOT-USD': 'DOT-USD',
    'AVAX-USD': 'AVAX-USD'
}

@st.cache_data
def load_data():
    try:
        btc_data = pd.read_csv('BTC_USD_data.csv', index_col=0, parse_dates=True)
        eth_data = pd.read_csv('ETH_USD_data.csv', index_col=0, parse_dates=True)
        predictions_data = joblib.load('predictions_forecasts.pkl')
        train_data = joblib.load('train_data.pkl')
        test_data = joblib.load('test_data.pkl')
        
        return {
            'BTC-USD': btc_data,
            'ETH-USD': eth_data
        }, predictions_data, train_data, test_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def ensure_columns(data):
    if 'Returns' not in data.columns:
        data['Returns'] = data['Close'].pct_change()
    if 'Volatility' not in data.columns:
        data['Volatility'] = data['Returns'].rolling(window=30).std() * np.sqrt(252) * 100
    if 'MA7' not in data.columns:
        data['MA7'] = data['Close'].rolling(window=7).mean()
    if 'MA30' not in data.columns:
        data['MA30'] = data['Close'].rolling(window=30).mean()
    if 'MA50' not in data.columns:
        data['MA50'] = data['Close'].rolling(window=50).mean()
    if 'MA200' not in data.columns:
        data['MA200'] = data['Close'].rolling(window=200).mean()
    if 'RSI' not in data.columns:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
    if 'MACD' not in data.columns:
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    if 'BB_Middle' not in data.columns:
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
    return data

@st.cache_data(ttl=3600)
def load_live_data(ticker):
    try:
        data = yf.download(ticker, period='2y', progress=False)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = ensure_columns(data)
            return data
        return None
    except:
        return None

data_dict, predictions_data, train_data, test_data = load_data()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.markdown("---")

selected_crypto = st.sidebar.selectbox(
    "Select Cryptocurrency",
    options=list(CRYPTO_LIST.keys()),
    index=0
)

crypto_name = selected_crypto

if selected_crypto in ['BTC-USD', 'ETH-USD']:
    data = data_dict[selected_crypto]
    data = ensure_columns(data)
    has_predictions = True
else:
    data = load_live_data(selected_crypto)
    has_predictions = False
    if data is None:
        st.error(f"Could not load data for {crypto_name}")
        st.stop()

selected_model = st.sidebar.selectbox(
    "Select Forecasting Model",
    options=['LSTM', 'ARIMA', 'SARIMA', 'Prophet', 'All Models'] if has_predictions else ['Live Data Only'],
    index=0
)

show_technical = st.sidebar.checkbox("Show Technical Indicators", value=True)
show_forecast = st.sidebar.checkbox("Show Future Forecast", value=True) if has_predictions else False

if has_predictions:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Performance")
    
    metrics_df = pd.DataFrame(predictions_data['all_metrics'][selected_crypto])
    best_model = metrics_df.sort_values('RMSE').iloc[0]
    
    st.sidebar.metric("🏆 Best Model", best_model['Model'])
    st.sidebar.metric("📉 RMSE", f"${best_model['RMSE']:,.2f}")
    st.sidebar.metric("📊 R² Score", f"{best_model['R² Score']:.4f}")
    st.sidebar.metric("📈 MAPE", f"{best_model['MAPE (%)']:.2f}%")

def generate_pdf_report(crypto_name, data, metrics_df=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    title = Paragraph(f"<b>Cryptocurrency Forecast Report: {crypto_name}</b>", styles['Title'])
    elements.append(title)
    doc.build(elements)
    buffer.seek(0)
    return buffer

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export Report")

pdf_buffer = generate_pdf_report(crypto_name, data, 
                                 pd.DataFrame(predictions_data['all_metrics'][selected_crypto]) if has_predictions else None)
csv_data = data.to_csv()

st.sidebar.download_button(
    label="📄 Download PDF Report",
    data=pdf_buffer,
    file_name=f"{crypto_name}_forecast_{datetime.now().strftime('%Y%m%d')}.pdf",
    mime="application/pdf",
    key="pdf_dl"
)

st.sidebar.download_button(
    label="📊 Download CSV Data",
    data=csv_data,
    file_name=f"{crypto_name}_data_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    key="csv_dl"
)

# ============================================================================
# MAIN METRICS
# ============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

current_price = data['Close'].iloc[-1]
price_change_1d = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
current_rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
current_vol = data['Volatility'].iloc[-1] if 'Volatility' in data.columns else 0
current_volume = data['Volume'].iloc[-1]

with col1:
    st.metric("💰 Current Price", f"${current_price:,.2f}", f"{price_change_1d:+.2f}%")

with col2:
    st.metric("📊 7-Day MA", f"${data['MA7'].iloc[-1]:,.2f}" if 'MA7' in data.columns else "N/A")

with col3:
    rsi_delta = "Overbought" if current_rsi > 70 else ("Oversold" if current_rsi < 30 else "Neutral")
    st.metric("📈 RSI", f"{current_rsi:.2f}", rsi_delta)

with col4:
    st.metric("🌊 Volatility (30D)", f"{current_vol:.2f}%")

with col5:
    st.metric("💹 Volume", f"{current_volume/1e9:.2f}B")

st.markdown("---")

theme_emoji = "🌙" if st.session_state.theme == 'dark' else "☀️"
st.success(f"{theme_emoji} Dashboard v6.2 Final - {crypto_name} | {selected_model}")

# ============================================================================
# CANDLESTICK CHART
# ============================================================================

st.markdown("## 📈 Price Analysis & Predictions")

chart_template = 'plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
bg_color = '#0e1117' if st.session_state.theme == 'dark' else '#ffffff'
text_color = '#D4C4A8' if st.session_state.theme == 'dark' else '#262730'
legend_font_color = '#C9B99B' if st.session_state.theme == 'dark' else '#1f2937'

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03, row_heights=[0.7, 0.3])

recent_data = data.iloc[-180:]
fig.add_trace(go.Candlestick(
    x=recent_data.index,
    open=recent_data['Open'] if 'Open' in recent_data.columns else recent_data['Close'],
    high=recent_data['High'] if 'High' in recent_data.columns else recent_data['Close'],
    low=recent_data['Low'] if 'Low' in recent_data.columns else recent_data['Close'],
    close=recent_data['Close'],
    name='OHLC',
    increasing_line_color=MODEL_COLORS["LSTM"],   # green
    decreasing_line_color="#D55E00",              # orange/red

), row=1, col=1)

if show_technical:
    if 'MA7' in data.columns:
        fig.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data['MA7'],
            mode='lines', name='MA7',
            line=dict(color="#0072B2", width=2),
            legendgroup='indicators'
        ), row=1, col=1)
    
    if 'MA30' in data.columns:
        fig.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data['MA30'],
            mode='lines', name='MA30',
            line=dict(color="#56B4E9", width=2, dash='dash'),
            legendgroup='indicators'
        ), row=1, col=1)

if show_forecast and has_predictions:
    try:
        if selected_model == 'All Models':
            model_colors = {
    'lstm': MODEL_COLORS["LSTM"],
    'arima': MODEL_COLORS["ARIMA"],
    'sarima': MODEL_COLORS["SARIMA"],
    'prophet': MODEL_COLORS["Prophet"]
}

            for model in ['lstm', 'arima', 'sarima', 'prophet']:
                pred_key = f'{model}_predictions'
                if pred_key in predictions_data and selected_crypto in predictions_data[pred_key]:
                    preds = predictions_data[pred_key][selected_crypto]
                    if isinstance(preds, pd.DataFrame) and 'predictions' in preds.columns:
                        fig.add_trace(go.Scatter(
                            x=preds.index,
                            y=preds['predictions'],
                            mode='lines',
                            name=f'{model.upper()} Forecast',
                            line=dict(color=model_colors[model], width=2.5, dash='dot'),
                            legendgroup='forecasts'
                        ), row=1, col=1)
        else:
            model_lower = selected_model.lower()
            pred_key = f'{model_lower}_predictions'
            if pred_key in predictions_data and selected_crypto in predictions_data[pred_key]:
                preds = predictions_data[pred_key][selected_crypto]
                if isinstance(preds, pd.DataFrame) and 'predictions' in preds.columns:
                    fig.add_trace(go.Scatter(
                        x=preds.index,
                        y=preds['predictions'],
                        mode='lines',
                        name=f'{selected_model} Forecast',
                        line=dict(color='#B8B76D', width=3, dash='dot'),
                        legendgroup='forecasts'
                    ), row=1, col=1)
    except Exception as e:
        st.warning(f"Could not load predictions: {str(e)}")

volume_colors = []
for i in range(len(recent_data)):
    if i == 0:
        volume_colors.append('#009E73')
    else:
        if recent_data['Close'].iloc[i] >= recent_data['Close'].iloc[i-1]:
            volume_colors.append('#009E73')
        else:
            volume_colors.append('#D55E00')

fig.add_trace(go.Bar(
    x=recent_data.index,
    y=recent_data['Volume'],
    name='Volume',
    marker_color=volume_colors,
    opacity=0.7,
    legendgroup='volume'
), row=2, col=1)

# ===== Color palettes (do NOT change your text colors) =====
PALETTE_OKABE_ITO = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#000000"]  # amber = #E69F00
PALETTE_EXTRA = ["#5778A4", "#E49444", "#D1615D", "#85B6B2", "#6A9F58", "#E7CA60", "#A87C9F", "#F1A2A9", "#967662", "#B8B0AC"]

MODEL_COLORS = {
    "ARIMA":   "#0072B2",
    "SARIMA":  "#56B4E9",
    "Prophet": "#E69F00",  # amber
    "LSTM":    "#009E73",
}


fig.update_layout(
    template=chart_template,
    plot_bgcolor=bg_color,
    paper_bgcolor=bg_color,
    font=dict(color=text_color, size=12),
    height=650,
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.01,
        bgcolor='rgba(0, 0, 0, 0)',
        bordercolor='rgba(0, 0, 0, 0)',
        borderwidth=0,
        font=dict(color=legend_font_color, size=12, family="Arial")
    ),
    showlegend=True
)

fig.update_yaxes(title_text="Price (USD)", row=1, col=1, title_font=dict(color=text_color))
fig.update_yaxes(title_text="Volume", row=2, col=1, title_font=dict(color=text_color))
fig.update_xaxes(title_text="Date", row=2, col=1, title_font=dict(color=text_color))

st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TECHNICAL INDICATORS (SAME AS BEFORE - ABBREVIATED FOR SPACE)
# ============================================================================

if show_technical:
    st.markdown("## 📊 Technical Indicators")
    
    tab1, tab2, tab3 = st.tabs(["📈 RSI Analysis", "📉 MACD Indicator", "🔔 Bollinger Bands"])
    
    with tab1:
        st.markdown("**RSI (Relative Strength Index)** - Momentum oscillator measuring speed and magnitude of price changes")
        
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=data.index[-200:], y=data['RSI'].iloc[-200:],
            mode='lines', name='RSI', line=dict(color='#C9B99B', width=3),
            fill='tozeroy', fillcolor='rgba(201, 185, 155, 0.2)'
        ))
        
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#A67C52", annotation_text="Overbought (70)", annotation_font_color=text_color)
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#8B7355", annotation_text="Oversold (30)", annotation_font_color=text_color)
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="#666", annotation_text="Neutral", annotation_font_color=text_color)
        
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="#A67C52", opacity=0.1, line_width=0)
        fig_rsi.add_hrect(y0=0, y1=30, fillcolor="#8B7355", opacity=0.1, line_width=0)
        
        fig_rsi.update_layout(
            template=chart_template, plot_bgcolor=bg_color, paper_bgcolor=bg_color,
            font=dict(color=text_color), height=450, xaxis_title="Date", yaxis_title="RSI",
            yaxis=dict(range=[0, 100]), showlegend=True,
            legend=dict(font=dict(color=legend_font_color), bgcolor='rgba(0, 0, 0, 0)', bordercolor='rgba(0, 0, 0, 0)', borderwidth=0)
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        rsi_current = data['RSI'].iloc[-1]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current RSI", f"{rsi_current:.2f}")
        with col2:
            st.metric("7-Day Avg RSI", f"{data['RSI'].iloc[-7:].mean():.2f}")
        with col3:
            status = "🔴 Overbought" if rsi_current > 70 else ("🟢 Oversold" if rsi_current < 30 else "🟡 Neutral")
            st.metric("Status", status)
    
    with tab2:
        st.markdown("**MACD (Moving Average Convergence Divergence)** - Trend-following momentum indicator")
        
        fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig_macd.add_trace(go.Scatter(x=data.index[-200:], y=data['Close'].iloc[-200:], mode='lines', name='Price', line=dict(color='#C9B99B', width=2.5)), row=1, col=1)
        fig_macd.add_trace(go.Scatter(x=data.index[-200:], y=data['MACD'].iloc[-200:], mode='lines', name='MACD', line=dict(color='#A67C52', width=2.5)), row=2, col=1)
        fig_macd.add_trace(go.Scatter(x=data.index[-200:], y=data['MACD_Signal'].iloc[-200:], mode='lines', name='Signal', line=dict(color='#8B7355', width=2.5)), row=2, col=1)
        
        hist_colors = ['#B8B76D' if val >= 0 else '#A67C52' for val in data['MACD_Hist'].iloc[-200:]]
        fig_macd.add_trace(go.Bar(x=data.index[-200:], y=data['MACD_Hist'].iloc[-200:], name='Histogram', marker_color=hist_colors, opacity=0.7), row=2, col=1)
        
        fig_macd.update_layout(
            template=chart_template, plot_bgcolor=bg_color, paper_bgcolor=bg_color,
            font=dict(color=text_color), height=550, showlegend=True, hovermode='x unified',
            legend=dict(font=dict(color=legend_font_color), bgcolor='rgba(0, 0, 0, 0)', bordercolor='rgba(0, 0, 0, 0)', borderwidth=0)
        )
        st.plotly_chart(fig_macd, use_container_width=True)
    
    with tab3:
        st.markdown("**Bollinger Bands** - Volatility bands placed above and below a moving average")
        
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=data.index[-200:], y=data['BB_Upper'].iloc[-200:], mode='lines', name='Upper Band', line=dict(color='#A67C52', width=2, dash='dash')))
        fig_bb.add_trace(go.Scatter(x=data.index[-200:], y=data['BB_Middle'].iloc[-200:], mode='lines', name='SMA (20)', line=dict(color='#C9B99B', width=2.5)))
        fig_bb.add_trace(go.Scatter(x=data.index[-200:], y=data['BB_Lower'].iloc[-200:], mode='lines', name='Lower Band', line=dict(color='#8B7355', width=2, dash='dash'), fill='tonexty', fillcolor='rgba(166, 124, 82, 0.15)'))
        fig_bb.add_trace(go.Scatter(x=data.index[-200:], y=data['Close'].iloc[-200:], mode='lines', name='Close Price', line=dict(color='#B8B76D', width=3)))
        
        fig_bb.update_layout(
            template=chart_template, plot_bgcolor=bg_color, paper_bgcolor=bg_color,
            font=dict(color=text_color), height=500, xaxis_title="Date", yaxis_title="Price (USD)", showlegend=True,
            legend=dict(font=dict(color=legend_font_color), bgcolor='rgba(0, 0, 0, 0)', bordercolor='rgba(0, 0, 0, 0)', borderwidth=0)
        )
        st.plotly_chart(fig_bb, use_container_width=True)
        
        data['BB_Width'] = ((data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']) * 100
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current BB Width", f"{data['BB_Width'].iloc[-1]:.2f}%")
        with col2:
            st.metric("30-Day Avg Width", f"{data['BB_Width'].iloc[-30:].mean():.2f}%")

# ============================================================================
# ADDITIONAL ANALYSIS
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Advanced Market Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Volume & Price Analysis")
    
    fig_vol = make_subplots(specs=[[{"secondary_y": True}]])
    
    vol_colors_60 = []
    for i in range(len(data.iloc[-60:])):
        if i == 0:
            vol_colors_60.append('#B8B76D')
        else:
            if data['Close'].iloc[-60+i] >= data['Close'].iloc[-60+i-1]:
                vol_colors_60.append('#B8B76D')
            else:
                vol_colors_60.append('#A67C52')
    
    fig_vol.add_trace(go.Bar(x=data.index[-60:], y=data['Volume'].iloc[-60:], name='Volume', marker_color=vol_colors_60, opacity=0.7), secondary_y=False)
    fig_vol.add_trace(go.Scatter(x=data.index[-60:], y=data['Close'].iloc[-60:], name='Price', line=dict(color='#C9B99B', width=2.5)), secondary_y=True)
    
    fig_vol.update_layout(
        template=chart_template, plot_bgcolor=bg_color, paper_bgcolor=bg_color,
        font=dict(color=text_color), height=400, showlegend=True,
        legend=dict(font=dict(color=legend_font_color), bgcolor='rgba(0, 0, 0, 0)', bordercolor='rgba(0, 0, 0, 0)', borderwidth=0)
    )
    fig_vol.update_yaxes(title_text="Volume", secondary_y=False, title_font=dict(color=text_color))
    fig_vol.update_yaxes(title_text="Price (USD)", secondary_y=True, title_font=dict(color=text_color))
    
    st.plotly_chart(fig_vol, use_container_width=True)

with col2:
    st.markdown("### 📊 Returns Distribution")
    
    returns = data['Returns'].dropna() * 100
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=returns[-252:], nbinsx=50, name='Daily Returns', marker_color='#C9B99B', opacity=0.8))
    
    mean_return = returns[-252:].mean()
    fig_dist.add_vline(x=mean_return, line_dash="dash", line_color="#B8B76D", line_width=2,
                      annotation_text=f"Mean: {mean_return:.2f}%", annotation_font_color=text_color)
    
    fig_dist.update_layout(
        template=chart_template, plot_bgcolor=bg_color, paper_bgcolor=bg_color,
        font=dict(color=text_color), height=400, xaxis_title="Daily Returns (%)", yaxis_title="Frequency", showlegend=True,
        legend=dict(font=dict(color=legend_font_color), bgcolor='rgba(0, 0, 0, 0)', bordercolor='rgba(0, 0, 0, 0)', borderwidth=0)
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Mean Return", f"{mean_return:.2f}%")
    with col_b:
        st.metric("Std Dev", f"{returns[-252:].std():.2f}%")

# ============================================================================
# MODEL COMPARISON - FIXED TABLE ORDER + PIE CHART
# ============================================================================

if has_predictions:
    st.markdown("---")
    st.markdown("## 🏆 Model Performance Comparison")
    
    col1, col2 = st.columns([2.5, 1.5])
    
    with col1:
        st.markdown("### 📊 Performance Metrics Table")
        comparison_df = pd.DataFrame(predictions_data['all_metrics'][selected_crypto])
        
        # FIXED: REVERSE ORDER - Show from 3, 2, 1, 0
        comparison_df_reversed = comparison_df.iloc[::-1].reset_index(drop=True)
        comparison_df_reversed.index = [3, 2, 1, 0]  # Set index to 3, 2, 1, 0
        
                # Highlight last 3 columns (RMSE, R² Score, MAPE) for LSTM row (index 3)
        def highlight_lstm_row(row):
            if row.name == 3:  # LSTM row
                return ['background-color: #ffd700; color: #0e1117; font-weight: bold' if col in ['RMSE', 'R² Score', 'MAPE (%)'] else '' for col in row.index]
            return ['' for _ in row.index]
        
        styled_df = comparison_df_reversed.style.format({
            'RMSE': '${:,.2f}',
            'MAE': '${:,.2f}',
            'R² Score': '{:.4f}',
            'MAPE (%)': '{:.2f}%'
        }).apply(highlight_lstm_row, axis=1)
        
        st.dataframe(styled_df, use_container_width=True)
    
    with col2:
        st.markdown("### 🥧 Model Accuracy Distribution")
        
        # FIXED: PIE CHART WITH ALL 4 MODELS VISIBLE
        fig_pie = go.Figure(data=[go.Pie(
            # Use absolute values of R² Score or convert to percentage for better visualization
labels=comparison_df['Model'].tolist(),
values=[max(0.01, abs(x)) for x in comparison_df['R² Score'].tolist()],  # Ensure positive values

            hole=0.35,
            marker=dict(
    colors=pie_colors,
    line=dict(color='#ffffff' if st.session_state.theme == 'light' else '#0e1117', width=2)
),
            textfont=dict(
                color='#0e1117',
                size=14,
                family="Arial Black"
            ),
            textposition='auto',
            textinfo='label+percent',
            insidetextorientation='radial',
            pull=[0, 0, 0, 0],
            hovertemplate='<b>%{label}</b><br>R² Score: %{value:.4f}<br>Share: %{percent}<extra></extra>'
        )])
        
        fig_pie.update_layout(
            template=chart_template,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            height=380,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                font=dict(color=legend_font_color, size=12),
                bgcolor='rgba(0, 0, 0, 0)',
                bordercolor='rgba(0, 0, 0, 0)',
                borderwidth=0
            ),
            margin=dict(l=20, r=120, t=40, b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

st.markdown("---")
st.markdown("## 📋 Summary Statistics")

summary_data = {
    'Metric': ['Current Price', 'All-Time High', 'All-Time Low', 'Avg Volume (30D)', 'Volatility (30D)', 'Current RSI'],
    'Value': [
        f"${current_price:,.2f}",
        f"${data['Close'].max():,.2f}",
        f"${data['Close'].min():,.2f}",
        f"{data['Volume'].iloc[-30:].mean()/1e9:.2f}B",
        f"{current_vol:.2f}%",
        f"{current_rsi:.2f}"
    ]
}
summary_df = pd.DataFrame(summary_data)
st.table(summary_df)

# ============================================================================
# TRADING SIGNALS
# ============================================================================

st.markdown("---")
st.markdown("## 🎯 Trading Signals & Market Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📈 Trend Analysis")
    if data['Close'].iloc[-1] > data['MA50'].iloc[-1]:
        st.success("🟢 **Bullish Trend**")
        st.write("Price is above MA50 indicating upward momentum")
    else:
        st.error("🔴 **Bearish Trend**")
        st.write("Price is below MA50 indicating downward pressure")
    
    if data['MA7'].iloc[-1] > data['MA30'].iloc[-1]:
        st.info("📈 Short-term: **Positive**")
    else:
        st.warning("📉 Short-term: **Negative**")

with col2:
    st.markdown("#### 🎯 Momentum Indicators")
    
    if current_rsi > 70:
        st.warning("⚠️ **RSI Overbought**")
        st.write("Consider taking profits")
    elif current_rsi < 30:
        st.success("✅ **RSI Oversold**")
        st.write("Potential buy opportunity")
    else:
        st.info("➡️ **RSI Neutral**")
        st.write("No strong signal")
    
    macd_signal = "Bullish 🟢" if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] else "Bearish 🔴"
    st.metric("MACD Signal", macd_signal)

with col3:
    st.markdown("#### 🌊 Volatility Status")
    
    avg_vol = data['Volatility'].mean()
    if current_vol > avg_vol * 1.5:
        st.error("🌊 **High Volatility**")
        st.write("Increased risk, use caution")
    elif current_vol < avg_vol * 0.5:
        st.success("😌 **Low Volatility**")
        st.write("Stable market conditions")
    else:
        st.info("📊 **Normal Volatility**")
        st.write("Standard market behavior")
    
    st.metric("Vol vs Average", f"{((current_vol / avg_vol - 1) * 100):+.1f}%")

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #8B7355; padding: 1rem;'>"
    f"✨ Dashboard v6.2 Final | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data Source: Yahoo Finance ✨"
    f"</div>",
    unsafe_allow_html=True
)
