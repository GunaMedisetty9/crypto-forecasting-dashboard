
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Page configuration
st.set_page_config(
    page_title="Crypto Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# THEME MANAGEMENT
# ============================================================================

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# ============================================================================
# PERFECT CSS - ALL ISSUES FIXED
# ============================================================================

if st.session_state.theme == 'dark':
    st.markdown("""
    <style>
        /* ===== FIX TOP WHITE AREA ===== */
        .stApp > header {
            background-color: #0e1117 !important;
        }
        
        [data-testid="stHeader"] {
            background-color: #0e1117 !important;
        }
        
        .stApp {
            background-color: #0e1117 !important;
        }
        
        /* ===== SIDEBAR DARK ===== */
        [data-testid="stSidebar"] {
            background-color: #1e2130 !important;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            background-color: #1e2130 !important;
        }
        
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #F7931A !important;
            font-size: 1.2rem !important;
        }
        
        /* ===== CONSISTENT TITLE SIZE ===== */
        .main-header {
            font-size: 2rem !important;
            color: #F7931A !important;
            text-align: center;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
            line-height: 1.2 !important;
        }
        
        .sub-header {
            font-size: 1.1rem !important;
            color: #b8b8b8 !important;
            text-align: center;
            font-weight: 400 !important;
            margin-top: 0 !important;
        }
        
        /* ===== TOGGLE BUTTON ===== */
        div[data-testid="column"]:last-child .stButton > button {
            background: transparent !important;
            color: #F7931A !important;
            border: 2px solid #F7931A !important;
            padding: 0.4rem 0.8rem !important;
            border-radius: 20px !important;
            font-size: 1.2rem !important;
            transition: all 0.3s ease !important;
            margin-top: 1rem !important;
        }
        
        div[data-testid="column"]:last-child .stButton > button:hover {
            background: #F7931A !important;
            color: #0e1117 !important;
            transform: scale(1.1) !important;
        }
        
        /* ===== CLEAN DOWNLOAD LINKS (NOT BUTTONS) ===== */
        .stDownloadButton {
            margin-bottom: 0.8rem !important;
        }
        
        .stDownloadButton > button {
            background-color: transparent !important;
            color: #ffffff !important;
            border: none !important;
            padding: 0.5rem 0.8rem !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            width: 100% !important;
            text-align: left !important;
            transition: all 0.2s ease !important;
            font-size: 0.95rem !important;
        }
        
        .stDownloadButton > button:hover {
            background-color: rgba(247, 147, 26, 0.15) !important;
            transform: translateX(5px) !important;
            color: #F7931A !important;
        }
        
        .stDownloadButton > button::before {
            margin-right: 8px;
        }
        
        /* ===== METRICS ===== */
        [data-testid="stMetricValue"] {
            color: #e5e5e5 !important;
            font-size: 1.6rem !important;
            font-weight: bold !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #b8b8b8 !important;
            font-size: 0.9rem !important;
        }
        
        [data-testid="stMetricDelta"] {
            color: #4ade80 !important;
        }
        
        /* ===== SELECT BOXES ===== */
        .stSelectbox > div > div {
            background-color: #262730 !important;
            color: #ffffff !important;
            border: 1px solid #444 !important;
        }
        
        .stSelectbox label {
            color: #ffffff !important;
        }
        
        /* ===== CHECKBOXES ===== */
        .stCheckbox > label {
            color: #ffffff !important;
            font-size: 0.95rem !important;
        }
        
        /* ===== MARKDOWN TEXT ===== */
        .stMarkdown {
            color: #d1d5db !important;
        }
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #F7931A !important;
        }
        
        /* ===== SUCCESS BOX ===== */
        .stSuccess {
            background-color: #1a4d2e !important;
            color: #4ade80 !important;
            border-left: 4px solid #22c55e !important;
        }
        
        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1e2130 !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #9ca3af !important;
        }
        
        .stTabs [aria-selected="true"] {
            color: #F7931A !important;
            border-bottom-color: #F7931A !important;
        }
        
        /* ===== HIDE BRANDING ===== */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* ===== HORIZONTAL RULE ===== */
        hr {
            border-color: #333 !important;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    # Light theme
    st.markdown("""
    <style>
        /* ===== CONSISTENT TITLE SIZE ===== */
        .main-header {
            font-size: 2rem !important;
            color: #1f77b4 !important;
            text-align: center;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
            line-height: 1.2 !important;
        }
        
        .sub-header {
            font-size: 1.1rem !important;
            color: #666 !important;
            text-align: center;
            font-weight: 400 !important;
            margin-top: 0 !important;
        }
        
        /* ===== TOGGLE BUTTON ===== */
        div[data-testid="column"]:last-child .stButton > button {
            background: transparent !important;
            color: #1f77b4 !important;
            border: 2px solid #1f77b4 !important;
            padding: 0.4rem 0.8rem !important;
            border-radius: 20px !important;
            font-size: 1.2rem !important;
            margin-top: 1rem !important;
        }
        
        div[data-testid="column"]:last-child .stButton > button:hover {
            background: #1f77b4 !important;
            color: white !important;
            transform: scale(1.1) !important;
        }
        
        /* ===== CLEAN DOWNLOAD LINKS ===== */
        .stDownloadButton > button {
            background-color: transparent !important;
            color: #1f77b4 !important;
            border: none !important;
            padding: 0.5rem 0.8rem !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            width: 100% !important;
            text-align: left !important;
            font-size: 0.95rem !important;
        }
        
        .stDownloadButton > button:hover {
            background-color: rgba(31, 119, 180, 0.1) !important;
            transform: translateX(5px) !important;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HEADER - CONSISTENT SIZE
# ============================================================================

col1, col2 = st.columns([6, 1])
with col1:
    st.markdown('<div class="main-header">📈 Cryptocurrency Market Forecasting Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">🤖 Time Series Analysis with ARIMA, SARIMA, Prophet & LSTM</div>', unsafe_allow_html=True)
with col2:
    st.button("🌓", on_click=toggle_theme, help="Toggle Theme", key="theme_toggle")

st.markdown("---")

# ============================================================================
# CRYPTOCURRENCY LIST
# ============================================================================

CRYPTO_LIST = {
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD',
    'Solana': 'SOL-USD',
    'Cardano': 'ADA-USD',
    'Ripple': 'XRP-USD',
    'Dogecoin': 'DOGE-USD',
    'Polkadot': 'DOT-USD',
    'Avalanche': 'AVAX-USD'
}

# ============================================================================
# DATA LOADING
# ============================================================================

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

@st.cache_data(ttl=3600)
def load_live_data(ticker):
    try:
        data = yf.download(ticker, period='2y', progress=False)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
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

crypto_name = st.sidebar.selectbox(
    "Select Cryptocurrency",
    options=list(CRYPTO_LIST.keys()),
    index=0
)
selected_crypto = CRYPTO_LIST[crypto_name]

# Load data
if selected_crypto in ['BTC-USD', 'ETH-USD']:
    data = data_dict[selected_crypto]
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

# ============================================================================
# MODEL PERFORMANCE
# ============================================================================

if has_predictions:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Performance")
    
    metrics_df = pd.DataFrame(predictions_data['all_metrics'][selected_crypto])
    best_model = metrics_df.sort_values('RMSE').iloc[0]
    
    st.sidebar.metric("Best Model", best_model['Model'])
    st.sidebar.metric("RMSE", f"${best_model['RMSE']:,.2f}")
    st.sidebar.metric("R² Score", f"{best_model['R² Score']:.4f}")
    st.sidebar.metric("MAPE", f"{best_model['MAPE (%)']:.2f}%")

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def generate_pdf_report(crypto_name, data, metrics_df=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    title = Paragraph(f"<b>Cryptocurrency Forecast Report: {crypto_name}</b>", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    date_text = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
    elements.append(date_text)
    elements.append(Spacer(1, 12))
    
    current_price = data['Close'].iloc[-1]
    stats_text = f"""
    <b>Current Market Status:</b><br/>
    Price: ${current_price:,.2f}<br/>
    7-Day MA: ${data['MA7'].iloc[-1]:,.2f}<br/>
    30-Day MA: ${data['MA30'].iloc[-1]:,.2f}<br/>
    """
    elements.append(Paragraph(stats_text, styles['Normal']))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ============================================================================
# CLEAN EXPORT SECTION - SIMPLE LINKS
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export Report")

pdf_buffer = generate_pdf_report(crypto_name, data, 
                                 pd.DataFrame(predictions_data['all_metrics'][selected_crypto]) if has_predictions else None)
csv_data = data.to_csv()

# Simple download links (not big buttons)
st.sidebar.download_button(
    label="📄 PDF Report",
    data=pdf_buffer,
    file_name=f"{crypto_name}_forecast_{datetime.now().strftime('%Y%m%d')}.pdf",
    mime="application/pdf",
    key="pdf_dl",
    use_container_width=False
)

st.sidebar.download_button(
    label="📊 CSV Data",
    data=csv_data,
    file_name=f"{crypto_name}_data_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    key="csv_dl",
    use_container_width=False
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
st.success(f"{theme_emoji} Dashboard v3.1 - {crypto_name} | Theme: {st.session_state.theme.capitalize()}")

# [Rest of your chart code here]
