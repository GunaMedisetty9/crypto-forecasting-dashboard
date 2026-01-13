
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
# PERFECT DARK/LIGHT THEME CSS
# ============================================================================

if st.session_state.theme == 'dark':
    st.markdown("""
    <style>
        /* ===== MAIN BACKGROUND ===== */
        .stApp {
            background-color: #0e1117 !important;
        }
        
        /* ===== SIDEBAR - DARK ===== */
        [data-testid="stSidebar"] {
            background-color: #1e2130 !important;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            background-color: #1e2130 !important;
        }
        
        /* All sidebar text white */
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        /* Sidebar headers */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #F7931A !important;
        }
        
        /* ===== TITLE HEADER ===== */
        .main-header {
            font-size: 2.5rem !important;
            color: #F7931A !important;
            text-align: center;
            font-weight: bold;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
        }
        
        /* ===== TOGGLE BUTTON - DARK STYLE ===== */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: 2px solid #F7931A !important;
            padding: 0.6rem 1.5rem !important;
            border-radius: 25px !important;
            font-weight: bold !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(247, 147, 26, 0.4) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(247, 147, 26, 0.6) !important;
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        }
        
        /* ===== DOWNLOAD BUTTONS - DIRECT DOWNLOAD ===== */
        .stDownloadButton > button {
            background-color: #2563eb !important;
            color: white !important;
            border: none !important;
            padding: 0.7rem 1.5rem !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            width: 100% !important;
            margin-left: 0 !important;
            margin-bottom: 0.8rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stDownloadButton > button:hover {
            background-color: #1d4ed8 !important;
            transform: translateX(5px) !important;
        }
        
        /* ===== METRICS - LIGHT GREY TEXT ===== */
        [data-testid="stMetricValue"] {
            color: #e5e5e5 !important;
            font-size: 1.8rem !important;
            font-weight: bold !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #b8b8b8 !important;
            font-size: 1rem !important;
        }
        
        [data-testid="stMetricDelta"] {
            color: #4ade80 !important;
        }
        
        /* ===== SELECT BOXES - DARK ===== */
        .stSelectbox > div > div {
            background-color: #262730 !important;
            color: #ffffff !important;
            border: 1px solid #444 !important;
        }
        
        /* ===== CHECKBOXES - WHITE TEXT ===== */
        .stCheckbox > label {
            color: #ffffff !important;
            font-size: 1rem !important;
        }
        
        .stCheckbox > label > span {
            color: #ffffff !important;
        }
        
        /* ===== ALL MARKDOWN TEXT - LIGHT GREY ===== */
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
            padding: 1rem !important;
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
        
        /* ===== HIDE STREAMLIT BRANDING ===== */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* ===== DATAFRAMES ===== */
        .dataframe {
            color: #ffffff !important;
            background-color: #1e2130 !important;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    # Light theme CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            font-weight: bold;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.6rem 1.5rem;
            border-radius: 25px;
            font-weight: bold;
        }
        
        .stDownloadButton > button {
            background-color: #2563eb;
            color: white;
            padding: 0.7rem 1.5rem;
            border-radius: 8px;
            width: 100%;
            margin-bottom: 0.8rem;
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HEADER WITH THEME TOGGLE
# ============================================================================

col1, col2 = st.columns([5, 1])
with col1:
    st.markdown('<p class="main-header">📈 Cryptocurrency Market Forecasting Dashboard</p>', unsafe_allow_html=True)
    st.markdown("### 🤖 Time Series Analysis with ARIMA, SARIMA, Prophet & LSTM")
with col2:
    st.button("🌓", on_click=toggle_theme, help="Toggle Dark/Light Theme")

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
# SIDEBAR CONTROLS
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
# MODEL PERFORMANCE (SIDEBAR)
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
# EXPORT SECTION (SIDEBAR) - CLEANER LAYOUT
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export Report")

# Generate reports when clicked
pdf_buffer = generate_pdf_report(crypto_name, data, 
                                 pd.DataFrame(predictions_data['all_metrics'][selected_crypto]) if has_predictions else None)
csv_data = data.to_csv()

# Direct download buttons with spacing
st.sidebar.download_button(
    label="📄 Download PDF Report",
    data=pdf_buffer,
    file_name=f"{crypto_name}_forecast_{datetime.now().strftime('%Y%m%d')}.pdf",
    mime="application/pdf",
    key="pdf_download"
)

st.sidebar.download_button(
    label="📊 Download CSV Data",
    data=csv_data,
    file_name=f"{crypto_name}_data_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    key="csv_download"
)

# ============================================================================
# MAIN DASHBOARD - METRICS
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

# Status indicator
theme_emoji = "🌙" if st.session_state.theme == 'dark' else "☀️"
st.success(f"{theme_emoji} Dashboard v3.0 - {crypto_name} | Theme: {st.session_state.theme.capitalize()}")

# [Add your chart code here - Price Analysis, Technical Indicators, etc.]
