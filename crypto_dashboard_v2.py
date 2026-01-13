
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from datetime import datetime
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

# ============================================================================
# WARM BROWN/SEPIA COLOR SCHEME
# ============================================================================

if st.session_state.theme == 'dark':
    st.markdown("""
    <style>
        /* ===== SMOOTH TRANSITIONS ===== */
        * {
            transition: background-color 0.5s ease, color 0.5s ease, border-color 0.5s ease !important;
        }
        
        /* ===== BACKGROUNDS ===== */
        .stApp, .stApp > header, [data-testid="stHeader"] {
            background-color: #0e1117 !important;
        }
        
        /* ===== SIDEBAR - WARM BROWN TEXT ===== */
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
            background-color: #1e2130 !important;
        }
        
        /* All sidebar text in warm brown/beige */
        [data-testid="stSidebar"] * {
            color: #D4C4A8 !important;
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .stMarkdown {
            color: #C9B99B !important;
            font-weight: 500 !important;
        }
        
        /* Sidebar labels */
        [data-testid="stSidebar"] label {
            color: #B8956A !important;
        }
        
        /* ===== MAIN HEADERS - WARM BROWN ===== */
        .main-header {
            font-size: 2rem !important;
            color: #C9B99B !important;
            text-align: center;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
        }
        
        .sub-header {
            font-size: 1.1rem !important;
            color: #A67C52 !important;
            text-align: center;
            font-weight: 400 !important;
        }
        
        /* ===== ALL MARKDOWN TEXT - WARM BEIGE ===== */
        .stMarkdown {
            color: #D4C4A8 !important;
        }
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #C9B99B !important;
        }
        
        .stMarkdown p, .stMarkdown span, .stMarkdown div {
            color: #D4C4A8 !important;
        }
        
        /* ===== METRICS - WARM TONES ===== */
        [data-testid="stMetricValue"] {
            color: #D4C4A8 !important;
            font-size: 1.6rem !important;
            font-weight: bold !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #B8956A !important;
            font-size: 0.9rem !important;
        }
        
        [data-testid="stMetricDelta"] {
            color: #8B7355 !important;
        }
        
        /* ===== FLOATING TOGGLE - TOP RIGHT ===== */
        [data-testid="column"]:last-child {
            position: fixed !important;
            top: 1rem !important;
            right: 1rem !important;
            z-index: 999999 !important;
            width: 60px !important;
        }
        
        [data-testid="column"]:last-child .stButton > button {
            background: linear-gradient(135deg, #A67C52 0%, #8B7355 100%) !important;
            color: #0e1117 !important;
            border: 2px solid #C9B99B !important;
            padding: 0 !important;
            border-radius: 50% !important;
            font-size: 1.8rem !important;
            width: 60px !important;
            height: 60px !important;
            min-width: 60px !important;
            min-height: 60px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: 0 4px 20px rgba(166, 124, 82, 0.4) !important;
            transition: all 0.3s ease !important;
            margin: 0 !important;
        }
        
        [data-testid="column"]:last-child .stButton > button:hover {
            transform: scale(1.1) rotate(15deg) !important;
            box-shadow: 0 6px 30px rgba(166, 124, 82, 0.6) !important;
            background: linear-gradient(135deg, #8B7355 0%, #A67C52 100%) !important;
        }
        
        /* ===== SELECT BOXES - WARM COLORS ===== */
        .stSelectbox > div > div {
            background-color: #262730 !important;
            color: #D4C4A8 !important;
            border: 1px solid #8B7355 !important;
        }
        
        .stSelectbox label {
            color: #B8956A !important;
        }
        
        /* Select dropdown options */
        .stSelectbox option {
            color: #D4C4A8 !important;
            background-color: #262730 !important;
        }
        
        /* ===== CHECKBOXES - WARM TEXT ===== */
        .stCheckbox > label {
            color: #D4C4A8 !important;
            font-size: 0.95rem !important;
        }
        
        .stCheckbox > label > span {
            color: #D4C4A8 !important;
        }
        
        /* ===== DOWNLOAD BUTTONS - WARM HOVER ===== */
        .stDownloadButton > button {
            background-color: transparent !important;
            color: #D4C4A8 !important;
            border: none !important;
            padding: 0.5rem 0.8rem !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            width: 100% !important;
            text-align: left !important;
            font-size: 0.95rem !important;
        }
        
        .stDownloadButton > button:hover {
            background-color: rgba(166, 124, 82, 0.15) !important;
            transform: translateX(5px) !important;
            color: #C9B99B !important;
        }
        
        /* ===== SUCCESS BOX - WARM TONES ===== */
        .stSuccess {
            background-color: rgba(139, 115, 85, 0.2) !important;
            color: #C9B99B !important;
            border-left: 4px solid #A67C52 !important;
        }
        
        /* ===== INFO/WARNING BOXES ===== */
        .stInfo {
            background-color: rgba(166, 124, 82, 0.2) !important;
            color: #D4C4A8 !important;
            border-left: 4px solid #B8956A !important;
        }
        
        .stWarning {
            background-color: rgba(184, 149, 106, 0.2) !important;
            color: #D4C4A8 !important;
            border-left: 4px solid #B8956A !important;
        }
        
        /* ===== TABS - WARM COLORS ===== */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1e2130 !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #8B7355 !important;
        }
        
        .stTabs [aria-selected="true"] {
            color: #C9B99B !important;
            border-bottom-color: #A67C52 !important;
        }
        
        /* ===== DATAFRAMES/TABLES - WARM TEXT ===== */
        .dataframe {
            color: #D4C4A8 !important;
            background-color: #1e2130 !important;
        }
        
        .dataframe th {
            color: #C9B99B !important;
            background-color: #262730 !important;
        }
        
        .dataframe td {
            color: #D4C4A8 !important;
        }
        
        /* ===== EXPANDER - WARM COLORS ===== */
        .streamlit-expanderHeader {
            color: #C9B99B !important;
        }
        
        /* ===== INPUT FIELDS ===== */
        .stTextInput input {
            color: #D4C4A8 !important;
            background-color: #262730 !important;
            border: 1px solid #8B7355 !important;
        }
        
        .stTextInput label {
            color: #B8956A !important;
        }
        
        /* ===== HORIZONTAL RULE ===== */
        hr {
            border-color: #8B7355 !important;
        }
        
        /* ===== HIDE BRANDING ===== */
        #MainMenu, footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
else:
    # Light theme (keep as is)
    st.markdown("""
    <style>
        * {
            transition: background-color 0.5s ease, color 0.5s ease !important;
        }
        
        .main-header {
            font-size: 2rem !important;
            color: #1f77b4 !important;
            text-align: center;
            font-weight: 700 !important;
        }
        
        .sub-header {
            font-size: 1.1rem !important;
            color: #666 !important;
            text-align: center;
        }
        
        [data-testid="column"]:last-child {
            position: fixed !important;
            top: 1rem !important;
            right: 1rem !important;
            z-index: 999999 !important;
            width: 60px !important;
        }
        
        [data-testid="column"]:last-child .stButton > button {
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
            color: #1f2937 !important;
            border: none !important;
            padding: 0 !important;
            border-radius: 50% !important;
            font-size: 1.8rem !important;
            width: 60px !important;
            height: 60px !important;
            min-width: 60px !important;
            min-height: 60px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: 0 4px 20px rgba(245, 158, 11, 0.4) !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="column"]:last-child .stButton > button:hover {
            transform: scale(1.1) rotate(15deg) !important;
            box-shadow: 0 6px 30px rgba(245, 158, 11, 0.6) !important;
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
    st.markdown('<div class="sub-header">🤖 Time Series Analysis with ARIMA, SARIMA, Prophet & LSTM</div>', unsafe_allow_html=True)
with col2:
    toggle_symbol = "☀️" if st.session_state.theme == 'dark' else "🌙"
    st.button(toggle_symbol, on_click=toggle_theme, key="theme_toggle", help="Toggle Theme")

st.markdown("---")

# ============================================================================
# DATA & DASHBOARD (Same as v3.3)
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

st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.markdown("---")

crypto_name = st.sidebar.selectbox(
    "Select Cryptocurrency",
    options=list(CRYPTO_LIST.keys()),
    index=0
)
selected_crypto = CRYPTO_LIST[crypto_name]

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

if has_predictions:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Performance")
    
    metrics_df = pd.DataFrame(predictions_data['all_metrics'][selected_crypto])
    best_model = metrics_df.sort_values('RMSE').iloc[0]
    
    st.sidebar.metric("Best Model", best_model['Model'])
    st.sidebar.metric("RMSE", f"${best_model['RMSE']:,.2f}")
    st.sidebar.metric("R² Score", f"{best_model['R² Score']:.4f}")
    st.sidebar.metric("MAPE", f"{best_model['MAPE (%)']:.2f}%")

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
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export Report")

pdf_buffer = generate_pdf_report(crypto_name, data, 
                                 pd.DataFrame(predictions_data['all_metrics'][selected_crypto]) if has_predictions else None)
csv_data = data.to_csv()

st.sidebar.download_button(
    label="📄 PDF Report",
    data=pdf_buffer,
    file_name=f"{crypto_name}_forecast_{datetime.now().strftime('%Y%m%d')}.pdf",
    mime="application/pdf",
    key="pdf_dl"
)

st.sidebar.download_button(
    label="📊 CSV Data",
    data=csv_data,
    file_name=f"{crypto_name}_data_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    key="csv_dl"
)

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
st.success(f"{theme_emoji} Dashboard v3.4 - {crypto_name} | Warm Sepia Theme")
