
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
# THEME TOGGLE
# ============================================================================

if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Custom CSS for Dark/Light Theme
if st.session_state.theme == 'dark':
    st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        .main-header {
            font-size: 3rem;
            color: #F7931A;
            text-align: center;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #1e2130;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .stSelectbox {
            background-color: #1e2130;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

# Title
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown('<p class="main-header">📈 Cryptocurrency Market Forecasting Dashboard</p>', unsafe_allow_html=True)
with col2:
    theme_btn = st.button("🌓 Toggle Theme", on_click=toggle_theme)

st.markdown("### 🤖 Time Series Analysis with ARIMA, SARIMA, Prophet & LSTM")
st.markdown("---")

# ============================================================================
# EXPANDED CRYPTOCURRENCY LIST
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

# Load data
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

# Load additional coins dynamically
@st.cache_data(ttl=3600)
def load_live_data(ticker):
    """Load live data for additional cryptocurrencies"""
    try:
        data = yf.download(ticker, period='2y', progress=False)
        if not data.empty:
            # Flatten columns if multi-level
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data
        return None
    except:
        return None

data_dict, predictions_data, train_data, test_data = load_data()

# Sidebar
st.sidebar.title("⚙️ Dashboard Controls")

# Crypto selection with expanded list
crypto_name = st.sidebar.selectbox(
    "Select Cryptocurrency",
    options=list(CRYPTO_LIST.keys()),
    index=0
)
selected_crypto = CRYPTO_LIST[crypto_name]

# Load data for selected crypto
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

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")

if has_predictions:
    metrics_df = pd.DataFrame(predictions_data['all_metrics'][selected_crypto])
    best_model = metrics_df.sort_values('RMSE').iloc[0]
    
    st.sidebar.metric("Best Model", best_model['Model'])
    st.sidebar.metric("RMSE", f"${best_model['RMSE']:,.2f}")
    st.sidebar.metric("R² Score", f"{best_model['R² Score']:.4f}")
    st.sidebar.metric("MAPE", f"{best_model['MAPE (%)']:.2f}%")

# ============================================================================
# EXPORT REPORT FUNCTION
# ============================================================================

def generate_pdf_report(crypto_name, data, metrics_df=None):
    """Generate PDF report with predictions and analysis"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph(f"<b>Cryptocurrency Forecast Report: {crypto_name}</b>", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    # Date
    date_text = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
    elements.append(date_text)
    elements.append(Spacer(1, 12))
    
    # Current Stats
    current_price = data['Close'].iloc[-1]
    stats_text = f"""
    <b>Current Market Status:</b><br/>
    Price: ${current_price:,.2f}<br/>
    7-Day MA: ${data['MA7'].iloc[-1]:,.2f}<br/>
    30-Day MA: ${data['MA30'].iloc[-1]:,.2f}<br/>
    RSI: {data['RSI'].iloc[-1]:.2f}<br/>
    Volatility: {data['Volatility'].iloc[-1]:.2f}%
    """
    elements.append(Paragraph(stats_text, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Model Performance Table
    if metrics_df is not None:
        elements.append(Paragraph("<b>Model Performance Comparison:</b>", styles['Heading2']))
        elements.append(Spacer(1, 12))
        
        table_data = [['Model', 'RMSE', 'R² Score', 'MAPE']]
        for _, row in metrics_df.iterrows():
            table_data.append([
                row['Model'],
                f"${row['RMSE']:,.2f}",
                f"{row['R² Score']:.4f}",
                f"{row['MAPE (%)']:.2f}%"
            ])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Export button in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export Report")

if st.sidebar.button("📄 Download PDF Report"):
    with st.spinner("Generating report..."):
        metrics_df = pd.DataFrame(predictions_data['all_metrics'][selected_crypto]) if has_predictions else None
        pdf_buffer = generate_pdf_report(crypto_name, data, metrics_df)
        
        st.sidebar.download_button(
            label="📥 Download PDF",
            data=pdf_buffer,
            file_name=f"{crypto_name}_forecast_report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

# Export CSV
if st.sidebar.button("📊 Download Data (CSV)"):
    csv = data.to_csv()
    st.sidebar.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{crypto_name}_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Main dashboard continues with rest of the original code...
# [Rest of your original dashboard code here - price analysis, charts, etc.]

# Current market status
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

st.success(f"✅ Dashboard v2.0 - {crypto_name} | Theme: {st.session_state.theme.capitalize()}")
