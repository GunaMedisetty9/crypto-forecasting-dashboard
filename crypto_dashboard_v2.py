
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

# ============================================================================
# CSS
# ============================================================================

if st.session_state.theme == 'dark':
    st.markdown("""
    <style>
        * {transition: background-color 0.5s ease, color 0.5s ease, border-color 0.5s ease !important;}
        .stApp, .stApp > header, [data-testid="stHeader"] {background-color: #0e1117 !important;}
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {background-color: #1e2130 !important;}
        [data-testid="stSidebar"] * {color: #D4C4A8 !important;}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color: #C9B99B !important;}
        [data-testid="stSidebar"] label {color: #B8956A !important;}
        .main-header {font-size: 2rem !important; color: #C9B99B !important; text-align: center; font-weight: 700 !important; margin-bottom: 0.5rem !important;}
        .sub-header {font-size: 1.1rem !important; color: #A67C52 !important; text-align: center; font-weight: 400 !important;}
        .stMarkdown {color: #D4C4A8 !important;}
        [data-testid="stMetricValue"] {color: #D4C4A8 !important; font-size: 1.6rem !important; font-weight: bold !important;}
        [data-testid="stMetricLabel"] {color: #B8956A !important;}
        [data-testid="stMetricDelta"] {color: #B8B76D !important;}
        [data-testid="column"]:last-child {position: fixed !important; top: 0.8rem !important; right: 0.8rem !important; z-index: 999999 !important; width: 60px !important;}
        [data-testid="column"]:last-child .stButton > button {
            background: linear-gradient(135deg, #A67C52 0%, #8B7355 100%) !important;
            color: #0e1117 !important; border: 2px solid #C9B99B !important; padding: 0 !important; border-radius: 50% !important;
            font-size: 1.8rem !important; width: 60px !important; height: 60px !important;
            display: flex !important; align-items: center !important; justify-content: center !important;
            box-shadow: 0 4px 20px rgba(166, 124, 82, 0.4) !important; transition: all 0.3s ease !important; margin: 0 !important;
        }
        [data-testid="column"]:last-child .stButton > button:hover {
            transform: scale(1.1) rotate(15deg) !important; box-shadow: 0 6px 30px rgba(166, 124, 82, 0.6) !important;
        }
        .stSelectbox > div > div {background-color: #262730 !important; color: #D4C4A8 !important; border: 1px solid #8B7355 !important;}
        .stSelectbox label {color: #B8956A !important;}
        .stCheckbox > label {color: #D4C4A8 !important;}
        .stDownloadButton > button {
            background-color: transparent !important; color: #D4C4A8 !important; border: none !important;
            padding: 0.5rem 0.8rem !important; border-radius: 6px !important; font-weight: 500 !important;
            width: 100% !important; text-align: left !important;
        }
        .stDownloadButton > button:hover {
            background-color: rgba(166, 124, 82, 0.15) !important; transform: translateX(5px) !important; color: #C9B99B !important;
        }
        .stSuccess {background-color: rgba(184, 183, 109, 0.15) !important; color: #B8B76D !important; border-left: 4px solid #B8B76D !important;}
        .stTabs [data-baseweb="tab-list"] {background-color: #1e2130 !important;}
        .stTabs [data-baseweb="tab"] {color: #8B7355 !important;}
        .stTabs [aria-selected="true"] {color: #C9B99B !important; border-bottom-color: #A67C52 !important;}
        hr {border-color: #8B7355 !important;}
        #MainMenu, footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        * {transition: background-color 0.5s ease, color 0.5s ease !important;}
        .main-header {font-size: 2rem !important; color: #1f77b4 !important; text-align: center; font-weight: 700 !important;}
        .sub-header {font-size: 1.1rem !important; color: #666 !important; text-align: center;}
        [data-testid="column"]:last-child {position: fixed !important; top: 0.8rem !important; right: 0.8rem !important; z-index: 999999 !important; width: 60px !important;}
        [data-testid="column"]:last-child .stButton > button {
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important; color: #1f2937 !important; border: none !important;
            padding: 0 !important; border-radius: 50% !important; font-size: 1.8rem !important; width: 60px !important; height: 60px !important;
            display: flex !important; align-items: center !important; justify-content: center !important;
            box-shadow: 0 4px 20px rgba(245, 158, 11, 0.4) !important; transition: all 0.3s ease !important;
        }
        [data-testid="column"]:last-child .stButton > button:hover {transform: scale(1.1) rotate(15deg) !important;}
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
# DATA LOADING
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
            
            # Calculate technical indicators
            data['MA7'] = data['Close'].rolling(window=7).mean()
            data['MA30'] = data['Close'].rolling(window=30).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            data['MA200'] = data['Close'].rolling(window=200).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            data['BB_Std'] = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
            data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
            
            # Volatility
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=30).std() * np.sqrt(252) * 100
            
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
st.success(f"{theme_emoji} Dashboard v4.1 - {crypto_name} | {selected_model}")

# ============================================================================
# PRICE CHART WITH PREDICTIONS
# ============================================================================

st.markdown("## 📈 Price Analysis & Predictions")

chart_template = 'plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
bg_color = '#0e1117' if st.session_state.theme == 'dark' else '#ffffff'
text_color = '#D4C4A8' if st.session_state.theme == 'dark' else '#262730'

fig = go.Figure()

# Historical prices
fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    name='Historical Price',
    line=dict(color='#C9B99B', width=2.5)
))

# Technical indicators
if show_technical:
    if 'MA7' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MA7'], mode='lines', name='MA7',
            line=dict(color='#A67C52', width=1.5, dash='dot')
        ))
    
    if 'MA30' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MA30'], mode='lines', name='MA30',
            line=dict(color='#8B7355', width=1.5, dash='dash')
        ))

# Predictions
if show_forecast and has_predictions:
    try:
        if selected_model == 'All Models':
            model_colors = {
                'lstm': '#B8B76D',
                'arima': '#A67C52',
                'sarima': '#8B7355',
                'prophet': '#C9B99B'
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
                            line=dict(color=model_colors[model], width=2, dash='dot')
                        ))
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
                        line=dict(color='#B8B76D', width=2.5, dash='dot')
                    ))
    except Exception as e:
        st.warning(f"Could not load predictions: {str(e)}")

fig.update_layout(
    template=chart_template,
    plot_bgcolor=bg_color,
    paper_bgcolor=bg_color,
    font=dict(color=text_color),
    height=550,
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TECHNICAL INDICATORS WITH REAL CHARTS
# ============================================================================

if show_technical:
    st.markdown("### 📊 Technical Indicators")
    
    tab1, tab2, tab3 = st.tabs(["📈 RSI", "📉 MACD", "🔔 Bollinger Bands"])
    
    with tab1:
        st.markdown("**RSI (Relative Strength Index)** - Momentum oscillator measuring speed and magnitude of price changes")
        
        # RSI Chart (FIXED LINE BELOW)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=data.index[-200:],
            y=data['RSI'].iloc[-200:],
            mode='lines',
            name='RSI',
            line=dict(color='#C9B99B', width=2)
        ))
        
        # Overbought/Oversold lines
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#A67C52", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#8B7355", annotation_text="Oversold (30)")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="#666", annotation_text="Neutral")
        
        fig_rsi.update_layout(
            template=chart_template,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            height=400,
            xaxis_title="Date",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100])  # FIXED: Added proper range value
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # RSI Stats
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
        
        # MACD Chart
        fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                 vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Price
        fig_macd.add_trace(go.Scatter(
            x=data.index[-200:], y=data['Close'].iloc[-200:],
            mode='lines', name='Price', line=dict(color='#C9B99B', width=2)
        ), row=1, col=1)
        
        # MACD Line
        fig_macd.add_trace(go.Scatter(
            x=data.index[-200:], y=data['MACD'].iloc[-200:],
            mode='lines', name='MACD', line=dict(color='#A67C52', width=2)
        ), row=2, col=1)
        
        # Signal Line
        fig_macd.add_trace(go.Scatter(
            x=data.index[-200:], y=data['MACD_Signal'].iloc[-200:],
            mode='lines', name='Signal', line=dict(color='#8B7355', width=2)
        ), row=2, col=1)
        
        # Histogram
        fig_macd.add_trace(go.Bar(
            x=data.index[-200:], y=data['MACD_Hist'].iloc[-200:],
            name='Histogram', marker_color='#B8B76D', opacity=0.5
        ), row=2, col=1)
        
        fig_macd.update_layout(
            template=chart_template,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        st.plotly_chart(fig_macd, use_container_width=True)
    
    with tab3:
        st.markdown("**Bollinger Bands** - Volatility bands placed above and below a moving average")
        
        # Bollinger Bands Chart
        fig_bb = go.Figure()
        
        # Upper Band
        fig_bb.add_trace(go.Scatter(
            x=data.index[-200:], y=data['BB_Upper'].iloc[-200:],
            mode='lines', name='Upper Band', line=dict(color='#A67C52', width=1, dash='dash')
        ))
        
        # Middle Band (SMA)
        fig_bb.add_trace(go.Scatter(
            x=data.index[-200:], y=data['BB_Middle'].iloc[-200:],
            mode='lines', name='SMA (20)', line=dict(color='#C9B99B', width=2)
        ))
        
        # Lower Band
        fig_bb.add_trace(go.Scatter(
            x=data.index[-200:], y=data['BB_Lower'].iloc[-200:],
            mode='lines', name='Lower Band', line=dict(color='#8B7355', width=1, dash='dash')
        ))
        
        # Price
        fig_bb.add_trace(go.Scatter(
            x=data.index[-200:], y=data['Close'].iloc[-200:],
            mode='lines', name='Close Price', line=dict(color='#B8B76D', width=2.5)
        ))
        
        fig_bb.update_layout(
            template=chart_template,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            height=450,
            xaxis_title="Date",
            yaxis_title="Price (USD)"
        )
        st.plotly_chart(fig_bb, use_container_width=True)
        
        # Bollinger Band Width
        data['BB_Width'] = ((data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']) * 100
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current BB Width", f"{data['BB_Width'].iloc[-1]:.2f}%")
        with col2:
            st.metric("30-Day Avg Width", f"{data['BB_Width'].iloc[-30:].mean():.2f}%")

# ============================================================================
# ADDITIONAL VISUALIZATIONS
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Additional Analysis")

col1, col2 = st.columns(2)

with col1:
    # Volume Chart
    st.markdown("### 📈 Volume Analysis")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=data.index[-60:],
        y=data['Volume'].iloc[-60:],
        name='Volume',
        marker_color='#A67C52'
    ))
    fig_vol.update_layout(
        template=chart_template,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        height=350,
        xaxis_title="Date",
        yaxis_title="Volume"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with col2:
    # Returns Distribution
    st.markdown("### 📊 Returns Distribution")
    returns = data['Returns'].dropna() * 100
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=returns[-252:],
        nbinsx=50,
        name='Daily Returns',
        marker_color='#C9B99B'
    ))
    fig_dist.update_layout(
        template=chart_template,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        height=350,
        xaxis_title="Daily Returns (%)",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# Model Comparison Table
if has_predictions:
    st.markdown("---")
    st.markdown("### 🏆 Model Performance Comparison")
    
    comparison_df = pd.DataFrame(predictions_data['all_metrics'][selected_crypto])
    st.dataframe(comparison_df, use_container_width=True)

# Summary Statistics
st.markdown("---")
st.markdown("### 📋 Summary Statistics")

summary_data = {
    'Metric': ['Current Price', 'All-Time High', 'All-Time Low', 'Avg Volume (30D)', 
               'Volatility (30D)', 'Current RSI'],
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
