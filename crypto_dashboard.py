
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crypto Market Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 2rem;
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
st.markdown('<p class="main-header">📈 Cryptocurrency Market Forecasting Dashboard</p>', unsafe_allow_html=True)
st.markdown("### 🤖 Time Series Analysis with ARIMA, SARIMA, Prophet & LSTM")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    btc_data = pd.read_csv('saved_data/BTC_USD_data.csv', index_col=0, parse_dates=True)
    eth_data = pd.read_csv('saved_data/ETH_USD_data.csv', index_col=0, parse_dates=True)
    predictions_data = joblib.load('saved_data/predictions_forecasts.pkl')
    train_data = joblib.load('saved_data/train_data.pkl')
    test_data = joblib.load('saved_data/test_data.pkl')
    
    return {
        'BTC-USD': btc_data,
        'ETH-USD': eth_data
    }, predictions_data, train_data, test_data

data_dict, predictions_data, train_data, test_data = load_data()

# Sidebar
st.sidebar.title("⚙️ Dashboard Controls")
selected_crypto = st.sidebar.selectbox(
    "Select Cryptocurrency",
    options=list(data_dict.keys()),
    index=0
)

selected_model = st.sidebar.selectbox(
    "Select Forecasting Model",
    options=['LSTM', 'ARIMA', 'SARIMA', 'Prophet', 'All Models'],
    index=0
)

show_technical = st.sidebar.checkbox("Show Technical Indicators", value=True)
show_forecast = st.sidebar.checkbox("Show Future Forecast", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")

# Get metrics for selected crypto
metrics_df = pd.DataFrame(predictions_data['all_metrics'][selected_crypto])
best_model = metrics_df.sort_values('RMSE').iloc[0]

st.sidebar.metric("Best Model", best_model['Model'])
st.sidebar.metric("RMSE", f"${best_model['RMSE']:,.2f}")
st.sidebar.metric("R² Score", f"{best_model['R² Score']:.4f}")
st.sidebar.metric("MAPE", f"{best_model['MAPE (%)']:.2f}%")

# Main dashboard
data = data_dict[selected_crypto]

# Current market status
col1, col2, col3, col4, col5 = st.columns(5)

current_price = data['Close'].iloc[-1]
price_change_1d = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
current_rsi = data['RSI'].iloc[-1]
current_vol = data['Volatility'].iloc[-1]
current_volume = data['Volume'].iloc[-1]

with col1:
    st.metric("💰 Current Price", f"${current_price:,.2f}", f"{price_change_1d:+.2f}%")

with col2:
    st.metric("📊 7-Day MA", f"${data['MA7'].iloc[-1]:,.2f}")

with col3:
    st.metric("📈 RSI", f"{current_rsi:.2f}", 
              "Overbought" if current_rsi > 70 else ("Oversold" if current_rsi < 30 else "Neutral"))

with col4:
    st.metric("🌊 Volatility (30D)", f"{current_vol:.2f}%")

with col5:
    st.metric("💹 Volume", f"{current_volume/1e9:.2f}B")

st.markdown("---")

# Price chart with predictions
st.markdown("## 📈 Price Analysis & Predictions")

fig = go.Figure()

# Historical prices
fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    name='Historical Price',
    line=dict(color='black', width=2)
))

# Add moving averages if selected
if show_technical:
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MA7'],
        name='MA7', line=dict(color='green', width=1, dash='dash'), opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MA30'],
        name='MA30', line=dict(color='blue', width=1, dash='dash'), opacity=0.6
    ))

# Add test predictions
test_dates = test_data[selected_crypto].index

if selected_model == 'ARIMA' or selected_model == 'All Models':
    if predictions_data['arima_predictions'][selected_crypto] is not None:
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=predictions_data['arima_predictions'][selected_crypto],
            name='ARIMA Predictions',
            line=dict(color='blue', width=2, dash='dot')
        ))

if selected_model == 'SARIMA' or selected_model == 'All Models':
    if predictions_data['sarima_predictions'][selected_crypto] is not None:
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=predictions_data['sarima_predictions'][selected_crypto],
            name='SARIMA Predictions',
            line=dict(color='green', width=2, dash='dot')
        ))

if selected_model == 'Prophet' or selected_model == 'All Models':
    if predictions_data['prophet_predictions'][selected_crypto] is not None:
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=predictions_data['prophet_predictions'][selected_crypto]['yhat'],
            name='Prophet Predictions',
            line=dict(color='purple', width=2, dash='dot')
        ))

if selected_model == 'LSTM' or selected_model == 'All Models':
    if predictions_data['lstm_predictions'][selected_crypto] is not None:
        lstm_dates = test_dates[60:]
        fig.add_trace(go.Scatter(
            x=lstm_dates,
            y=predictions_data['lstm_predictions'][selected_crypto][:len(lstm_dates)],
            name='LSTM Predictions',
            line=dict(color='red', width=2)
        ))

# Add future forecast
if show_forecast:
    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=30, freq='D')
    
    if selected_model == 'LSTM':
        forecast = predictions_data['lstm_forecasts'][selected_crypto]
        fig.add_trace(go.Scatter(
            x=future_dates, y=forecast,
            name='30-Day Forecast (LSTM)',
            line=dict(color='red', width=3, dash='dash')
        ))
    elif selected_model == 'ARIMA':
        forecast = predictions_data['arima_forecasts'][selected_crypto]
        fig.add_trace(go.Scatter(
            x=future_dates, y=forecast,
            name='30-Day Forecast (ARIMA)',
            line=dict(color='blue', width=3, dash='dash')
        ))

fig.update_layout(
    title=f"{selected_crypto} - Price Analysis & Forecasting",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    height=600,
    hovermode='x unified',
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)

# Technical Indicators
if show_technical:
    st.markdown("## 🔧 Technical Indicators")
    
    tab1, tab2, tab3 = st.tabs(["RSI & MACD", "Bollinger Bands", "Volume Analysis"])
    
    with tab1:
        fig = make_subplots(rows=2, cols=1, row_heights=[0.5, 0.5],
                           subplot_titles=('RSI (Relative Strength Index)', 'MACD'))
        
        # RSI
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI',
                                line=dict(color='orange', width=2)), row=1, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD',
                                line=dict(color='blue', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal',
                                line=dict(color='red', width=2)), row=2, col=1)
        
        fig.update_layout(height=700, showlegend=True, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close',
                                line=dict(color='black', width=2)))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name='Upper Band',
                                line=dict(color='red', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name='Lower Band',
                                line=dict(color='green', width=1, dash='dash'),
                                fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
        
        fig.update_layout(title="Bollinger Bands", height=500, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume',
                            marker_color='steelblue', opacity=0.6))
        
        fig.update_layout(title="Trading Volume", height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# Model Comparison
st.markdown("## 🏆 Model Performance Comparison")

metrics_df = pd.DataFrame(predictions_data['all_metrics'][selected_crypto])
metrics_df = metrics_df.sort_values('RMSE')

col1, col2 = st.columns(2)

with col1:
    st.dataframe(metrics_df.style.highlight_min(subset=['RMSE', 'MAPE (%)']).highlight_max(subset=['R² Score']),
                use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics_df['Model'],
        y=metrics_df['RMSE'],
        marker_color=['red', 'green', 'blue', 'orange']
    ))
    fig.update_layout(title="RMSE Comparison (Lower is Better)", 
                     xaxis_title="Model", yaxis_title="RMSE",
                     height=400, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### 📝 About This Dashboard")
st.info("""
**Time Series Cryptocurrency Market Analysis Dashboard**

This dashboard provides comprehensive analysis and forecasting for cryptocurrency markets using:
- **ARIMA**: Classical statistical forecasting
- **SARIMA**: Seasonal ARIMA with weekly patterns
- **Prophet**: Facebook's robust forecasting with trend & seasonality
- **LSTM**: Deep learning neural network for complex patterns

**Best Model**: LSTM consistently outperforms with 91-97% accuracy (R²).
""")
