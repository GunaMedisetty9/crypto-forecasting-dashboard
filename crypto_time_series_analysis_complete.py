
# ============================================================================
# TIME SERIES CRYPTOCURRENCY MARKET ANALYSIS
# Complete End-to-End Project in Google Colab
# ============================================================================

# ============================================================================
# PHASE 1: INSTALLATION & IMPORTS
# ============================================================================

# Install required libraries
!pip install yfinance prophet pmdarima ta-lib-python plotly streamlit pyngrok -q

import warnings
warnings.filterwarnings('ignore')

# Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Data Collection
import yfinance as yf
from datetime import datetime, timedelta

# Statistical & Time Series
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# Machine Learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Utilities
import pickle
import json
from itertools import product

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ All libraries imported successfully!")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Pandas Version: {pd.__version__}")

# ============================================================================
# PHASE 2: DATA COLLECTION
# ============================================================================

class CryptoDataCollector:
    """Collects OHLCV data for cryptocurrencies"""

    def __init__(self, symbols=['BTC-USD', 'ETH-USD'], start_date='2020-01-01'):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.data = {}

    def fetch_data(self):
        """Fetch historical cryptocurrency data"""
        print("\n" + "="*60)
        print("DATA COLLECTION PHASE")
        print("="*60)

        for symbol in self.symbols:
            print(f"\nFetching {symbol} data from {self.start_date} to {self.end_date}...")
            try:
                df = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                if not df.empty:
                    self.data[symbol] = df
                    print(f"✓ {symbol}: {len(df)} records collected")
                    print(f"  Date Range: {df.index[0].date()} to {df.index[-1].date()}")
                else:
                    print(f"✗ No data available for {symbol}")
            except Exception as e:
                print(f"✗ Error fetching {symbol}: {str(e)}")

        return self.data

    def get_sentiment_proxy(self, symbol):
        """Create sentiment proxy using volatility and volume"""
        df = self.data[symbol].copy()

        # Volatility as sentiment proxy
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=7).std()

        # Volume-based sentiment (normalized)
        df['Volume_Sentiment'] = (df['Volume'] - df['Volume'].rolling(30).mean()) / df['Volume'].rolling(30).std()

        # Price momentum sentiment
        df['Price_Momentum'] = df['Close'].pct_change(periods=7)

        # Composite sentiment score
        df['Sentiment_Score'] = (
            df['Volume_Sentiment'].fillna(0) * 0.4 + 
            df['Price_Momentum'].fillna(0) * 100 * 0.6
        )

        return df

# Initialize and fetch data
collector = CryptoDataCollector(symbols=['BTC-USD', 'ETH-USD'], start_date='2020-01-01')
crypto_data = collector.fetch_data()

# Select primary cryptocurrency for analysis
PRIMARY_CRYPTO = 'BTC-USD'
df_raw = crypto_data[PRIMARY_CRYPTO].copy()

print(f"\n✓ Primary cryptocurrency for analysis: {PRIMARY_CRYPTO}")
print(f"✓ Total records: {len(df_raw)}")
print(f"\nData Preview:")
print(df_raw.head())

# ============================================================================
# PHASE 3: DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================================

class CryptoPreprocessor:
    """Preprocessing and feature engineering for crypto data"""

    def __init__(self, df):
        self.df = df.copy()

    def handle_missing_values(self):
        """Handle missing values"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING PHASE")
        print("="*60)

        print(f"\nMissing values before cleaning:")
        print(self.df.isnull().sum())

        # Forward fill then backward fill
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(method='bfill', inplace=True)

        print(f"\n✓ Missing values after cleaning:")
        print(self.df.isnull().sum())

        return self

    def engineer_features(self):
        """Create technical indicators and features"""
        print("\n" + "-"*60)
        print("FEATURE ENGINEERING")
        print("-"*60)

        df = self.df

        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Moving Averages
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA30'] = df['Close'].rolling(window=30).mean()
        df['MA90'] = df['Close'].rolling(window=90).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()

        # Exponential Moving Averages
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=30).std()

        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        # Price Range
        df['Daily_Range'] = df['High'] - df['Low']
        df['Daily_Range_Pct'] = (df['High'] - df['Low']) / df['Open']

        # Trend indicators
        df['Price_vs_MA30'] = (df['Close'] - df['MA30']) / df['MA30']
        df['Price_vs_MA200'] = (df['Close'] - df['MA200']) / df['MA200']

        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)

        # Time-based features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Year'] = df.index.year

        self.df = df

        print(f"\n✓ Features engineered: {len(df.columns)} total columns")
        print(f"✓ New technical indicators: MA, EMA, MACD, RSI, Bollinger Bands, Volatility")

        return self

    def clean_data(self):
        """Final cleaning step"""
        # Drop rows with NaN values after feature engineering
        initial_len = len(self.df)
        self.df.dropna(inplace=True)
        final_len = len(self.df)

        print(f"\n✓ Rows removed due to NaN: {initial_len - final_len}")
        print(f"✓ Final dataset size: {final_len} records")

        return self.df

# Preprocess data
preprocessor = CryptoPreprocessor(df_raw)
df_processed = preprocessor.handle_missing_values().engineer_features().clean_data()

print(f"\n✓ Data preprocessing complete!")
print(f"\nProcessed Data Shape: {df_processed.shape}")
print(f"\nSample processed data:")
print(df_processed[['Close', 'MA7', 'MA30', 'RSI', 'MACD', 'Volatility']].tail())

# ============================================================================
# PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

class CryptoEDA:
    """Comprehensive EDA for cryptocurrency data"""

    def __init__(self, df, crypto_name):
        self.df = df
        self.crypto_name = crypto_name

    def plot_price_trends(self):
        """Plot price trends with moving averages"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{self.crypto_name} Price Trends', 'Trading Volume'),
            row_heights=[0.7, 0.3]
        )

        # Price and MAs
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Close'], 
                                 name='Close', line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['MA7'], 
                                 name='MA7', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['MA30'], 
                                 name='MA30', line=dict(color='green', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['MA200'], 
                                 name='MA200', line=dict(color='red', width=1)), row=1, col=1)

        # Volume
        colors = ['red' if row['Close'] < row['Open'] else 'green' for _, row in self.df.iterrows()]
        fig.add_trace(go.Bar(x=self.df.index, y=self.df['Volume'], 
                             name='Volume', marker_color=colors, showlegend=False), row=2, col=1)

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_layout(height=700, title_text=f"{self.crypto_name} Historical Analysis", hovermode='x unified')

        return fig

    def plot_technical_indicators(self):
        """Plot technical indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('RSI', 'MACD', 'Bollinger Bands'),
            vertical_spacing=0.05,
            row_heights=[0.33, 0.33, 0.34]
        )

        # RSI
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['RSI'], name='RSI', 
                                 line=dict(color='purple', width=1)), row=1, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['MACD'], name='MACD', 
                                 line=dict(color='blue', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['MACD_Signal'], name='Signal', 
                                 line=dict(color='orange', width=1)), row=2, col=1)
        fig.add_trace(go.Bar(x=self.df.index, y=self.df['MACD_Hist'], name='Histogram', 
                             marker_color='gray'), row=2, col=1)

        # Bollinger Bands
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Close'], name='Close', 
                                 line=dict(color='blue', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['BB_Upper'], name='Upper BB', 
                                 line=dict(color='red', width=1, dash='dash')), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['BB_Lower'], name='Lower BB', 
                                 line=dict(color='green', width=1, dash='dash')), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['BB_Middle'], name='Middle BB', 
                                 line=dict(color='orange', width=1)), row=3, col=1)

        fig.update_layout(height=900, title_text=f"{self.crypto_name} Technical Indicators", hovermode='x unified')

        return fig

    def statistical_summary(self):
        """Generate statistical summary"""
        summary = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25%', '75%', 'Skewness', 'Kurtosis'],
            'Close': [
                self.df['Close'].mean(),
                self.df['Close'].median(),
                self.df['Close'].std(),
                self.df['Close'].min(),
                self.df['Close'].max(),
                self.df['Close'].quantile(0.25),
                self.df['Close'].quantile(0.75),
                self.df['Close'].skew(),
                self.df['Close'].kurtosis()
            ],
            'Returns': [
                self.df['Returns'].mean(),
                self.df['Returns'].median(),
                self.df['Returns'].std(),
                self.df['Returns'].min(),
                self.df['Returns'].max(),
                self.df['Returns'].quantile(0.25),
                self.df['Returns'].quantile(0.75),
                self.df['Returns'].skew(),
                self.df['Returns'].kurtosis()
            ],
            'Volume': [
                self.df['Volume'].mean(),
                self.df['Volume'].median(),
                self.df['Volume'].std(),
                self.df['Volume'].min(),
                self.df['Volume'].max(),
                self.df['Volume'].quantile(0.25),
                self.df['Volume'].quantile(0.75),
                self.df['Volume'].skew(),
                self.df['Volume'].kurtosis()
            ]
        })
        return summary

    def correlation_analysis(self):
        """Correlation heatmap"""
        features = ['Close', 'Volume', 'MA7', 'MA30', 'RSI', 'MACD', 'Volatility', 'Returns']
        corr_matrix = self.df[features].corr()

        fig = px.imshow(corr_matrix, 
                        labels=dict(color="Correlation"),
                        x=features,
                        y=features,
                        color_continuous_scale='RdBu_r',
                        aspect="auto",
                        title=f"{self.crypto_name} Feature Correlation Matrix")
        fig.update_layout(height=600)

        return fig

    def decompose_series(self):
        """Time series decomposition"""
        decomposition = seasonal_decompose(self.df['Close'], model='multiplicative', period=365)

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.05
        )

        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Close'], mode='lines', name='Original'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)

        fig.update_layout(height=1000, title_text=f"{self.crypto_name} Time Series Decomposition", showlegend=False)

        return fig

    def check_stationarity(self):
        """Augmented Dickey-Fuller test for stationarity"""
        result = adfuller(self.df['Close'].dropna())

        print(f"\nStationarity Test (ADF Test) for {self.crypto_name}:")
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"p-value: {result[1]:.6f}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.3f}")

        if result[1] <= 0.05:
            print("✓ Series is STATIONARY (p-value <= 0.05)")
        else:
            print("✗ Series is NON-STATIONARY (p-value > 0.05)")
            print("  Differencing required for ARIMA/SARIMA models")

        return result

# Perform EDA
eda = CryptoEDA(df_processed, PRIMARY_CRYPTO)

print("\n1. Statistical Summary:")
stats_summary = eda.statistical_summary()
print(stats_summary.to_string(index=False))

print("\n2. Stationarity Check:")
adf_result = eda.check_stationarity()

print("\n✓ EDA Visualizations prepared (will be displayed later)")

# Save figures for later display
fig_price_trends = eda.plot_price_trends()
fig_technical = eda.plot_technical_indicators()
fig_correlation = eda.correlation_analysis()
fig_decomposition = eda.decompose_series()

# ============================================================================
# PHASE 5: TIME SERIES FORECASTING MODELS
# ============================================================================

print("\n" + "="*60)
print("TIME SERIES FORECASTING MODELS")
print("="*60)

# Split data
train_size = int(len(df_processed) * 0.7)
val_size = int(len(df_processed) * 0.15)
test_size = len(df_processed) - train_size - val_size

train_data = df_processed.iloc[:train_size]
val_data = df_processed.iloc[train_size:train_size+val_size]
test_data = df_processed.iloc[train_size+val_size:]

print(f"\nData Split:")
print(f"  Training: {len(train_data)} records ({train_data.index[0].date()} to {train_data.index[-1].date()})")
print(f"  Validation: {len(val_data)} records ({val_data.index[0].date()} to {val_data.index[-1].date()})")
print(f"  Test: {len(test_data)} records ({test_data.index[0].date()} to {test_data.index[-1].date()})")

# ============================================================================
# MODEL 1: ARIMA
# ============================================================================

print("\n" + "-"*60)
print("MODEL 1: ARIMA (AutoRegressive Integrated Moving Average)")
print("-"*60)

class ARIMAModel:
    """ARIMA model implementation"""

    def __init__(self, train, test):
        self.train = train['Close']
        self.test = test['Close']
        self.model = None
        self.predictions = None
        self.best_order = None

    def find_best_order(self, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
        """Grid search for best ARIMA parameters"""
        print("\nSearching for optimal ARIMA parameters...")

        best_aic = np.inf
        best_order = None

        for p in range(p_range[0], p_range[1]+1):
            for d in range(d_range[0], d_range[1]+1):
                for q in range(q_range[0], q_range[1]+1):
                    try:
                        model = ARIMA(self.train, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue

        self.best_order = best_order
        print(f"✓ Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        return best_order

    def train_model(self, order=None):
        """Train ARIMA model"""
        if order is None:
            order = self.best_order if self.best_order else (1, 1, 1)

        print(f"\nTraining ARIMA{order}...")
        self.model = ARIMA(self.train, order=order)
        self.fitted_model = self.model.fit()

        print(f"✓ ARIMA model trained successfully")
        print(f"\nModel Summary:")
        print(f"  AIC: {self.fitted_model.aic:.2f}")
        print(f"  BIC: {self.fitted_model.bic:.2f}")

        return self

    def forecast(self, steps=None):
        """Generate forecasts"""
        if steps is None:
            steps = len(self.test)

        self.predictions = self.fitted_model.forecast(steps=steps)
        print(f"\n✓ Generated {len(self.predictions)} forecasts")

        return self.predictions

# ARIMA implementation
arima_model = ARIMAModel(train_data, test_data)
arima_model.find_best_order(p_range=(0, 3), d_range=(1, 2), q_range=(0, 3))
arima_model.train_model()
arima_predictions = arima_model.forecast()

# ============================================================================
# MODEL 2: SARIMA
# ============================================================================

print("\n" + "-"*60)
print("MODEL 2: SARIMA (Seasonal ARIMA)")
print("-"*60)

class SARIMAModel:
    """SARIMA model implementation"""

    def __init__(self, train, test, seasonal_period=7):
        self.train = train['Close']
        self.test = test['Close']
        self.seasonal_period = seasonal_period
        self.model = None
        self.predictions = None

    def train_model(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        """Train SARIMA model"""
        print(f"\nTraining SARIMA{order}x{seasonal_order}...")

        self.model = SARIMAX(self.train, 
                            order=order, 
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
        self.fitted_model = self.model.fit(disp=False)

        print(f"✓ SARIMA model trained successfully")
        print(f"\nModel Summary:")
        print(f"  AIC: {self.fitted_model.aic:.2f}")
        print(f"  BIC: {self.fitted_model.bic:.2f}")

        return self

    def forecast(self, steps=None):
        """Generate forecasts"""
        if steps is None:
            steps = len(self.test)

        self.predictions = self.fitted_model.forecast(steps=steps)
        print(f"\n✓ Generated {len(self.predictions)} forecasts")

        return self.predictions

# SARIMA implementation
sarima_model = SARIMAModel(train_data, test_data, seasonal_period=7)
sarima_model.train_model(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
sarima_predictions = sarima_model.forecast()

# ============================================================================
# MODEL 3: PROPHET
# ============================================================================

print("\n" + "-"*60)
print("MODEL 3: PROPHET (Facebook Prophet)")
print("-"*60)

class ProphetModel:
    """Facebook Prophet model implementation"""

    def __init__(self, train, test):
        self.train = train[['Close']].reset_index()
        self.train.columns = ['ds', 'y']
        self.test = test[['Close']].reset_index()
        self.test.columns = ['ds', 'y']
        self.model = None
        self.predictions = None

    def train_model(self, changepoint_prior_scale=0.05, seasonality_prior_scale=10):
        """Train Prophet model"""
        print("\nTraining Prophet model...")

        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )

        self.model.fit(self.train)
        print("✓ Prophet model trained successfully")

        return self

    def forecast(self):
        """Generate forecasts"""
        future = self.model.make_future_dataframe(periods=len(self.test))
        forecast = self.model.predict(future)

        # Extract test predictions
        self.predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(len(self.test))
        self.forecast_df = forecast

        print(f"\n✓ Generated {len(self.predictions)} forecasts")

        return self.predictions['yhat'].values

# Prophet implementation
prophet_model = ProphetModel(train_data, test_data)
prophet_model.train_model()
prophet_predictions = prophet_model.forecast()

# ============================================================================
# MODEL 4: LSTM
# ============================================================================

print("\n" + "-"*60)
print("MODEL 4: LSTM (Long Short-Term Memory)")
print("-"*60)

class LSTMModel:
    """LSTM model implementation"""

    def __init__(self, train, val, test, lookback=60):
        self.train = train
        self.val = val
        self.test = test
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.predictions = None

    def prepare_data(self):
        """Prepare data for LSTM"""
        print("\nPreparing data for LSTM...")

        # Scale data
        train_scaled = self.scaler.fit_transform(self.train[['Close']])
        val_scaled = self.scaler.transform(self.val[['Close']])
        test_scaled = self.scaler.transform(self.test[['Close']])

        # Create sequences
        def create_sequences(data, lookback):
            X, y = [], []
            for i in range(lookback, len(data)):
                X.append(data[i-lookback:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        self.X_train, self.y_train = create_sequences(train_scaled, self.lookback)
        self.X_val, self.y_val = create_sequences(val_scaled, self.lookback)
        self.X_test, self.y_test = create_sequences(test_scaled, self.lookback)

        # Reshape for LSTM [samples, time steps, features]
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        self.X_val = self.X_val.self.X_val.shape[1], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)

        print(f"✓ Training sequences: {self.X_train.shape}")
        print(f"✓ Validation sequences: {self.X_val.shape}")
        print(f"✓ Test sequences: {self.X_test.shape}")

        return self

    def build_model(self, units=[50, 50], dropout=0.2):
        """Build LSTM architecture"""
        print("\nBuilding LSTM model...")

        self.model = Sequential()

        # First LSTM layer
        self.model.add(LSTM(units=units[0], return_sequences=True, input_shape=(self.lookback, 1)))
        self.model.add(Dropout(dropout))

        # Second LSTM layer
        self.model.add(LSTM(units=units[1], return_sequences=False))
        self.model.add(Dropout(dropout))

        # Output layer
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')

        print("✓ LSTM model architecture:")
        self.model.summary()

        return self

    def train_model(self, epochs=50, batch_size=32):
        """Train LSTM model"""
        print(f"\nTraining LSTM model for {epochs} epochs...")

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )

        print(f"✓ LSTM model trained (stopped at epoch {len(history.history['loss'])})")
        print(f"  Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")

        self.history = history
        return self

    def forecast(self):
        """Generate forecasts"""
        predictions_scaled = self.model.predict(self.X_test)
        self.predictions = self.scaler.inverse_transform(predictions_scaled).flatten()

        print(f"\n✓ Generated {len(self.predictions)} forecasts")

        return self.predictions

# LSTM implementation
lstm_model = LSTMModel(train_data, val_data, test_data, lookback=60)
lstm_model.prepare_data()
lstm_model.build_model(units=[100, 50], dropout=0.2)
lstm_model.train_model(epochs=50, batch_size=32)
lstm_predictions = lstm_model.forecast()

# Align predictions with test data
test_actual = test_data['Close'].values
arima_pred_aligned = arima_predictions[:len(test_actual)]
sarima_pred_aligned = sarima_predictions[:len(test_actual)]
prophet_pred_aligned = prophet_predictions[:len(test_actual)]

# LSTM predictions are shorter due to lookback
lstm_test_actual = test_data.iloc[lstm_model.lookback:]['Close'].values
lstm_pred_aligned = lstm_predictions

print("\n✓ All models trained successfully!")

# ============================================================================
# PHASE 6: MODEL EVALUATION
# ============================================================================

print("\n" + "="*60)
print("MODEL EVALUATION & COMPARISON")
print("="*60)

def calculate_metrics(y_true, y_pred, model_name):
    """Calculate regression metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'Model': model_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R² Score': r2,
        'MAPE (%)': mape
    }

# Calculate metrics for all models
metrics_list = []

metrics_list.append(calculate_metrics(test_actual, arima_pred_aligned, 'ARIMA'))
metrics_list.append(calculate_metrics(test_actual, sarima_pred_aligned, 'SARIMA'))
metrics_list.append(calculate_metrics(test_actual, prophet_pred_aligned, 'Prophet'))
metrics_list.append(calculate_metrics(lstm_test_actual, lstm_pred_aligned, 'LSTM'))

# Create comparison DataFrame
metrics_df = pd.DataFrame(metrics_list)

print("\nModel Performance Comparison:")
print("="*80)
print(metrics_df.to_string(index=False))
print("="*80)

# Find best model
best_model_rmse = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']
best_model_r2 = metrics_df.loc[metrics_df['R² Score'].idxmax(), 'Model']

print(f"\n✓ Best Model (RMSE): {best_model_rmse}")
print(f"✓ Best Model (R² Score): {best_model_r2}")

# ============================================================================
# PHASE 7: ADVANCED VISUALIZATIONS
# ============================================================================

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Prediction comparison plot
fig_predictions = go.Figure()

# Test dates
test_dates = test_data.index
lstm_test_dates = test_data.iloc[lstm_model.lookback:].index

# Actual values
fig_predictions.add_trace(go.Scatter(
    x=test_dates, y=test_actual,
    mode='lines', name='Actual',
    line=dict(color='black', width=2)
))

# ARIMA predictions
fig_predictions.add_trace(go.Scatter(
    x=test_dates, y=arima_pred_aligned,
    mode='lines', name='ARIMA',
    line=dict(color='blue', width=1.5)
))

# SARIMA predictions
fig_predictions.add_trace(go.Scatter(
    x=test_dates, y=sarima_pred_aligned,
    mode='lines', name='SARIMA',
    line=dict(color='green', width=1.5)
))

# Prophet predictions
fig_predictions.add_trace(go.Scatter(
    x=test_dates, y=prophet_pred_aligned,
    mode='lines', name='Prophet',
    line=dict(color='orange', width=1.5)
))

# LSTM predictions
fig_predictions.add_trace(go.Scatter(
    x=lstm_test_dates, y=lstm_pred_aligned,
    mode='lines', name='LSTM',
    line=dict(color='red', width=1.5)
))

fig_predictions.update_layout(
    title=f'{PRIMARY_CRYPTO} - Model Predictions Comparison',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    hovermode='x unified',
    height=600
)

# Error distribution plot
fig_errors = make_subplots(
    rows=2, cols=2,
    subplot_titles=('ARIMA Errors', 'SARIMA Errors', 'Prophet Errors', 'LSTM Errors')
)

arima_errors = test_actual - arima_pred_aligned
sarima_errors = test_actual - sarima_pred_aligned
prophet_errors = test_actual - prophet_pred_aligned
lstm_errors = lstm_test_actual - lstm_pred_aligned

fig_errors.add_trace(go.Histogram(x=arima_errors, name='ARIMA', nbinsx=30), row=1, col=1)
fig_errors.add_trace(go.Histogram(x=sarima_errors, name='SARIMA', nbinsx=30), row=1, col=2)
fig_errors.add_trace(go.Histogram(x=prophet_errors, name='Prophet', nbinsx=30), row=2, col=1)
fig_errors.add_trace(go.Histogram(x=lstm_errors, name='LSTM', nbinsx=30), row=2, col=2)

fig_errors.update_layout(
    title='Prediction Error Distributions',
    showlegend=False,
    height=600
)

print("\n✓ All visualizations generated successfully")

# ============================================================================
# PHASE 8: SAVE MODELS AND RESULTS
# ============================================================================

print("\n" + "="*60)
print("SAVING MODELS AND RESULTS")
print("="*60)

# Save models
import joblib

joblib.dump(arima_model.fitted_model, 'arima_model.pkl')
joblib.dump(sarima_model.fitted_model, 'sarima_model.pkl')
joblib.dump(prophet_model.model, 'prophet_model.pkl')
lstm_model.model.save('lstm_model.h5')
joblib.dump(lstm_model.scaler, 'lstm_scaler.pkl')

print("\n✓ Models saved:")
print("  - arima_model.pkl")
print("  - sarima_model.pkl")
print("  - prophet_model.pkl")
print("  - lstm_model.h5")
print("  - lstm_scaler.pkl")

# Save metrics
metrics_df.to_csv('model_metrics.csv', index=False)
print("\n✓ Metrics saved to model_metrics.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'Date': test_dates,
    'Actual': test_actual,
    'ARIMA': arima_pred_aligned,
    'SARIMA': sarima_pred_aligned,
    'Prophet': prophet_pred_aligned
})
predictions_df.to_csv('predictions.csv', index=False)
print("✓ Predictions saved to predictions.csv")

print("\n" + "="*80)
print("PROJECT EXECUTION COMPLETE!")
print("="*80)
