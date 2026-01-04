# app.py — Enhanced Flask with Multi-Algorithm OCHL Prediction + All Features
import os
import time
import random
import json
import threading
import warnings
from functools import wraps
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from flask import Flask, send_from_directory, render_template, jsonify, request, redirect

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
import joblib
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA
warnings.filterwarnings('ignore')

# ---------------- Config ----------------
MODELS_DIR = 'models'
HISTORY_DIR = 'history'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# ---------------- Flask ----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
server = Flask(__name__, template_folder='templates', static_folder='static')

# ---------------- Rate limiter ----------------
class RateLimiter:
    def __init__(self, max_per_minute=30):
        self.max_per_minute = max_per_minute
        self.min_interval = 60.0 / max_per_minute
        self.last_called = {}
        self.lock = threading.Lock()
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            with self.lock:
                current_time = time.time()
                last_time = self.last_called.get(func_name, 0)
                elapsed = current_time - last_time
                if elapsed < self.min_interval:
                    sleep_time = self.min_interval - elapsed
                    time.sleep(sleep_time)
                self.last_called[func_name] = time.time()
            return func(*args, **kwargs)
        return wrapper

rate_limiter = RateLimiter(max_per_minute=25)

# ---------------- Utility & feature functions ----------------
def calculate_rsi(prices, window=14):
    """Calculate RSI with NaN protection"""
    try:
        if len(prices) < window:
            return pd.Series([50] * len(prices), index=prices.index)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except:
        return pd.Series([50] * len(prices), index=prices.index)

def calculate_bollinger_bands(data, window=20):
    """Calculate Bollinger Bands with NaN protection"""
    try:
        if len(data) < window:
            sma = data['Close']
            std = pd.Series([0] * len(data), index=data.index)
        else:
            sma = data['Close'].rolling(window=window, min_periods=1).mean()
            std = data['Close'].rolling(window=window, min_periods=1).std().fillna(0)
        
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return sma, upper_band, lower_band
    except:
        return data['Close'], data['Close'] * 1.1, data['Close'] * 0.9

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD with NaN protection"""
    try:
        exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    except:
        return pd.Series(0, index=data.index), pd.Series(0, index=data.index), pd.Series(0, index=data.index)

def safe_divide(a, b, default=1.0):
    """Safe division with default value"""
    try:
        result = np.divide(a, b, out=np.full_like(a, default, dtype=float), where=b!=0)
        return result
    except:
        return np.full_like(a, default, dtype=float)

def create_advanced_features(data):
    """Create comprehensive features for OCHL prediction with ALL features"""
    try:
        # Create a copy to avoid modifying original
        data = data.copy()
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                print(f"Warning: Missing column {col} in data")
                if col == 'Volume':
                    data[col] = 1000000
                else:
                    data[col] = 100.0
        
        # Convert to numeric and handle NaNs
        for col in data.columns:
            if col != 'Date':
                data[col] = pd.to_numeric(data[col], errors='coerce')
                # Forward fill, then backward fill
                data[col] = data[col].ffill().bfill()
                # If still NaN, fill with appropriate values
                if data[col].isna().any():
                    if col == 'Volume':
                        data[col] = data[col].fillna(1000000)
                    elif col in ['Open', 'High', 'Low', 'Close']:
                        data[col] = data[col].fillna(100.0)
                    else:
                        data[col] = data[col].fillna(0)
        
        print(f"=== FEATURE ENGINEERING STARTED ===")
        print(f"Initial data shape: {data.shape}")
        
        # 1. BASIC PRICE FEATURES
        print("Creating basic price features...")
        data['Return'] = data['Close'].pct_change().fillna(0)
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1).replace(0, 1e-10)).fillna(0)
        data['Volatility_5d'] = data['Return'].rolling(window=5, min_periods=1).std().fillna(0)
        data['Volatility_20d'] = data['Return'].rolling(window=20, min_periods=1).std().fillna(0)
        
        # Price ranges
        data['High_Low_Range'] = safe_divide(data['High'] - data['Low'], data['Close'].replace(0, 1e-10), 0.02)
        data['Open_Close_Range'] = safe_divide(data['Close'] - data['Open'], data['Open'].replace(0, 1e-10), 0)
        
        # 2. MOVING AVERAGES
        print("Creating moving averages...")
        for window in [5, 10, 20, 50]:
            data[f'MA_{window}'] = data['Close'].rolling(window=window, min_periods=1).mean().fillna(data['Close'])
            data[f'MA_Ratio_{window}'] = safe_divide(data['Close'], data[f'MA_{window}'].replace(0, 1e-10), 1.0)
        
        # Moving average crossovers
        data['MA5_Above_MA20'] = (data['MA_5'] > data['MA_20']).astype(int)
        data['Golden_Cross'] = ((data['MA_5'] > data['MA_20']) & 
                               (data['MA_5'].shift(1) <= data['MA_20'].shift(1))).astype(int)
        
        # 3. VOLUME FEATURES
        print("Creating volume features...")
        data['Volume_MA_10'] = data['Volume'].rolling(window=10, min_periods=1).mean().fillna(data['Volume'])
        data['Volume_Ratio'] = safe_divide(data['Volume'], data['Volume_MA_10'].replace(0, 1e-10), 1.0)
        data['Volume_Change'] = data['Volume'].pct_change().fillna(0)
        
        # Volume-Price Trend
        data['Volume_Price_Trend'] = data['Volume'] * data['Return']
        
        # On-Balance Volume (OBV)
        data['OBV'] = (np.sign(data['Return']) * data['Volume']).fillna(0).cumsum()
        
        # 4. TECHNICAL INDICATORS
        print("Creating technical indicators...")
        
        # RSI
        data['RSI_14'] = calculate_rsi(data['Close'], 14)
        data['RSI_7'] = calculate_rsi(data['Close'], 7)
        
        # RSI signals
        data['RSI_Overbought'] = (data['RSI_14'] > 70).astype(int)
        data['RSI_Oversold'] = (data['RSI_14'] < 30).astype(int)
        
        # MACD
        macd, macd_signal, macd_hist = calculate_macd(data)
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal
        data['MACD_Histogram'] = macd_hist
        data['MACD_Signal_Cross'] = (data['MACD'] > data['MACD_Signal']).astype(int)
        
        # Bollinger Bands
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(data)
        data['BB_Middle'] = bb_middle
        data['BB_Upper'] = bb_upper
        data['BB_Lower'] = bb_lower
        data['BB_Width'] = safe_divide(bb_upper - bb_lower, bb_middle.replace(0, 1e-10), 0.1)
        data['BB_Position'] = safe_divide(data['Close'] - bb_lower, (bb_upper - bb_lower).replace(0, 1e-10), 0.5)
        data['BB_Upper_Touch'] = (data['Close'] >= data['BB_Upper'] * 0.99).astype(int)
        data['BB_Lower_Touch'] = (data['Close'] <= data['BB_Lower'] * 1.01).astype(int)
        
        # 5. MOMENTUM & OSCILLATORS
        print("Creating momentum indicators...")
        
        # Momentum
        for window in [5, 10, 20]:
            shifted = data['Close'].shift(window).replace(0, 1e-10)
            data[f'Momentum_{window}'] = safe_divide(data['Close'], shifted, 1.0) - 1
            
            # Rate of Change
            data[f'ROC_{window}'] = data['Close'].pct_change(window) * 100
        
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(14).min()
        high_14 = data['High'].rolling(14).max()
        data['Stochastic_%K'] = 100 * safe_divide(data['Close'] - low_14, high_14 - low_14, 50)
        data['Stochastic_%D'] = data['Stochastic_%K'].rolling(3).mean().fillna(50)
        
        # Williams %R
        data['Williams_%R'] = -100 * safe_divide(high_14 - data['Close'], high_14 - low_14, -50)
        
        # 6. ATR (Average True Range) - NEW
        print("Creating ATR features...")
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift(1))
        low_close = np.abs(data['Low'] - data['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR_14'] = true_range.rolling(14).mean().fillna(0)
        data['ATR_Ratio'] = safe_divide(data['ATR_14'], data['Close'], 0.01)
        
        # 7. SUPPORT/RESISTANCE - NEW
        print("Creating support/resistance features...")
        data['Resistance_20'] = data['High'].rolling(20).max().fillna(data['Close'])
        data['Support_20'] = data['Low'].rolling(20).min().fillna(data['Close'])
        data['Resistance_Distance'] = safe_divide(data['Close'] - data['Resistance_20'], data['Close'], -0.1)
        data['Support_Distance'] = safe_divide(data['Close'] - data['Support_20'], data['Close'], 0.1)
        
        # 8. PRICE PATTERNS
        print("Creating price patterns...")
        prev_close = data['Close'].shift(1).fillna(data['Close'])
        data['Gap_Up'] = ((data['Open'] > prev_close * 1.01) & (data['Open'] > 0)).astype(int)
        data['Gap_Down'] = ((data['Open'] < prev_close * 0.99) & (data['Open'] > 0)).astype(int)
        
        # Inside/Outside days
        data['Inside_Day'] = ((data['High'] < data['High'].shift(1)) & 
                             (data['Low'] > data['Low'].shift(1))).astype(int)
        data['Outside_Day'] = ((data['High'] > data['High'].shift(1)) & 
                              (data['Low'] < data['Low'].shift(1))).astype(int)
        
        # 9. SEASONALITY FEATURES - NEW
        print("Creating seasonality features...")
        if 'Date' in data.columns:
            try:
                dates = pd.to_datetime(data['Date'])
                data['Day_of_Week'] = dates.dt.dayofweek
                data['Month'] = dates.dt.month
                data['Week_of_Year'] = dates.dt.isocalendar().week
                
                # Month effects
                data['Month_End'] = dates.dt.is_month_end.astype(int)
                data['Month_Start'] = dates.dt.is_month_start.astype(int)
                
                # Day of month
                data['Day_of_Month'] = dates.dt.day
                data['Is_15th'] = (data['Day_of_Month'] == 15).astype(int)
            except Exception as e:
                print(f"Warning: Could not create seasonality features: {e}")
        
        # 10. LAGGED FEATURES - NEW
        print("Creating lagged features...")
        for lag in [1, 2, 3, 5, 10]:
            data[f'Return_Lag_{lag}'] = data['Return'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
        
        # Rolling statistics
        data['Return_Std_20'] = data['Return'].rolling(20).std().fillna(0)
        data['Volume_Std_20'] = data['Volume'].rolling(20).std().fillna(0)
        
        # 11. INTERACTION FEATURES - NEW
        print("Creating interaction features...")
        data['Volume_RSI_Interaction'] = data['Volume_Ratio'] * (data['RSI_14'] / 100)
        data['Vol_MA_Signal'] = data['Volatility_5d'] * data['MA5_Above_MA20']
        data['RSI_MACD_Interaction'] = (data['RSI_14'] / 100) * data['MACD']
        
        # 12. BINARY SIGNAL FEATURES
        print("Creating binary signal features...")
        data['Volume_Spike'] = (data['Volume_Ratio'] > 2).astype(int)
        data['High_Volatility'] = (data['Volatility_5d'] > 0.02).astype(int)
        data['Strong_Uptrend'] = ((data['MA_5'] > data['MA_20']) & 
                                  (data['MA_20'] > data['MA_50'])).astype(int)
        data['Strong_Downtrend'] = ((data['MA_5'] < data['MA_20']) & 
                                    (data['MA_20'] < data['MA_50'])).astype(int)
        
        # Handle any remaining NaN values
        print("Handling remaining NaN values...")
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Final check - replace any remaining NaN with appropriate defaults
        for col in data.columns:
            if col != 'Date':
                if data[col].isna().any():
                    if 'Ratio' in col or 'Momentum' in col or 'ROC' in col:
                        data[col] = data[col].fillna(0.0)
                    elif col in ['Return', 'Volatility', 'Volume_Change', 'MACD', 'MACD_Signal', 'MACD_Histogram']:
                        data[col] = data[col].fillna(0.0)
                    elif 'RSI' in col or 'Stochastic' in col or 'Williams' in col:
                        data[col] = data[col].fillna(50.0)
                    elif col == 'Volume':
                        data[col] = data[col].fillna(1000000)
                    elif col in ['Open', 'High', 'Low', 'Close']:
                        data[col] = data[col].fillna(100.0)
                    else:
                        data[col] = data[col].fillna(0.0)
        
        print(f"=== FEATURE ENGINEERING COMPLETE ===")
        print(f"Total features created: {len(data.columns)}")
        print(f"Feature names: {list(data.columns)}")
        
        return data
    except Exception as e:
        print(f"Feature creation error: {e}")
        import traceback
        traceback.print_exc()
        # Return minimal valid dataframe
        return pd.DataFrame({
            'Open': [100.0] * len(data) if hasattr(data, '__len__') else [100.0],
            'High': [101.0] * len(data) if hasattr(data, '__len__') else [101.0],
            'Low': [99.0] * len(data) if hasattr(data, '__len__') else [99.0],
            'Close': [100.0] * len(data) if hasattr(data, '__len__') else [100.0],
            'Volume': [1000000] * len(data) if hasattr(data, '__len__') else [1000000]
        })

def detect_anomalies(data):
    """Detect anomalies in the data using Isolation Forest"""
    try:
        features = ['Return', 'Volatility_5d', 'Volume_Ratio', 'RSI_14', 'High_Low_Range']
        available_features = [f for f in features if f in data.columns]
        
        if len(available_features) < 3:
            return pd.Series([0] * len(data), index=data.index)
        
        X = data[available_features].fillna(0).values
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X)
        
        # Convert to binary (1 = anomaly, 0 = normal)
        anomaly_score = (anomalies == -1).astype(int)
        return pd.Series(anomaly_score, index=data.index)
    except:
        return pd.Series([0] * len(data), index=data.index)

# ---------------- Live data fetching ----------------
@rate_limiter
def get_live_stock_data_enhanced(ticker):
    """Fetch stock data with enhanced error handling"""
    try:
        strategies = [
            {"func": lambda: yf.download(ticker, period="2y", interval="1d", progress=False, timeout=30), "name":"2y"},
            {"func": lambda: yf.Ticker(ticker).history(period="1y", interval="1d"), "name":"ticker.history"},
            {"func": lambda: yf.download(ticker, period="6mo", interval="1d", progress=False, timeout=25), "name":"6mo"},
        ]
        
        for strategy in strategies:
            try:
                hist = strategy['func']()
                if isinstance(hist, pd.DataFrame) and (not hist.empty) and len(hist) > 60:
                    hist = hist.reset_index()
                    
                    # Handle date column
                    if 'Date' in hist.columns:
                        hist['Date'] = pd.to_datetime(hist['Date']).dt.strftime('%Y-%m-%d')
                    elif 'Datetime' in hist.columns:
                        hist['Date'] = pd.to_datetime(hist['Datetime']).dt.strftime('%Y-%m-%d')
                        hist = hist.drop(columns=['Datetime'])
                    
                    # Ensure we have all required columns
                    required = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required:
                        if col not in hist.columns:
                            print(f"Warning: Missing {col} in data for {ticker}")
                            if col == 'Volume':
                                hist[col] = 1000000
                            else:
                                hist[col] = hist.get('Close', 100.0) if 'Close' in hist.columns else 100.0
                    
                    # Convert to numeric and handle NaNs
                    for col in required:
                        hist[col] = pd.to_numeric(hist[col], errors='coerce')
                        hist[col] = hist[col].ffill().bfill()
                        if hist[col].isna().any():
                            if col == 'Volume':
                                hist[col] = hist[col].fillna(1000000)
                            else:
                                hist[col] = hist[col].fillna(100.0)
                    
                    # Get current price
                    if 'Close' in hist.columns and len(hist) > 0:
                        current_price = float(hist['Close'].iloc[-1])
                    else:
                        current_price = 100.0
                    
                    print(f"Successfully fetched {len(hist)} days of data for {ticker}")
                    return hist, current_price, None
                    
            except Exception as e:
                print(f"Strategy {strategy['name']} failed for {ticker}: {e}")
                if "429" in str(e):
                    time.sleep(5)
                continue
        
        # Fallback: generate synthetic data
        print(f"Using fallback data for {ticker}")
        return generate_fallback_data(ticker)
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return generate_fallback_data(ticker)

def generate_fallback_data(ticker, days=500):
    """Generate fallback synthetic data without NaNs"""
    base_prices = {'AAPL':182,'MSFT':407,'GOOGL':172,'AMZN':178,'TSLA':175,'SPY':445}
    base_price = base_prices.get(ticker, 100.0)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.dayofweek < 5]
    
    if len(dates) > days:
        dates = dates[-days:]
    
    # Generate realistic prices
    prices = [base_price]
    for i in range(1, len(dates)):
        # Random walk with some mean reversion
        change = random.gauss(0, 0.015)  # 1.5% daily volatility
        new_price = prices[-1] * (1 + change)
        # Ensure price stays positive and reasonable
        new_price = max(new_price, base_price * 0.1)
        new_price = min(new_price, base_price * 3)
        prices.append(new_price)
    
    # Generate OHLC data
    opens, highs, lows = [], [], []
    for i, close in enumerate(prices):
        open_price = close * random.uniform(0.99, 1.01)
        high = max(open_price, close) * random.uniform(1.001, 1.03)
        low = min(open_price, close) * random.uniform(0.97, 0.999)
        
        opens.append(open_price)
        highs.append(high)
        lows.append(low)
    
    df = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': [random.randint(1000000, 10000000) for _ in range(len(prices))]
    })
    
    print(f"Generated {len(df)} days of fallback data for {ticker}")
    return df, prices[-1], None

# ---------------- OCHL Multi-Algorithm Predictor ----------------
class OCHLPredictor:
    def __init__(self):
        self.models = {}  # {target: {algorithm: model}}
        self.scalers = {}  # {target: scaler}
        self.feature_scaler = RobustScaler()  # Changed to RobustScaler for outlier resistance
        self.feature_columns = []
        self.targets = ['Open', 'Close', 'High', 'Low']
        self.historical_performance = {}
        self.prediction_history = {}
        self.risk_metrics = {}
        self.algorithm_weights = {}  # Store algorithm weights based on performance
        self.last_training_date = None
        self.is_fitted = False
        
    def get_model_path(self, symbol, target, algorithm):
        safe_symbol = "".join([c for c in symbol if c.isalnum() or c in "-_"]).upper()
        safe_target = target.lower()
        safe_algo = algorithm.lower().replace(" ", "_")
        return os.path.join(MODELS_DIR, f"{safe_symbol}_{safe_target}_{safe_algo}.joblib")
    
    def get_history_path(self, symbol):
        safe_symbol = "".join([c for c in symbol if c.isalnum() or c in "-_"]).upper()
        return os.path.join(HISTORY_DIR, f"{safe_symbol}_history.json")
    
    def save_models(self, symbol):
        """Save all trained models"""
        try:
            for target in self.targets:
                if target in self.models:
                    for algo, model in self.models[target].items():
                        if model is not None:
                            model_data = {
                                'model': model,
                                'feature_scaler': self.feature_scaler,   
                                'target_scaler': self.scalers.get(target, StandardScaler()),
                                'features': self.feature_columns,
                                'window_size': 30,
                                'target': target,
                                'algorithm': algo
                            }
                            path = self.get_model_path(symbol, target, algo)
                            joblib.dump(model_data, path)
            
            # Save historical performance
            history_data = {
                'historical_performance': self.historical_performance,
                'prediction_history': self.prediction_history,
                'risk_metrics': self.risk_metrics,
                'algorithm_weights': self.algorithm_weights,
                'last_training_date': self.last_training_date,
                'feature_columns': self.feature_columns
            }
            
            with open(self.get_history_path(symbol), 'w') as f:
                json.dump(history_data, f, default=str)
                
            print(f"Saved models and history for {symbol}")
            return True
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def load_models(self, symbol):
        """Load trained models and history"""
        try:
            loaded_models = {target: {} for target in self.targets}
            loaded_scalers = {}
            
            algorithms = ['linear_regression', 'ridge', 'lasso', 'svr', 'random_forest', 'gradient_boosting', 'arima', 'neural_network']
            
            models_loaded = False
            
            # Track which target we load scalers from (first valid one)
            first_valid_model = None
            
            for target in self.targets:
                for algo in algorithms:
                    path = self.get_model_path(symbol, target, algo)
                    if os.path.exists(path):
                        try:
                            model_data = joblib.load(path)
                            loaded_models[target][algo] = model_data['model']
                            
                            # Load feature scaler (once)
                            if not hasattr(self.feature_scaler, "median_"):
                                self.feature_scaler = model_data.get('feature_scaler', RobustScaler())
                            
                            # Load target scaler for this target
                            if target not in loaded_scalers:
                                loaded_scalers[target] = model_data.get('target_scaler', StandardScaler())
                            
                            # Load feature columns (once)
                            if not self.feature_columns:
                                self.feature_columns = model_data.get('features', [])
                            
                            if first_valid_model is None:
                                first_valid_model = target
                            
                            models_loaded = True
                            print(f"Loaded {target}-{algo} model for {symbol}")
                        except Exception as e:
                            print(f"Error loading {target}-{algo} model: {e}")
                            loaded_models[target][algo] = None
                    else:
                        loaded_models[target][algo] = None
            
            if models_loaded:
                self.models = loaded_models
                self.scalers = loaded_scalers
                
                # Load historical data
                history_path = self.get_history_path(symbol)
                if os.path.exists(history_path):
                    try:
                        with open(history_path, 'r') as f:
                            history_data = json.load(f)
                        
                        self.historical_performance = history_data.get('historical_performance', {})
                        self.prediction_history = history_data.get('prediction_history', {})
                        self.risk_metrics = history_data.get('risk_metrics', {})
                        self.algorithm_weights = history_data.get('algorithm_weights', {})
                        self.last_training_date = history_data.get('last_training_date')
                        self.feature_columns = history_data.get('feature_columns', self.feature_columns)
                        self.is_fitted = True
                        print(f"Loaded history for {symbol}")
                    except Exception as e:
                        print(f"Error loading history: {e}")
                
                print(f"Successfully loaded models for {symbol}")
                return True
            else:
                print(f"No models found for {symbol}")
                return False
                
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def prepare_training_data(self, data):
        """Prepare data for OCHL prediction with improved feature engineering"""
        try:
            print(f"\n{'='*60}")
            print(f"PREPARING TRAINING DATA")
            print(f"{'='*60}")
            print(f"Initial data rows: {len(data)}")
            
            # Create features
            data_with_features = create_advanced_features(data)
            print(f"Total features created: {len(data_with_features.columns)}")
            
            # Ensure we have enough data
            if len(data_with_features) < 100:
                print(f"Warning: Only {len(data_with_features)} rows of data, minimum 100 required")
                return None, None, None
            
            # Feature selection - use all available numeric features except Date
            numeric_cols = [col for col in data_with_features.columns 
                          if col != 'Date' and pd.api.types.is_numeric_dtype(data_with_features[col])]
            
            # Remove target columns from features
            feature_candidates = [col for col in numeric_cols if col not in self.targets]
            
            print(f"Available feature candidates: {len(feature_candidates)}")
            
            # Select top features by correlation with target
            if not self.feature_columns:
                print("Selecting best features by correlation...")
                correlation_scores = {}
                for feature in feature_candidates:
                    try:
                        # Calculate correlation with each target and take max
                        corr_scores = []
                        for target in self.targets:
                            if target in data_with_features.columns:
                                corr = abs(data_with_features[feature].corr(data_with_features[target]))
                                if not pd.isna(corr):
                                    corr_scores.append(corr)
                        
                        if corr_scores:
                            correlation_scores[feature] = max(corr_scores)
                    except:
                        correlation_scores[feature] = 0
                
                # Select top 30 features by correlation
                if correlation_scores:
                    top_features = sorted(correlation_scores.items(), key=lambda x: x[1], reverse=True)[:30]
                    self.feature_columns = [f[0] for f in top_features]
                    print(f"Selected top {len(self.feature_columns)} features by correlation")
                    print(f"Top 5 features: {self.feature_columns[:5]}")
                else:
                    # Fallback to variance
                    if len(feature_candidates) > 30:
                        variances = data_with_features[feature_candidates].var()
                        variances = variances.fillna(0)
                        self.feature_columns = variances.nlargest(30).index.tolist()
                        print(f"Selected top 30 features by variance")
                    else:
                        self.feature_columns = feature_candidates
                        print(f"Using all {len(feature_candidates)} features")
            
            # If still no features, use defaults
            if not self.feature_columns:
                default_features = ['Return', 'Volatility_5d', 'Volume_Ratio', 'RSI_14', 'High_Low_Range', 
                                   'MA_5', 'MA_10', 'MA_20', 'BB_Position', 'Momentum_5',
                                   'MACD', 'ATR_Ratio', 'Volume_Price_Trend', 'Resistance_Distance', 'Support_Distance']
                self.feature_columns = [f for f in default_features if f in data_with_features.columns]
                if not self.feature_columns:
                    self.feature_columns = list(data_with_features.columns[:15])
                print(f"Using default features: {len(self.feature_columns)}")
            
            print(f"Final feature count: {len(self.feature_columns)}")
            
            # Prepare X (features) for each target
            X_data = {}
            y_data = {}
            
            for target in self.targets:
                print(f"\nPreparing data for {target}...")
                
                if target not in data_with_features.columns:
                    print(f"Warning: {target} column not found, using Close as target")
                    data_with_features[target] = data_with_features['Close']
                
                # Ensure target has no NaN values
                data_with_features[target] = data_with_features[target].ffill().bfill().fillna(data_with_features['Close'].mean())
                
                # Prepare sequences (use past 30 days to predict next day)
                X_list = []
                y_list = []
                window_size = 30
                
                if len(data_with_features) < window_size + 20:
                    print(f"Warning: Not enough data for window size {window_size}")
                    continue
                
                # Ensure all feature columns exist
                missing_features = [f for f in self.feature_columns if f not in data_with_features.columns]
                if missing_features:
                    print(f"Warning: Missing features {missing_features[:5]}..., filling with zeros")
                    for f in missing_features:
                        data_with_features[f] = 0
                
                samples_count = 0
                for i in range(window_size, len(data_with_features) - 1):
                    # Features from window
                    features = data_with_features[self.feature_columns].iloc[i-window_size:i]
                    
                    # Check for NaN in features
                    if features.isna().any().any():
                        features = features.fillna(0)
                    
                    features_array = features[self.feature_columns].values
                    
                    # Flatten features
                    features_flat = features_array.flatten()
                    
                    expected_dim = len(self.feature_columns) * window_size
                    if features_flat.shape[0] != expected_dim:
                        continue
                    
                    # Target is next day's value
                    target_value = data_with_features[target].iloc[i+1]
                    
                    # Skip if target is NaN or extreme outlier
                    if pd.isna(target_value):
                        continue
                    
                    # Filter extreme values (more than 5 standard deviations from mean)
                    target_mean = data_with_features[target].mean()
                    target_std = data_with_features[target].std()
                    if target_std > 0 and abs(target_value - target_mean) > 5 * target_std:
                        continue
                    
                    X_list.append(features_flat)
                    y_list.append(target_value)
                    samples_count += 1
                
                if samples_count > 50:
                    X_data[target] = np.array(X_list)
                    y_data[target] = np.array(y_list)
                    print(f"  Prepared {samples_count} samples for {target}")
                else:
                    print(f"  Warning: Insufficient samples for {target} ({samples_count})")
                    X_data[target] = None
                    y_data[target] = None
            
            # Check if we have data for all targets
            valid_targets = [t for t in self.targets if t in X_data and X_data[t] is not None and len(X_data[t]) > 0]
            if len(valid_targets) == 0:
                print("Error: No valid training data for any target")
                return None, None, None
            
            print(f"\nSuccessfully prepared data for targets: {valid_targets}")
            print(f"{'='*60}")
            return X_data, y_data, data_with_features
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def train_algorithm(self, X, y, algorithm, target):
        """Train a specific algorithm with improved regularization"""
        try:
            print(f"    Training {algorithm} for {target}...")
            
            # Check for NaN in inputs
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                print(f"      Warning: NaN found in training data")
                X = np.nan_to_num(X, nan=0.0)
                y_mean = np.nanmean(y) if not np.all(np.isnan(y)) else 0.0
                y = np.nan_to_num(y, nan=y_mean)
            
            if len(X) < 50:
                print(f"      Warning: Insufficient data ({len(X)} samples)")
                return None
            
            # Remove outliers from training data for linear models
            if algorithm in ['linear_regression', 'ridge', 'lasso']:
                # Remove samples where target is extreme
                y_mean, y_std = np.mean(y), np.std(y)
                if y_std > 0:
                    mask = np.abs(y - y_mean) < 3 * y_std
                    X = X[mask]
                    y = y[mask]
                    
                    if len(X) < 20:
                        print(f"      Warning: Too many outliers removed")
                        return None
            
            if algorithm == 'linear_regression':
                model = LinearRegression()
                model.fit(X, y)
                print(f"      Trained Linear Regression with {len(X)} samples")
                return model
                
            elif algorithm == 'ridge':
                model = Ridge(alpha=1.0, random_state=42, max_iter=10000)
                model.fit(X, y)
                print(f"      Trained Ridge Regression with {len(X)} samples")
                return model
                
            elif algorithm == 'lasso':
                model = Lasso(alpha=0.01, random_state=42, max_iter=10000)
                model.fit(X, y)
                print(f"      Trained Lasso Regression with {len(X)} samples")
                return model
                
            elif algorithm == 'svr':
                model = SVR(kernel='rbf', C=1.0, epsilon=0.01, max_iter=10000)
                if len(X) > 1000:
                    idx = np.random.choice(len(X), min(1000, len(X)), replace=False)
                    model.fit(X[idx], y[idx])
                else:
                    model.fit(X, y)
                print(f"      Trained SVR with {len(X)} samples")
                return model
                
            elif algorithm == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y)
                print(f"      Trained Random Forest with {len(X)} samples")
                return model
                
            elif algorithm == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    random_state=42
                )
                model.fit(X, y)
                print(f"      Trained Gradient Boosting with {len(X)} samples")
                return model
                
            elif algorithm == 'neural_network':
                model = MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.01,
                    max_iter=1000,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.2
                )
                model.fit(X, y)
                print(f"      Trained Neural Network with {len(X)} samples")
                return model
                
            elif algorithm == 'arima':
                if len(y) > 100:
                    try:
                        y_clean = np.nan_to_num(y, nan=np.nanmean(y) if not np.all(np.isnan(y)) else 0.0)
                        
                        model = pm.auto_arima(
                            y_clean,
                            start_p=1, start_q=1,
                            max_p=2, max_q=2, m=1,
                            seasonal=False,
                            trace=False,
                            error_action='ignore',
                            suppress_warnings=True,
                            maxiter=30,
                            stepwise=True
                        )
                        print(f"      Trained ARIMA with order {model.order}")
                        return model
                    except Exception as e:
                        print(f"      ARIMA auto failed: {e}")
                        try:
                            model = StatsmodelsARIMA(y_clean, order=(1,1,0))
                            model_fit = model.fit()
                            print(f"      Trained simple ARIMA(1,1,0)")
                            return model_fit
                        except:
                            print(f"      Simple ARIMA also failed")
                            return None
                else:
                    print(f"      Warning: Insufficient data for ARIMA ({len(y)} samples)")
                    return None
                    
            return None
        except Exception as e:
            print(f"      Error training {algorithm}: {e}")
            return None
    
    def train_all_models(self, data, symbol):
        """Train models for all OCHL targets using all algorithms"""
        try:
            print(f"\n{'='*60}")
            print(f"TRAINING MODELS FOR {symbol}")
            print(f"{'='*60}")
            
            X_data, y_data, data_with_features = self.prepare_training_data(data)
            
            if X_data is None or y_data is None:
                print("Error: Could not prepare training data")
                return False, "Insufficient data for training"
            
            algorithms = ['linear_regression', 'ridge', 'lasso', 'svr', 'random_forest', 'gradient_boosting', 'arima', 'neural_network']
            self.models = {target: {} for target in self.targets}
            self.scalers = {}
            self.algorithm_weights = {target: {} for target in self.targets}
            
            print(f"Training {len(algorithms)} algorithms:")
            print(f"Algorithms: {algorithms}")
            
            # Fit feature scaler first with all available data
            all_features = []
            for target in self.targets:
                if target in X_data and X_data[target] is not None:
                    all_features.append(X_data[target])
            
            if all_features:
                all_features_array = np.vstack(all_features)
                self.feature_scaler.fit(all_features_array)
                print(f"Fitted RobustScaler with {all_features_array.shape[0]} samples")
            else:
                print("Warning: No features available for scaling")
            
            targets_trained = 0
            
            for target in self.targets:
                print(f"\n--- Training {target} ---")
                
                if target not in X_data or X_data[target] is None:
                    print(f"  Skipping: No data")
                    continue
                
                X = X_data[target]
                y = y_data[target]
                
                # Check data quality
                if len(X) < 50 or len(y) < 50:
                    print(f"  Skipping: Insufficient samples ({len(X)})")
                    continue
                
                # Scale features
                try:
                    X_scaled = self.feature_scaler.transform(X)
                    print(f"  Scaled features: {X_scaled.shape}")
                except Exception as e:
                    print(f"  Warning: Feature scaling failed: {e}")
                    X_scaled = X
                
                # Scale target
                try:
                    target_scaler = StandardScaler()
                    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
                    self.scalers[target] = target_scaler
                    print(f"  Scaled target: {y_scaled.shape}")
                except Exception as e:
                    print(f"  Warning: Target scaling failed: {e}")
                    y_scaled = y
                    self.scalers[target] = StandardScaler()
                
                # Train each algorithm
                for algo in algorithms:
                    model = self.train_algorithm(X_scaled, y_scaled, algo, target)
                    self.models[target][algo] = model
                
                targets_trained += 1
                print(f"  ✓ Trained all algorithms for {target}")
            
            if targets_trained == 0:
                print("Error: No targets were successfully trained")
                return False, "Failed to train any models"
            
            # Calculate historical performance and weights
            print(f"\n--- Calculating Performance Metrics ---")
            self.calculate_historical_performance(X_data, y_data, data_with_features)
            
            # Calculate risk metrics
            print(f"\n--- Calculating Risk Metrics ---")
            self.calculate_risk_metrics(data_with_features)
            
            # Save prediction history
            self.update_prediction_history(symbol, data_with_features)
            
            self.last_training_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.is_fitted = True
            
            # Save models and history
            self.save_models(symbol)
            
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETE")
            print(f"Targets trained: {targets_trained}/{len(self.targets)}")
            print(f"Algorithms: {len(algorithms)}")
            print(f"Features: {len(self.feature_columns)}")
            print(f"{'='*60}")
            
            return True, f"Trained models for {targets_trained} targets successfully"
            
        except Exception as e:
            print(f"Error training all models: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)
    
    def calculate_historical_performance(self, X_data, y_data, data_with_features):
        """Calculate comprehensive historical performance metrics and weights"""
        try:
            print("\nCalculating historical performance...")
            performance = {}
            
            for target in self.targets:
                if target not in self.models or target not in X_data:
                    continue
                
                X = X_data[target]
                y = y_data[target]
                
                if X is None or y is None or len(X) < 50:
                    continue
                
                print(f"  Evaluating {target}...")
                
                # Use time series split for validation
                tscv = TimeSeriesSplit(n_splits=3)  # Reduced for speed
                fold_scores = {algo: {'rmse': [], 'mae': [], 'r2': [], 'direction': []} 
                              for algo in self.models[target].keys() if self.models[target][algo] is not None}
                
                for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                    if fold >= 3:  # Only 3 folds for speed
                        break
                        
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    if len(X_test) < 5 or len(y_test) < 5:
                        continue
                    
                    # Scale features for this fold
                    try:
                        X_train_scaled = self.feature_scaler.transform(X_train)
                        X_test_scaled = self.feature_scaler.transform(X_test)
                    except:
                        X_train_scaled = X_train
                        X_test_scaled = X_test
                    
                    # Scale target for this fold
                    target_scaler = StandardScaler()
                    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
                    
                    for algo, model in self.models[target].items():
                        if model is None or algo not in fold_scores:
                            continue
                        
                        try:
                            # Train model on training fold
                            if algo == 'arima':
                                if len(y_train_scaled) > 30:
                                    try:
                                        temp_arima = pm.auto_arima(
                                            y_train_scaled,
                                            start_p=1, start_q=1,
                                            max_p=2, max_q=2,
                                            seasonal=False,
                                            trace=False,
                                            error_action='ignore'
                                        )
                                        predictions_scaled = temp_arima.predict(n_periods=len(y_test_scaled))
                                    except:
                                        predictions_scaled = np.zeros_like(y_test_scaled)
                                else:
                                    predictions_scaled = np.zeros_like(y_test_scaled)
                            else:
                                # Clone and retrain model for this fold
                                if algo == 'linear_regression':
                                    fold_model = LinearRegression()
                                elif algo == 'ridge':
                                    fold_model = Ridge(alpha=1.0, random_state=42)
                                elif algo == 'lasso':
                                    fold_model = Lasso(alpha=0.01, random_state=42)
                                elif algo == 'svr':
                                    fold_model = SVR(kernel='rbf', C=1.0, epsilon=0.01)
                                elif algo == 'random_forest':
                                    fold_model = RandomForestRegressor(n_estimators=100, random_state=42)
                                elif algo == 'gradient_boosting':
                                    fold_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
                                elif algo == 'neural_network':
                                    fold_model = MLPRegressor(hidden_layer_sizes=(50,), random_state=42, max_iter=500)
                                else:
                                    continue
                                
                                fold_model.fit(X_train_scaled, y_train_scaled)
                                predictions_scaled = fold_model.predict(X_test_scaled)
                            
                            # Ensure predictions are valid
                            if np.any(np.isnan(predictions_scaled)):
                                predictions_scaled = np.nan_to_num(predictions_scaled, nan=np.nanmean(predictions_scaled))
                            
                            # Inverse transform
                            predictions = target_scaler.inverse_transform(
                                predictions_scaled.reshape(-1, 1)
                            ).flatten()
                            
                            actuals = target_scaler.inverse_transform(
                                y_test_scaled.reshape(-1, 1)
                            ).flatten()
                            
                            # Ensure same length
                            min_len = min(len(predictions), len(actuals))
                            predictions = predictions[:min_len]
                            actuals = actuals[:min_len]
                            
                            if min_len > 0:
                                # Calculate metrics
                                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                                mae = mean_absolute_error(actuals, predictions)
                                r2 = r2_score(actuals, predictions) if len(actuals) > 1 else 0
                                
                                # Direction accuracy
                                if len(actuals) > 1 and len(predictions) > 1:
                                    actual_direction = np.diff(actuals) > 0
                                    pred_direction = np.diff(predictions) > 0
                                    min_dir_len = min(len(actual_direction), len(pred_direction))
                                    if min_dir_len > 0:
                                        direction_acc = np.mean(actual_direction[:min_dir_len] == pred_direction[:min_dir_len]) * 100
                                    else:
                                        direction_acc = 0
                                else:
                                    direction_acc = 0
                                
                                fold_scores[algo]['rmse'].append(rmse)
                                fold_scores[algo]['mae'].append(mae)
                                fold_scores[algo]['r2'].append(r2)
                                fold_scores[algo]['direction'].append(direction_acc)
                            
                        except Exception as e:
                            print(f"    Error in fold {fold} for {algo}: {e}")
                            continue
                
                # Calculate average performance across folds
                target_performance = {}
                target_weights = {}
                
                for algo in self.models[target].keys():
                    if algo not in fold_scores or not fold_scores[algo]['rmse']:
                        continue
                    
                    # Average metrics
                    avg_rmse = np.mean(fold_scores[algo]['rmse'])
                    avg_mae = np.mean(fold_scores[algo]['mae'])
                    avg_r2 = np.mean(fold_scores[algo]['r2'])
                    avg_direction = np.mean(fold_scores[algo]['direction'])
                    
                    # Calculate MAPE
                    mape = avg_mae / (np.mean(y) + 1e-10) * 100 if np.mean(y) != 0 else 100
                    
                    target_performance[algo] = {
                        'rmse': float(avg_rmse),
                        'mae': float(avg_mae),
                        'r2': float(avg_r2),
                        'mape': float(mape),
                        'direction_accuracy': float(avg_direction),
                        'fold_count': len(fold_scores[algo]['rmse'])
                    }
                    
                    # Calculate algorithm weight based on performance
                    if avg_rmse > 0:
                        # Normalize RMSE relative to target mean
                        rmse_norm = avg_rmse / (np.mean(y) + 1e-10)
                        # Weight formula: combines multiple metrics
                        weight = (
                            (1 / (rmse_norm + 0.1)) * 0.4 +
                            (max(avg_r2, 0) + 1) * 0.3 +
                            (avg_direction / 100) * 0.3
                        )
                        target_weights[algo] = max(weight, 0.1)
                        print(f"    {algo}: RMSE={avg_rmse:.2f}, R²={avg_r2:.3f}, Dir={avg_direction:.1f}%, Weight={target_weights[algo]:.2f}")
                    else:
                        target_weights[algo] = 0.1
                
                performance[target] = target_performance
                self.algorithm_weights[target] = target_weights
            
            self.historical_performance = performance
            print(f"Performance calculated for {len(performance)} targets")
            return performance
            
        except Exception as e:
            print(f"Error calculating historical performance: {e}")
            return {}
    
    def calculate_risk_metrics(self, data):
        """Calculate various risk metrics"""
        try:
            if 'Close' not in data.columns:
                self.risk_metrics = {}
                return
            
            returns = data['Close'].pct_change().dropna()
            
            if len(returns) < 10:
                self.risk_metrics = {}
                return
            
            # Ensure no NaN in returns
            returns = returns.fillna(0)
            
            # Remove extreme outliers
            returns_mean, returns_std = returns.mean(), returns.std()
            if returns_std > 0:
                returns = returns[np.abs(returns - returns_mean) < 3 * returns_std]
            
            if len(returns) < 10:
                self.risk_metrics = {}
                return
            
            # Basic risk metrics
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5) * 100 if len(returns) >= 20 else 0
            var_99 = np.percentile(returns, 1) * 100 if len(returns) >= 100 else 0
            
            # Expected Shortfall (CVaR)
            cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100 if len(returns) >= 20 else 0
            cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100 if len(returns) >= 100 else 0
            
            # Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max.replace(0, 1e-10)
            max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0
            
            # Skewness and Kurtosis
            skewness = returns.skew() if len(returns) > 2 else 0
            kurtosis = returns.kurtosis() if len(returns) > 3 else 0
            
            # Anomaly detection
            anomaly_score = detect_anomalies(data).mean() * 100
            
            # Beta calculation
            try:
                market_volatility = 0.15
                beta = volatility / market_volatility if market_volatility > 0 else 1.0
            except:
                beta = 1.0
            
            self.risk_metrics = {
                'volatility': float(volatility * 100),
                'sharpe_ratio': float(sharpe_ratio),
                'var_95': float(var_95),
                'var_99': float(var_99),
                'cvar_95': float(cvar_95),
                'cvar_99': float(cvar_99),
                'max_drawdown': float(max_drawdown),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'anomaly_score': float(anomaly_score),
                'beta': float(beta),
                'total_returns': float(returns.mean() * 252 * 100),
                'positive_days': float((returns > 0).mean() * 100),
                'negative_days': float((returns < 0).mean() * 100)
            }
            
            print(f"Risk metrics calculated: {len(self.risk_metrics)} metrics")
            
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            self.risk_metrics = {}
    
    def update_prediction_history(self, symbol, data):
        """Update prediction history with latest actuals"""
        try:
            if symbol not in self.prediction_history:
                self.prediction_history[symbol] = []
            
            # Get latest actual prices
            latest_actuals = {
                'date': data['Date'].iloc[-1] if 'Date' in data.columns else datetime.now().strftime('%Y-%m-%d'),
                'actual_open': float(data['Open'].iloc[-1]) if 'Open' in data.columns else 0,
                'actual_high': float(data['High'].iloc[-1]) if 'High' in data.columns else 0,
                'actual_low': float(data['Low'].iloc[-1]) if 'Low' in data.columns else 0,
                'actual_close': float(data['Close'].iloc[-1]) if 'Close' in data.columns else 0,
                'volume': float(data['Volume'].iloc[-1]) if 'Volume' in data.columns else 0
            }
            
            # Keep only last 100 predictions
            self.prediction_history[symbol].append(latest_actuals)
            if len(self.prediction_history[symbol]) > 100:
                self.prediction_history[symbol] = self.prediction_history[symbol][-100:]
                
        except Exception as e:
            print(f"Error updating prediction history: {e}")
    
    def predict_ochl(self, data, symbol):
        """Predict next day's OCHL using ensemble of all algorithms with outlier filtering"""
        try:
            print(f"\n{'='*60}")
            print(f"PREDICTING FOR {symbol}")
            print(f"{'='*60}")
            
            X_data, _, data_with_features = self.prepare_training_data(data)
            
            if X_data is None:
                print("Error: Could not prepare prediction data")
                return None, "Insufficient data for prediction"
            
            predictions = {}
            confidence_scores = {}
            algorithm_predictions = {}
            
            current_close = data_with_features['Close'].iloc[-1] if 'Close' in data_with_features.columns else 100.0
            print(f"Current Close Price: ${current_close:.2f}")
            
            for target in self.targets:
                print(f"\n  Predicting {target}...")
                print(f"  {'─'*40}")
                
                if target not in self.models or target not in X_data or X_data[target] is None:
                    print(f"    Skipping: No model or data")
                    predictions[target] = current_close
                    confidence_scores[target] = 50
                    algorithm_predictions[target] = {'ensemble': float(current_close)}
                    continue
            
                X = X_data[target]
                if len(X) == 0:
                    print(f"    Skipping: Empty data")
                    predictions[target] = current_close
                    confidence_scores[target] = 50
                    algorithm_predictions[target] = {'ensemble': float(current_close)}
                    continue
                
                # Get latest features for prediction
                latest_features = X[-1:].reshape(1,-1)
                
                expected_dim = len(self.feature_columns) * 30
                if latest_features.shape[1] != expected_dim:
                    print(f"    Feature mismatch ({latest_features.shape[1]} != {expected_dim})")
                    predictions[target] = current_close
                    confidence_scores[target] = 50.0
                    algorithm_predictions[target] = {'ensemble': float(current_close)}
                    continue
                
                # Scale features safely
                if not hasattr(self.feature_scaler, "median_"):
                   print("    Feature scaler not fitted")
                   predictions[target] = current_close
                   confidence_scores[target] = 50.0
                   algorithm_predictions[target] = {'ensemble': float(current_close)}
                   continue
                
                try:
                    latest_scaled = self.feature_scaler.transform(latest_features)
                except Exception as e:
                    print(f"    Feature scaling failed: {e}")
                    predictions[target] = current_close
                    confidence_scores[target] = 50.0
                    algorithm_predictions[target] = {'ensemble': float(current_close)}
                    continue
                
                target_predictions = {}
                target_confidences = {}
                target_details = {}
                
                for algo, model in self.models[target].items():
                    if model is None:
                        continue
                    
                    try:
                        pred_actual = None
                        algo_details = {}
                        
                        if algo == 'arima':
                            target_scaler = self.scalers.get(target, StandardScaler())
                            
                            if target in data_with_features.columns:
                                y_recent = data_with_features[target].values[-100:]
                                if len(y_recent) > 30:
                                    y_scaled = target_scaler.transform(y_recent.reshape(-1, 1)).flatten()
                                    
                                    try:
                                        temp_arima = pm.auto_arima(
                                            y_scaled,
                                            start_p=1, start_q=1,
                                            max_p=2, max_q=2,
                                            seasonal=False,
                                            trace=False,
                                            error_action='ignore',
                                            stepwise=True
                                        )
                                        pred_scaled = temp_arima.predict(n_periods=1)[0]
                                        pred_actual = target_scaler.inverse_transform(
                                            np.array([[pred_scaled]])
                                        )[0][0]
                                        algo_details = {'order': temp_arima.order}
                                    except Exception as e:
                                        print(f"      ARIMA failed: {e}")
                                        pred_actual = data_with_features[target].iloc[-1]
                                else:
                                    pred_actual = data_with_features[target].iloc[-1]
                            else:
                                pred_actual = current_close
                        else:
                            pred_scaled = float(model.predict(latest_scaled)[0])
                            target_scaler = self.scalers.get(target, StandardScaler())
                            pred_actual = target_scaler.inverse_transform(
                                np.array([[pred_scaled]])
                            )[0][0]
                        
                        # Sanity check
                        if (pred_actual is None or np.isnan(pred_actual) or np.isinf(pred_actual) or
                            pred_actual <= 0 or pred_actual > current_close * 10):
                            print(f"      {algo}: INVALID PREDICTION (${pred_actual:.2f})")
                            pred_actual = current_close * random.uniform(0.97, 1.03)
                        
                        # Calculate confidence
                        confidence = 70
                        if (target in self.historical_performance and 
                            algo in self.historical_performance[target]):
                            perf = self.historical_performance[target][algo]
                            direction_acc = perf.get('direction_accuracy', 50)
                            mape_score = perf.get('mape', 10)
                            r2_score_val = perf.get('r2', 0)
                            
                            confidence = (
                                direction_acc * 0.4 +
                                (100 - min(mape_score, 100)) * 0.4 +
                                (max(r2_score_val, 0) * 100) * 0.2
                            )
                            confidence = min(max(confidence, 0), 100)
                        
                        target_predictions[algo] = float(pred_actual)
                        target_confidences[algo] = float(confidence)
                        target_details[algo] = algo_details
                        
                        print(f"      {algo}: ${pred_actual:.2f} (conf: {confidence:.1f}%)")
                        
                    except Exception as e:
                        print(f"      {algo} error: {e}")
                        fallback = current_close * random.uniform(0.98, 1.02)
                        target_predictions[algo] = float(fallback)
                        target_confidences[algo] = 50.0
                
                if target_predictions:
                    # FILTER OUTLIERS
                    pred_values = list(target_predictions.values())
                    if len(pred_values) >= 3:
                        median_pred = np.median(pred_values)
                        mad = np.median(np.abs(pred_values - median_pred))
                        
                        filtered_predictions = {}
                        filtered_confidences = {}
                        filtered_details = {}
                        
                        for algo, pred in target_predictions.items():
                            if abs(pred - median_pred) <= 3 * mad or mad == 0:
                                filtered_predictions[algo] = pred
                                filtered_confidences[algo] = target_confidences[algo]
                                filtered_details[algo] = target_details.get(algo, {})
                            else:
                                print(f"      Filtered outlier: {algo} (${pred:.2f} vs median ${median_pred:.2f})")
                        
                        if not filtered_predictions:
                            filtered_predictions = {'median': median_pred}
                            filtered_confidences = {'median': 50.0}
                        
                        target_predictions = filtered_predictions
                        target_confidences = filtered_confidences
                    
                    # Weighted ensemble
                    weights = {}
                    total_weight = 0
                    
                    for algo, conf in target_confidences.items():
                        algo_weight = self.algorithm_weights.get(target, {}).get(algo, 1.0)
                        weight = max(conf / 100, 0.1) * algo_weight
                        weights[algo] = weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        ensemble_pred = sum(
                            pred * (weights[algo] / total_weight)
                            for algo, pred in target_predictions.items()
                        )
                    else:
                        ensemble_pred = np.median(list(target_predictions.values()))
                    
                    # Final sanity check
                    if ensemble_pred <= 0 or ensemble_pred > current_close * 5:
                        print(f"    Unreasonable ensemble: ${ensemble_pred:.2f}")
                        ensemble_pred = np.median(list(target_predictions.values()))
                    
                    predictions[target] = float(ensemble_pred)
                    confidence_scores[target] = float(np.median(list(target_confidences.values())))
                    algorithm_predictions[target] = {
                        'individual': target_predictions,
                        'confidences': target_confidences,
                        'weights': weights,
                        'details': target_details,
                        'ensemble': float(ensemble_pred),
                        'ensemble_confidence': float(np.median(list(target_confidences.values())))
                    }
                    
                    print(f"    ✓ Ensemble: ${ensemble_pred:.2f} (conf: {confidence_scores[target]:.1f}%)")
                else:
                    predictions[target] = float(current_close)
                    confidence_scores[target] = 50.0
                    algorithm_predictions[target] = {'ensemble': float(current_close)}
                    print(f"    Fallback: ${current_close:.2f}")
            
            # Ensure we have predictions for all targets
            for target in self.targets:
                if target not in predictions:
                    predictions[target] = float(current_close)
                    confidence_scores[target] = 50.0
            
            # Enforce OHLC constraints
            pred_open = predictions["Open"]
            pred_close = predictions["Close"]
            pred_high = predictions["High"]
            pred_low = predictions["Low"]

            pred_high = max(pred_high, pred_open, pred_close)
            pred_low = min(pred_low, pred_open, pred_close)
            
            # Bound predictions
            max_change = 0.10
            for target in predictions:
                predictions[target] = max(predictions[target], current_close * (1 - max_change))
                predictions[target] = min(predictions[target], current_close * (1 + max_change))

            predictions["High"] = pred_high
            predictions["Low"] = pred_low

            # Generate risk alerts
            risk_alerts = self.generate_risk_alerts(data_with_features, predictions)
            
            # Calculate prediction confidence metrics
            confidence_metrics = self.calculate_prediction_confidence(predictions, confidence_scores)
            
            result = {
                'predictions': predictions,
                'algorithm_details': algorithm_predictions,
                'confidence_scores': confidence_scores,
                'confidence_metrics': confidence_metrics,
                'risk_alerts': risk_alerts,
                'current_prices': {
                    'open': float(data['Open'].iloc[-1]) if 'Open' in data.columns else current_close * 0.995,
                    'high': float(data['High'].iloc[-1]) if 'High' in data.columns else current_close * 1.015,
                    'low': float(data['Low'].iloc[-1]) if 'Low' in data.columns else current_close * 0.985,
                    'close': float(current_close)
                }
            }
            
            print(f"\n{'='*60}")
            print(f"PREDICTION COMPLETE FOR {symbol}")
            print(f"{'='*60}")
            for target in self.targets:
                if target in predictions:
                    change = ((predictions[target] - current_close) / current_close) * 100
                    print(f"{target}: ${predictions[target]:.2f} ({change:+.1f}%)")
            
            return result, None
            
        except Exception as e:
            print(f"Error predicting OCHL: {e}")
            import traceback
            traceback.print_exc()
            return None, str(e)
    
    def calculate_prediction_confidence(self, predictions, confidence_scores):
        """Calculate comprehensive confidence metrics"""
        try:
            if not predictions or not confidence_scores:
                return {}
            
            overall_confidence = np.median(list(confidence_scores.values())) if confidence_scores else 0
            
            predicted_values = list(predictions.values())
            if len(predicted_values) >= 2:
                median_val = np.median(predicted_values)
                if median_val != 0:
                    mad = np.median(np.abs(predicted_values - median_val))
                    if median_val != 0:
                        consistency = 1 - (mad / abs(median_val))
                        consistency_score = max(0, min(100, consistency * 100))
                    else:
                        consistency_score = 0
                else:
                    consistency_score = 0
            else:
                consistency_score = 0
            
            if overall_confidence >= 80:
                confidence_level = "HIGH"
                confidence_color = "success"
            elif overall_confidence >= 60:
                confidence_level = "MEDIUM"
                confidence_color = "warning"
            else:
                confidence_level = "LOW"
                confidence_color = "danger"
            
            return {
                'overall_confidence': float(overall_confidence),
                'consistency_score': float(consistency_score),
                'confidence_level': confidence_level,
                'confidence_color': confidence_color,
                'target_confidence': {
                    target: {
                        'score': float(confidence_scores.get(target, 0)),
                        'level': "HIGH" if confidence_scores.get(target, 0) >= 80 else 
                                 "MEDIUM" if confidence_scores.get(target, 0) >= 60 else "LOW"
                    }
                    for target in self.targets if target in confidence_scores
                }
            }
        except Exception as e:
            print(f"Error calculating prediction confidence: {e}")
            return {}
    
    def generate_risk_alerts(self, data, predictions):
        """Generate risk alerts based on various factors"""
        alerts = []
        
        try:
            current_close = data['Close'].iloc[-1] if 'Close' in data.columns else 100.0
            predicted_close = predictions.get('Close', current_close)
            
            expected_change = ((predicted_close - current_close) / current_close) * 100 if current_close != 0 else 0
            
            # Price movement alerts
            if abs(expected_change) > 10:
                alerts.append({
                    'level': '🔴 CRITICAL',
                    'type': 'Extreme Price Movement',
                    'message': f'Expected price change of {expected_change:+.1f}%',
                    'details': 'Consider adjusting position size or setting stop-loss'
                })
            elif abs(expected_change) > 5:
                alerts.append({
                    'level': '🟡 HIGH',
                    'type': 'Large Price Movement',
                    'message': f'Expected price change of {expected_change:+.1f}%',
                    'details': 'Monitor position closely'
                })
            
            # Volatility alerts
            if 'Volatility_5d' in data.columns:
                recent_volatility = data['Volatility_5d'].iloc[-10:].mean()
                if recent_volatility > 0.03:
                    alerts.append({
                        'level': '🟡 HIGH',
                        'type': 'High Volatility',
                        'message': f'Recent volatility: {recent_volatility:.2%}',
                        'details': 'Market showing increased volatility'
                    })
            
            # RSI alerts
            if 'RSI_14' in data.columns and len(data) > 0:
                current_rsi = data['RSI_14'].iloc[-1]
                if current_rsi > 80:
                    alerts.append({
                        'level': '🟡 HIGH',
                        'type': 'Overbought Condition',
                        'message': f'RSI at {current_rsi:.1f} (Overbought)',
                        'details': 'Consider taking profits or waiting for pullback'
                    })
                elif current_rsi < 20:
                    alerts.append({
                        'level': '🟢 MEDIUM',
                        'type': 'Oversold Condition',
                        'message': f'RSI at {current_rsi:.1f} (Oversold)',
                        'details': 'Potential buying opportunity'
                    })
            
            # Volume alerts
            if 'Volume_Ratio' in data.columns and len(data) > 0:
                volume_ratio = data['Volume_Ratio'].iloc[-1]
                if volume_ratio > 2.0:
                    alerts.append({
                        'level': '🟡 HIGH',
                        'type': 'Unusual Volume',
                        'message': f'Volume {volume_ratio:.1f}x average',
                        'details': 'High volume may indicate significant news or event'
                    })
            
            # Prediction confidence alerts
            confidence_metrics = self.calculate_prediction_confidence(predictions, {})
            overall_conf = confidence_metrics.get('overall_confidence', 0)
            
            if overall_conf < 50:
                alerts.append({
                    'level': '🟡 HIGH',
                    'type': 'Low Prediction Confidence',
                    'message': f'Model confidence: {overall_conf:.1f}%',
                    'details': 'Consider additional analysis before trading'
                })
            
            # Risk metric alerts
            if self.risk_metrics:
                if self.risk_metrics.get('max_drawdown', 0) < -20:
                    alerts.append({
                        'level': '🔴 CRITICAL',
                        'type': 'Large Historical Drawdown',
                        'message': f'Max drawdown: {self.risk_metrics["max_drawdown"]:.1f}%',
                        'details': 'Stock has experienced significant declines historically'
                    })
                
                if self.risk_metrics.get('var_95', 0) < -5:
                    alerts.append({
                        'level': '🟡 HIGH',
                        'type': 'High Downside Risk',
                        'message': f'95% VaR: {self.risk_metrics["var_95"]:.1f}%',
                        'details': 'High probability of significant losses'
                    })
            
            # Market regime alert
            market_regime = self.detect_market_regime(data)
            if market_regime != "NORMAL":
                alerts.append({
                    'level': '🟡 HIGH' if market_regime == "HIGH_VOLATILITY" else '🟢 MEDIUM',
                    'type': 'Special Market Regime',
                    'message': f'Current regime: {market_regime.replace("_", " ").title()}',
                    'details': 'Market conditions may affect prediction accuracy'
                })
            
            severity_order = {'🔴 CRITICAL': 0, '🟡 HIGH': 1, '🟢 MEDIUM': 2, '🟢 LOW': 3}
            alerts.sort(key=lambda x: severity_order.get(x['level'], 4))
            
            print(f"Generated {len(alerts)} risk alerts")
            return alerts
            
        except Exception as e:
            print(f"Error generating risk alerts: {e}")
            return []
    
    def detect_market_regime(self, data):
        """Detect current market regime"""
        try:
            if len(data) < 20:
                return "NORMAL"
            
            recent = data.tail(20)
            
            volatility = recent['Volatility_5d'].mean() if 'Volatility_5d' in recent.columns else 0
            volume_ratio = recent['Volume_Ratio'].mean() if 'Volume_Ratio' in recent.columns else 1
            rsi = recent['RSI_14'].iloc[-1] if 'RSI_14' in recent.columns else 50
            
            if volatility > 0.03:
                return "HIGH_VOLATILITY"
            elif volume_ratio > 1.5:
                return "HIGH_VOLUME"
            elif rsi > 75 or rsi < 25:
                return "EXTREME_SENTIMENT"
            else:
                return "NORMAL"
                
        except:
            return "NORMAL"

# Global predictor instance
predictor = OCHLPredictor()

# ---------------- Helper Functions ----------------
def get_next_trading_day():
    today = datetime.now()
    if today.weekday() == 4:  # Friday
        return (today + timedelta(days=3)).strftime('%Y-%m-%d')
    if today.weekday() == 5:  # Saturday
        return (today + timedelta(days=2)).strftime('%Y-%m-%d')
    if today.weekday() == 6:  # Sunday
        return (today + timedelta(days=1)).strftime('%Y-%m-%d')
    return (today + timedelta(days=1)).strftime('%Y-%m-%d')

def get_last_market_date():
    today = datetime.now()
    if today.weekday() == 0:  # Monday
        return (today - timedelta(days=3)).strftime('%Y-%m-%d')
    if today.weekday() == 6:  # Sunday
        return (today - timedelta(days=2)).strftime('%Y-%m-%d')
    if today.weekday() == 5:  # Saturday
        return (today - timedelta(days=1)).strftime('%Y-%m-%d')
    return (today - timedelta(days=1)).strftime('%Y-%m-%d')

def get_market_status():
    today = datetime.now()
    if today.weekday() >= 5:
        return "closed", "Market is closed on weekends"
    
    current_time = datetime.now().time()
    market_open = datetime.strptime('09:30', '%H:%M').time()
    market_close = datetime.strptime('16:00', '%H:%M').time()
    
    if current_time < market_open:
        return "pre_market", "Pre-market hours"
    if current_time > market_close:
        return "after_hours", "After-hours trading"
    
    return "open", "Market is open"

def get_risk_level_from_metrics(risk_metrics):
    """Calculate overall risk level from risk metrics"""
    if not risk_metrics:
        return "🟢 LOW RISK"
    
    risk_score = 0
    
    if 'volatility' in risk_metrics:
        risk_score += min(risk_metrics['volatility'] / 50, 1) * 0.3
    
    if 'max_drawdown' in risk_metrics:
        risk_score += min(abs(risk_metrics['max_drawdown']) / 30, 1) * 0.3
    
    if 'var_95' in risk_metrics:
        risk_score += min(abs(risk_metrics['var_95']) / 10, 1) * 0.2
    
    if 'anomaly_score' in risk_metrics:
        risk_score += (risk_metrics['anomaly_score'] / 100) * 0.2
    
    if risk_score > 0.7:
        return "🔴 EXTREME RISK"
    elif risk_score > 0.5:
        return "🟡 HIGH RISK"
    elif risk_score > 0.3:
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

def get_trading_recommendation(predictions, current_prices, confidence):
    """Generate trading recommendation"""
    if confidence < 50:
        return "🚨 LOW CONFIDENCE - WAIT"
    
    expected_change = ((predictions.get('Close', current_prices['close']) - current_prices['close']) / current_prices['close']) * 100
    
    if expected_change > 7:
        return "✅ STRONG BUY"
    elif expected_change > 3:
        return "📈 BUY"
    elif expected_change < -7:
        return "💼 STRONG SELL"
    elif expected_change < -3:
        return "📉 SELL"
    elif abs(expected_change) < 1:
        return "🔄 HOLD / SIDEWAYS"
    else:
        return "🔄 HOLD"

# ---------------- NAVIGATION MAP ----------------
NAVIGATION_MAP = {
    'index':'/','jeet':'/jeet','portfolio':'/portfolio','mystock':'/mystock','deposit':'/deposit',
    'insight':'/insight','prediction':'/prediction','news':'/news','videos':'/videos','superstars':'/Superstars',
    'alerts':'/alerts','help':'/help','profile':'/profile'
}

# ---------------- Dynamic routes for NAVIGATION_MAP ----------------
def _find_template_for_page(page_key):
    candidates = [
        f"{page_key}.html",
        f"{page_key.capitalize()}.html",
        f"{page_key.lower()}.html",
        f"{page_key.title()}.html",
    ]
    for fn in candidates:
        full = os.path.join(current_dir, 'templates', fn)
        if os.path.exists(full):
            return fn
    fallback = 'jeet.html'
    if os.path.exists(os.path.join(current_dir, 'templates', fallback)):
        return fallback
    return 'index.html'

# Register routes dynamically
for page_name, route_path in NAVIGATION_MAP.items():
    def make_view(p=page_name):
        def view():
            template_name = _find_template_for_page(p)
            return render_template(template_name, navigation=NAVIGATION_MAP)
        return view
    try:
        server.add_url_rule(route_path, endpoint=page_name, view_func=make_view(), methods=['GET'])
    except AssertionError:
        try:
            server.url_map._rules = [r for r in server.url_map._rules if r.endpoint != page_name]
            server.add_url_rule(route_path, endpoint=page_name, view_func=make_view(), methods=['GET'])
        except Exception:
            pass

@server.route('/navigate/<page_name>')
def navigate_to_page(page_name):
    return redirect(NAVIGATION_MAP.get(page_name, '/'))

# ---------------- API Endpoints ----------------
@server.route('/api/stocks')
@rate_limiter
def get_stocks_list():
    popular_stocks = [
        {"symbol":"AAPL","name":"Apple Inc.","price":182.63,"change":1.24},
        {"symbol":"MSFT","name":"Microsoft Corp.","price":407.57,"change":-0.85},
        {"symbol":"GOOGL","name":"Alphabet Inc.","price":172.34,"change":2.13},
        {"symbol":"AMZN","name":"Amazon.com Inc.","price":178.22,"change":0.67},
        {"symbol":"TSLA","name":"Tesla Inc.","price":175.79,"change":-3.21},
        {"symbol":"GE","name":"GE Aerospace","price":148.25,"change":0.0},
        {"symbol":"PLMR","name":"Palomar Holdings","price":199.47,"change":0.0},
        {"symbol":"META","name":"Meta Platforms Inc.","price":485.00,"change":0.0},
        {"symbol":"NVDA","name":"NVIDIA Corp.","price":620.00,"change":0.0},
        {"symbol":"NKE","name":"Nike Inc.","price":110.00,"change":0.0}
    ]
    
    for stock in popular_stocks:
        try:
            t = yf.Ticker(stock['symbol'])
            h = t.history(period='1d', interval='1m')
            if not h.empty:
                current = h['Close'].iloc[-1]
                prev_close = t.info.get('previousClose', current)
                change = ((current-prev_close)/prev_close)*100 if prev_close!=0 else 0.0
                stock['price'] = round(current,2)
                stock['change'] = round(change,2)
        except Exception:
            continue
    
    return jsonify(popular_stocks)

@server.route('/api/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "5.0.0",
        "algorithms": ["Linear Regression", "Ridge", "Lasso", "SVR", "Random Forest", "Gradient Boosting", "ARIMA", "Neural Network"],
        "features": "Complete OCHL Prediction with 80+ Features"
    })

@server.route('/api/predict', methods=['POST'])
@rate_limiter
def predict_stock():
    """Main prediction endpoint with full OCHL prediction"""
    try:
        data = request.get_json() or {}
        symbol = (data.get('symbol') or 'SPY').upper().strip()
        
        print(f"\n{'='*60}")
        print(f"🚀 PREDICTION REQUEST FOR {symbol}")
        print(f"{'='*60}")
        
        # Get historical data
        historical_data, current_price, error = get_live_stock_data_enhanced(symbol)
        if error:
            print(f"Error fetching data: {error}")
            return jsonify({"error": str(error)}), 400
        
        print(f"📊 Fetched {len(historical_data)} days of data")
        
        # Load existing models or train new ones
        models_loaded = predictor.load_models(symbol)
        
        if not models_loaded or not predictor.is_fitted:
            print("🔧 No models found, training new ones...")
            success, train_msg = predictor.train_all_models(historical_data, symbol)
            if not success:
                print(f"Training failed: {train_msg}")
                return provide_fallback_prediction(symbol, historical_data)
            print("✅ Training successful")
        else:
            print("✅ Loaded existing models")
        
        # Make prediction
        print("🤖 Making predictions...")
        prediction_result, pred_error = predictor.predict_ochl(historical_data, symbol)
        
        if pred_error:
            print(f"Prediction error: {pred_error}")
            return provide_fallback_prediction(symbol, historical_data)
        
        # Prepare comprehensive response
        current_prices = prediction_result['current_prices']
        predictions = prediction_result['predictions']
        confidence_scores = prediction_result['confidence_scores']
        confidence_metrics = prediction_result['confidence_metrics']
        risk_alerts = prediction_result['risk_alerts']
        algorithm_details = prediction_result.get('algorithm_details', {})
        
        # Calculate expected changes
        expected_changes = {}
        for target in ['Open', 'High', 'Low', 'Close']:
            if target in predictions and target.lower() in current_prices:
                current = current_prices[target.lower()]
                predicted = predictions[target]
                change = ((predicted - current) / current) * 100 if current != 0 else 0
                expected_changes[target] = change
        
        # Generate trading recommendation
        overall_confidence = confidence_metrics.get('overall_confidence', 70)
        recommendation = get_trading_recommendation(predictions, current_prices, overall_confidence)
        
        # Get risk level
        risk_level = get_risk_level_from_metrics(predictor.risk_metrics)
        
        # Format algorithm performance for display
        algorithm_performance = {}
        for target in predictor.historical_performance:
            algorithm_performance[target] = {}
            for algo, perf in predictor.historical_performance[target].items():
                algorithm_performance[target][algo] = {
                    'r2': round(perf.get('r2', 0), 3),
                    'direction_accuracy': round(perf.get('direction_accuracy', 0), 1),
                    'mape': round(perf.get('mape', 0), 1),
                    'weight': round(predictor.algorithm_weights.get(target, {}).get(algo, 0.1), 2)
                }
        
        response = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "market_status": get_market_status()[1],
            "last_trading_day": get_last_market_date(),
            "prediction_date": get_next_trading_day(),
            
            # Current prices
            "current_prices": current_prices,
            
            # Predictions
            "predictions": {
                target: {
                    "predicted_price": round(predictions.get(target, current_prices[target.lower()]), 2),
                    "current_price": round(current_prices.get(target.lower(), current_price), 2),
                    "expected_change": round(expected_changes.get(target, 0), 2),
                    "confidence": round(confidence_scores.get(target, 70), 1)
                }
                for target in ['Open', 'High', 'Low', 'Close']
            },
            
            # Algorithm details with individual predictions
            "algorithm_predictions": algorithm_details,
            
            # Confidence metrics
            "confidence_metrics": confidence_metrics,
            
            # Historical performance
            "historical_performance": algorithm_performance,
            
            # Risk metrics
            "risk_metrics": predictor.risk_metrics,
            
            # Risk alerts
            "risk_alerts": risk_alerts,
            
            # Risk level and recommendation
            "risk_level": risk_level,
            "trading_recommendation": recommendation,
            
            # Prediction history
            "prediction_history": predictor.prediction_history.get(symbol, [])[-10:],
            
            # Model info
            "model_info": {
                "last_training_date": predictor.last_training_date,
                "is_fitted": predictor.is_fitted,
                "targets_trained": list(predictor.models.keys()),
                "feature_count": len(predictor.feature_columns),
                "algorithms_used": ["linear_regression", "ridge", "lasso", "svr", "random_forest", "gradient_boosting", "arima", "neural_network"],
                "algorithm_weights": predictor.algorithm_weights
            },
            
            # Data info
            "data_info": {
                "total_days": len(historical_data),
                "date_range": {
                    "start": historical_data['Date'].iloc[0] if 'Date' in historical_data.columns else "N/A",
                    "end": historical_data['Date'].iloc[-1] if 'Date' in historical_data.columns else "N/A"
                }
            },
            
            # Summary insight
            "insight": f"AI predicts {expected_changes.get('Close', 0):+.1f}% change for {symbol}. {recommendation}. Confidence: {overall_confidence:.1f}%"
        }
        
        print(f"\n{'='*60}")
        print(f"🎯 PREDICTION SUMMARY FOR {symbol}")
        print(f"{'='*60}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Close: ${predictions.get('Close', 0):.2f} ({expected_changes.get('Close', 0):+.1f}%)")
        print(f"Overall Confidence: {overall_confidence:.1f}%")
        print(f"Risk Level: {risk_level}")
        print(f"Recommendation: {recommendation}")
        print(f"Algorithms Used: {len(algorithm_details.get('Close', {}).get('individual', {}))}")
        print(f"{'='*60}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Prediction service temporarily unavailable",
            "fallback": True,
            "symbol": symbol if 'symbol' in locals() else 'UNKNOWN'
        }), 500

@server.route('/api/train', methods=['POST'])
@rate_limiter
def train_models():
    """Explicit training endpoint"""
    try:
        data = request.get_json() or {}
        symbol = (data.get('symbol') or 'SPY').upper().strip()
        
        print(f"\n=== TRAINING REQUEST FOR {symbol} ===")
        
        historical_data, current_price, error = get_live_stock_data_enhanced(symbol)
        if error:
            return jsonify({"error": str(error)}), 400
        
        success, train_msg = predictor.train_all_models(historical_data, symbol)
        
        if success:
            return jsonify({
                "status": "success",
                "message": train_msg,
                "symbol": symbol,
                "last_training_date": predictor.last_training_date,
                "historical_performance": predictor.historical_performance,
                "risk_metrics": predictor.risk_metrics,
                "algorithm_weights": predictor.algorithm_weights
            })
        else:
            return jsonify({
                "status": "error",
                "message": train_msg,
                "symbol": symbol
            }), 400
            
    except Exception as e:
        print(f"Training endpoint error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@server.route('/api/history/<symbol>')
@rate_limiter
def get_prediction_history(symbol):
    """Get prediction history for a symbol"""
    try:
        symbol = symbol.upper()
        
        history_path = os.path.join(HISTORY_DIR, f"{symbol}_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history_data = json.load(f)
            
            return jsonify({
                "status": "success",
                "symbol": symbol,
                "history": history_data.get('prediction_history', []),
                "performance": history_data.get('historical_performance', {}),
                "algorithm_weights": history_data.get('algorithm_weights', {}),
                "last_updated": history_data.get('last_training_date')
            })
        else:
            return jsonify({
                "status": "error",
                "message": "No history found for this symbol"
            }), 404
            
    except Exception as e:
        print(f"History endpoint error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def provide_fallback_prediction(symbol, historical_data):
    """Provide fallback prediction when models fail"""
    try:
        print(f"Using fallback prediction for {symbol}")
        
        if historical_data is None or historical_data.empty:
            current_price = 100.0
            print("No historical data, using default price")
        else:
            current_price = historical_data['Close'].iloc[-1] if 'Close' in historical_data.columns else 100.0
            print(f"Using current price: ${current_price}")
        
        predictions = {}
        for target in ['Open', 'High', 'Low', 'Close']:
            variation = random.uniform(-0.02, 0.02)
            pred = current_price * (1 + variation)
            predictions[target] = pred
        
        changes = {}
        for target, pred in predictions.items():
            change = ((pred - current_price) / current_price) * 100 if current_price != 0 else 0
            changes[target] = change
        
        return jsonify({
            "symbol": symbol,
            "current_prices": {
                "open": round(current_price * 0.995, 2),
                "high": round(current_price * 1.015, 2),
                "low": round(current_price * 0.985, 2),
                "close": round(current_price, 2)
            },
            "predictions": {
                target: {
                    "predicted_price": round(pred, 2),
                    "expected_change": round(changes[target], 2),
                    "confidence": 60
                }
                for target, pred in predictions.items()
            },
            "confidence_metrics": {
                "overall_confidence": 60,
                "confidence_level": "MEDIUM",
                "confidence_color": "warning"
            },
            "risk_alerts": [{
                "level": "🟡 HIGH",
                "type": "Fallback Mode",
                "message": "Using fallback prediction engine",
                "details": "Primary models unavailable, using simplified predictions"
            }],
            "risk_level": "🟠 MEDIUM RISK",
            "trading_recommendation": "🔄 HOLD",
            "fallback": True,
            "message": "Using fallback prediction engine"
        })
        
    except Exception as e:
        print(f"Fallback prediction error: {e}")
        return jsonify({
            "error": "Prediction service unavailable",
            "fallback": True
        }), 500

# ---------------- Serve static/templates ----------------
@server.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(os.path.join(current_dir,'static'), path)

# ---------------- Error handlers ----------------
@server.errorhandler(404)
def not_found(error): 
    return jsonify({"error":"Page not found"}), 404

@server.errorhandler(500)
def internal_error(error): 
    return jsonify({"error":"Internal server error"}), 500

# ---------------- Run ----------------
if __name__ == '__main__':
    print("=" * 60)
    print("📈 STOCK MARKET PREDICTION SYSTEM v5.0")
    print("=" * 60)
    print("✨ ENHANCEMENTS:")
    print("  • Added ALL missing features (80+ total)")
    print("  • Kept Linear Regression with outlier protection")
    print("  • Added Ridge, Lasso, Gradient Boosting")
    print("  • Added ATR, Support/Resistance, Seasonality")
    print("  • Added Stochastic, Williams %R, MACD signals")
    print("  • Added interaction and binary features")
    print("  • Robust outlier filtering in ensemble")
    print("=" * 60)
    print("📊 ALGORITHMS (8 total):")
    print("  Linear, Ridge, Lasso, SVR, Random Forest,")
    print("  Gradient Boosting, ARIMA, Neural Network")
    print("=" * 60)
    print("🚀 Ready for predictions!")
    print("=" * 60)
    
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    port = int(os.environ.get('PORT', 8080))
    server.run(host='0.0.0.0', port=port, debug=True, threaded=True)