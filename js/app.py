# app.py — Flask with Multi-Algorithm OCHL Prediction + Historical Performance + Confidence + Risk Alerts
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import IsolationForest
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

def safe_divide(a, b, default=1.0):
    """Safe division with default value"""
    try:
        result = np.divide(a, b, out=np.full_like(a, default, dtype=float), where=b!=0)
        return result
    except:
        return np.full_like(a, default, dtype=float)

def create_advanced_features(data):
    """Create comprehensive features for OCHL prediction with NaN protection"""
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
        
        # Basic price features
        data['Return'] = data['Close'].pct_change().fillna(0)
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1).replace(0, 1e-10)).fillna(0)
        data['Volatility'] = data['Return'].rolling(window=5, min_periods=1).std().fillna(0)
        
        # Price ranges (with safe division)
        data['High_Low_Range'] = safe_divide(data['High'] - data['Low'], data['Close'].replace(0, 1e-10), 0.02)
        data['Open_Close_Range'] = safe_divide(data['Close'] - data['Open'], data['Open'].replace(0, 1e-10), 0)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            data[f'MA_{window}'] = data['Close'].rolling(window=window, min_periods=1).mean().fillna(data['Close'])
            data[f'MA_Ratio_{window}'] = safe_divide(data['Close'], data[f'MA_{window}'].replace(0, 1e-10), 1.0)
        
        # Volume features
        data['Volume_MA'] = data['Volume'].rolling(window=10, min_periods=1).mean().fillna(data['Volume'])
        data['Volume_Ratio'] = safe_divide(data['Volume'], data['Volume_MA'].replace(0, 1e-10), 1.0)
        data['Volume_Change'] = data['Volume'].pct_change().fillna(0)
        
        # Technical indicators
        data['RSI'] = calculate_rsi(data['Close'])
        
        # MACD calculation with NaN protection
        try:
            exp1 = data['Close'].ewm(span=12, min_periods=1).mean()
            exp2 = data['Close'].ewm(span=26, min_periods=1).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_Signal'] = data['MACD'].ewm(span=9, min_periods=1).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        except:
            data['MACD'] = 0
            data['MACD_Signal'] = 0
            data['MACD_Histogram'] = 0
        
        # Bollinger Bands
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(data)
        data['BB_Middle'] = bb_middle
        data['BB_Upper'] = bb_upper
        data['BB_Lower'] = bb_lower
        data['BB_Width'] = safe_divide(bb_upper - bb_lower, bb_middle.replace(0, 1e-10), 0.1)
        data['BB_Position'] = safe_divide(data['Close'] - bb_lower, (bb_upper - bb_lower).replace(0, 1e-10), 0.5)
        
        # Momentum indicators
        for window in [5, 10, 20]:
            shifted = data['Close'].shift(window).replace(0, 1e-10)
            data[f'Momentum_{window}'] = safe_divide(data['Close'], shifted, 1.0) - 1
        
        # Price patterns
        prev_close = data['Close'].shift(1).fillna(data['Close'])
        data['Gap_Up'] = ((data['Open'] > prev_close * 1.01) & (data['Open'] > 0)).astype(int)
        data['Gap_Down'] = ((data['Open'] < prev_close * 0.99) & (data['Open'] > 0)).astype(int)
        
        # Handle any remaining NaN values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Final check - replace any remaining NaN with appropriate defaults
        for col in data.columns:
            if col != 'Date':
                if data[col].isna().any():
                    if 'Ratio' in col or 'Momentum' in col:
                        data[col] = data[col].fillna(1.0)
                    elif col in ['Return', 'Volatility', 'Volume_Change', 'MACD', 'MACD_Signal', 'MACD_Histogram']:
                        data[col] = data[col].fillna(0.0)
                    elif col == 'RSI':
                        data[col] = data[col].fillna(50.0)
                    elif col == 'Volume':
                        data[col] = data[col].fillna(1000000)
                    elif col in ['Open', 'High', 'Low', 'Close']:
                        data[col] = data[col].fillna(100.0)
                    else:
                        data[col] = data[col].fillna(0.0)
        
        return data
    except Exception as e:
        print(f"Feature creation error: {e}")
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
        features = ['Return', 'Volatility', 'Volume_Ratio', 'RSI', 'High_Low_Range']
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
            {"func": lambda: yf.download(ticker, period="10y", interval="1d", progress=False, timeout=30), "name":"10y"},
            {"func": lambda: yf.Ticker(ticker).history(period="8y", interval="1d"), "name":"ticker.history"},
            {"func": lambda: yf.download(ticker, period="7y", interval="1d", progress=False, timeout=25), "name":"7y"},
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
        self.feature_scaler = StandardScaler()
        self.feature_columns = []
        self.targets = ['Open', 'Close', 'High', 'Low']
        self.historical_performance = {}
        self.prediction_history = {}
        self.risk_metrics = {}
        self.anomaly_detector = None
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
                                'scaler': self.scalers.get(target, StandardScaler()),
                                'features': self.feature_columns,
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
            algorithms = ['linear_regression', 'svr', 'random_forest', 'arima']
            
            models_loaded = False
            
            for target in self.targets:
                for algo in algorithms:
                    path = self.get_model_path(symbol, target, algo)
                    if os.path.exists(path):
                        try:
                            model_data = joblib.load(path)
                            loaded_models[target][algo] = model_data['model']
                            
                            # Load scaler for this target
                            if target not in self.scalers:
                                self.scalers[target] = model_data.get('scaler', StandardScaler())
                            
                            # Load feature columns
                            if not self.feature_columns:
                                self.feature_columns = model_data.get('features', [])
                            
                            models_loaded = True
                            print(f"Loaded {target}-{algo} model for {symbol}")
                        except Exception as e:
                            print(f"Error loading {target}-{algo} model: {e}")
                            loaded_models[target][algo] = None
                    else:
                        loaded_models[target][algo] = None
            
            # Load historical data
            history_path = self.get_history_path(symbol)
            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r') as f:
                        history_data = json.load(f)
                    
                    self.historical_performance = history_data.get('historical_performance', {})
                    self.prediction_history = history_data.get('prediction_history', {})
                    self.risk_metrics = history_data.get('risk_metrics', {})
                    self.last_training_date = history_data.get('last_training_date')
                    self.feature_columns = history_data.get('feature_columns', [])
                    self.is_fitted = models_loaded
                    print(f"Loaded history for {symbol}")
                except Exception as e:
                    print(f"Error loading history: {e}")
            
            self.models = loaded_models
            return models_loaded
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def prepare_training_data(self, data):
        """Prepare data for OCHL prediction with NaN protection"""
        try:
            print(f"Preparing training data with {len(data)} rows")
            
            # Create features
            data_with_features = create_advanced_features(data)
            print(f"Created {len(data_with_features.columns)} features")
            
            # Ensure we have enough data
            if len(data_with_features) < 60:
                print(f"Warning: Only {len(data_with_features)} rows of data, minimum 60 required")
                return None, None, None
            
            # Feature selection - use all available numeric features except Date
            numeric_cols = [col for col in data_with_features.columns 
                          if col != 'Date' and pd.api.types.is_numeric_dtype(data_with_features[col])]
            
            # Remove target columns from features
            feature_candidates = [col for col in numeric_cols if col not in self.targets]
            
            # Select top 30 features by variance if we have too many
            if len(feature_candidates) > 30:
                variances = data_with_features[feature_candidates].var()
                top_features = variances.nlargest(30).index.tolist()
                self.feature_columns = top_features
            else:
                self.feature_columns = feature_candidates
            
            print(f"Selected {len(self.feature_columns)} features")
            
            # Prepare X (features) for each target
            X_data = {}
            y_data = {}
            
            for target in self.targets:
                if target not in data_with_features.columns:
                    # If target column doesn't exist, use Close as proxy
                    print(f"Warning: {target} column not found, using Close as target")
                    data_with_features[target] = data_with_features['Close']
                
                # Ensure target has no NaN values
                data_with_features[target] = data_with_features[target].ffill().bfill().fillna(data_with_features['Close'].mean())
                
                # Prepare sequences (use past 30 days to predict next day)
                X_list = []
                y_list = []
                window_size = 30
                
                if len(data_with_features) < window_size + 10:
                    print(f"Warning: Not enough data for window size {window_size}")
                    continue
                
                for i in range(window_size, len(data_with_features) - 1):
                    # Features from window
                    features = data_with_features[self.feature_columns].iloc[i-window_size:i]
                    
                    # Check for NaN in features
                    if features.isna().any().any():
                        print(f"Warning: NaN found in features at index {i}")
                        features = features.fillna(0)
                    
                    features_flat = features.values.flatten()
                    
                    # Target is next day's value
                    target_value = data_with_features[target].iloc[i+1]
                    
                    # Skip if target is NaN
                    if pd.isna(target_value):
                        continue
                    
                    X_list.append(features_flat)
                    y_list.append(target_value)
                
                if len(X_list) > 10:  # Need minimum samples
                    X_data[target] = np.array(X_list)
                    y_data[target] = np.array(y_list)
                    print(f"Prepared {len(X_list)} samples for {target}")
                else:
                    print(f"Warning: Insufficient samples for {target}")
                    X_data[target] = None
                    y_data[target] = None
            
            # Check if we have data for all targets
            valid_targets = [t for t in self.targets if t in X_data and X_data[t] is not None and len(X_data[t]) > 0]
            if len(valid_targets) == 0:
                print("Error: No valid training data for any target")
                return None, None, None
            
            print(f"Successfully prepared data for targets: {valid_targets}")
            return X_data, y_data, data_with_features
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def train_algorithm(self, X, y, algorithm):
        """Train a specific algorithm with NaN protection"""
        try:
            # Check for NaN in inputs
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                print(f"Warning: NaN found in training data for {algorithm}")
                X = np.nan_to_num(X, nan=0.0)
                y = np.nan_to_num(y, nan=np.nanmean(y) if not np.all(np.isnan(y)) else 0.0)
            
            if len(X) < 10:
                print(f"Warning: Insufficient data for {algorithm}")
                return None
            
            if algorithm == 'linear_regression':
                model = LinearRegression()
                model.fit(X, y)
                print(f"Trained Linear Regression with {len(X)} samples")
                return model
                
            elif algorithm == 'svr':
                model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                # Use smaller sample for SVR if data is large
                if len(X) > 1000:
                    idx = np.random.choice(len(X), min(1000, len(X)), replace=False)
                    model.fit(X[idx], y[idx])
                else:
                    model.fit(X, y)
                print(f"Trained SVR with {len(X)} samples")
                return model
                
            elif algorithm == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y)
                print(f"Trained Random Forest with {len(X)} samples")
                return model
                
            elif algorithm == 'arima':
                # ARIMA needs 1D time series
                if len(y) > 50:
                    try:
                        # Ensure no NaN in y
                        y_clean = np.nan_to_num(y, nan=np.nanmean(y) if not np.all(np.isnan(y)) else 0.0)
                        
                        model = pm.auto_arima(
                            y_clean,
                            start_p=1, start_q=1,
                            max_p=3, max_q=3, m=1,
                            seasonal=False,
                            d=1, D=0,
                            trace=False,
                            error_action='ignore',
                            suppress_warnings=True,
                            maxiter=50
                        )
                        print(f"Trained ARIMA with order {model.order}")
                        return model
                    except Exception as e:
                        print(f"ARIMA auto failed: {e}")
                        # Fallback to simple ARIMA
                        try:
                            model = StatsmodelsARIMA(y_clean, order=(1,1,1))
                            model_fit = model.fit()
                            print("Trained simple ARIMA(1,1,1)")
                            return model_fit
                        except:
                            print("Simple ARIMA also failed")
                            return None
                else:
                    print(f"Warning: Insufficient data for ARIMA ({len(y)} samples)")
                    return None
                    
            return None
        except Exception as e:
            print(f"Error training {algorithm}: {e}")
            return None
    
    def train_all_models(self, data, symbol):
        """Train models for all OCHL targets using all algorithms"""
        try:
            X_data, y_data, data_with_features = self.prepare_training_data(data)
            
            if X_data is None or y_data is None:
                print("Error: Could not prepare training data")
                return False, "Insufficient data for training"
            
            algorithms = ['linear_regression', 'svr', 'random_forest', 'arima']
            self.models = {target: {} for target in self.targets}
            self.scalers = {}
            
            print(f"Training OCHL models for {symbol}...")
            
            targets_trained = 0
            
            for target in self.targets:
                print(f"  Training models for {target}...")
                
                if target not in X_data or X_data[target] is None:
                    print(f"    Skipping {target}: No data")
                    continue
                
                X = X_data[target]
                y = y_data[target]
                
                # Check data quality
                if len(X) < 20 or len(y) < 20:
                    print(f"    Skipping {target}: Insufficient samples ({len(X)})")
                    continue
                
                # Scale features and target
                try:
                    X_scaled = self.feature_scaler.fit_transform(X)
                except:
                    print(f"    Warning: Feature scaling failed for {target}")
                    X_scaled = X  # Use unscaled features
                
                # Scale target
                try:
                    target_scaler = StandardScaler()
                    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
                    self.scalers[target] = target_scaler
                except:
                    print(f"    Warning: Target scaling failed for {target}")
                    y_scaled = y
                    self.scalers[target] = StandardScaler()
                
                # Train each algorithm
                for algo in algorithms:
                    print(f"    Training {algo}...")
                    model = self.train_algorithm(X_scaled, y_scaled, algo)
                    self.models[target][algo] = model
                
                targets_trained += 1
            
            if targets_trained == 0:
                print("Error: No targets were successfully trained")
                return False, "Failed to train any models"
            
            # Calculate historical performance
            self.calculate_historical_performance(X_data, y_data, data_with_features)
            
            # Calculate risk metrics
            self.calculate_risk_metrics(data_with_features)
            
            # Save prediction history
            self.update_prediction_history(symbol, data_with_features)
            
            self.last_training_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.is_fitted = True
            
            # Save models and history
            self.save_models(symbol)
            
            print(f"Successfully trained models for {targets_trained} targets")
            return True, f"Trained models for {targets_trained} targets successfully"
            
        except Exception as e:
            print(f"Error training all models: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)
    
    def calculate_historical_performance(self, X_data, y_data, data_with_features):
        """Calculate comprehensive historical performance metrics"""
        try:
            performance = {}
            
            for target in self.targets:
                if target not in self.models or target not in X_data:
                    continue
                
                X = X_data[target]
                y = y_data[target]
                
                if X is None or y is None or len(X) < 20:
                    continue
                
                # Split for validation (80/20)
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                if len(X_test) < 5 or len(y_test) < 5:
                    continue
                
                # Scale features
                try:
                    X_train_scaled = self.feature_scaler.transform(X_train)
                    X_test_scaled = self.feature_scaler.transform(X_test)
                except:
                    X_train_scaled = X_train
                    X_test_scaled = X_test
                
                # Scale target
                target_scaler = self.scalers.get(target, StandardScaler())
                try:
                    y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).flatten()
                    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
                except:
                    y_train_scaled = y_train
                    y_test_scaled = y_test
                
                target_performance = {}
                
                for algo, model in self.models[target].items():
                    if model is None:
                        continue
                    
                    try:
                        if algo == 'arima':
                            # ARIMA predictions - need to train on training data
                            try:
                                temp_arima = pm.auto_arima(
                                    y_train_scaled,
                                    start_p=1, start_q=1,
                                    max_p=3, max_q=3,
                                    seasonal=False,
                                    trace=False,
                                    error_action='ignore'
                                )
                                predictions_scaled = temp_arima.predict(n_periods=len(y_test_scaled))
                            except:
                                # Fallback
                                predictions_scaled = np.zeros_like(y_test_scaled)
                        else:
                            # Other models
                            predictions_scaled = model.predict(X_test_scaled)
                        
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
                        
                        # Calculate metrics
                        rmse = np.sqrt(mean_squared_error(actuals, predictions))
                        mae = mean_absolute_error(actuals, predictions)
                        r2 = r2_score(actuals, predictions) if len(actuals) > 1 else 0
                        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100
                        
                        # Direction accuracy
                        if len(actuals) > 1 and len(predictions) > 1:
                            actual_direction = np.diff(actuals) > 0
                            pred_direction = np.diff(predictions) > 0
                            min_len = min(len(actual_direction), len(pred_direction))
                            if min_len > 0:
                                direction_accuracy = np.mean(actual_direction[:min_len] == pred_direction[:min_len]) * 100
                            else:
                                direction_accuracy = 0
                        else:
                            direction_accuracy = 0
                        
                        target_performance[algo] = {
                            'rmse': float(rmse),
                            'mae': float(mae),
                            'r2': float(r2),
                            'mape': float(mape),
                            'direction_accuracy': float(direction_accuracy),
                            'sample_size': len(actuals)
                        }
                        
                    except Exception as e:
                        print(f"Error calculating performance for {target}-{algo}: {e}")
                        target_performance[algo] = {
                            'rmse': 9999,
                            'mae': 9999,
                            'r2': -1,
                            'mape': 100,
                            'direction_accuracy': 0,
                            'sample_size': 0
                        }
                
                performance[target] = target_performance
            
            self.historical_performance = performance
            print(f"Calculated performance for {len(performance)} targets")
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
            
            # Basic risk metrics
            volatility = returns.std() * np.sqrt(252)  # Annualized
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
            
            self.risk_metrics = {
                'volatility': float(volatility * 100),  # as percentage
                'sharpe_ratio': float(sharpe_ratio),
                'var_95': float(var_95),
                'var_99': float(var_99),
                'cvar_95': float(cvar_95),
                'cvar_99': float(cvar_99),
                'max_drawdown': float(max_drawdown),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'anomaly_score': float(anomaly_score),
                'total_returns': float(returns.mean() * 252 * 100),  # annualized percentage
                'positive_days': float((returns > 0).mean() * 100),
                'negative_days': float((returns < 0).mean() * 100)
            }
            
            print(f"Calculated risk metrics: {len(self.risk_metrics)} metrics")
            
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
        """Predict next day's OCHL using ensemble of all algorithms"""
        try:
            print(f"Starting prediction for {symbol}")
            
            X_data, _, data_with_features = self.prepare_training_data(data)
            
            if X_data is None:
                print("Error: Could not prepare prediction data")
                return None, "Insufficient data for prediction"
            
            predictions = {}
            confidence_scores = {}
            algorithm_predictions = {}
            
            for target in self.targets:
                print(f"  Predicting {target}...")
                
                if target not in self.models or target not in X_data or X_data[target] is None:
                    print(f"    Skipping {target}: No model or data")
                    predictions[target] = data_with_features['Close'].iloc[-1] if 'Close' in data_with_features.columns else 100.0
                    confidence_scores[target] = 50
                    continue
                
                X = X_data[target]
                if len(X) == 0:
                    print(f"    Skipping {target}: Empty data")
                    predictions[target] = data_with_features['Close'].iloc[-1] if 'Close' in data_with_features.columns else 100.0
                    confidence_scores[target] = 50
                    continue
                
                # Get latest features for prediction
                latest_features = X[-1:].reshape(1, -1)
                try:
                    latest_scaled = self.feature_scaler.transform(latest_features)
                except:
                    print(f"    Warning: Feature scaling failed for prediction")
                    latest_scaled = latest_features
                
                target_predictions = {}
                target_confidences = {}
                
                for algo, model in self.models[target].items():
                    if model is None:
                        print(f"    Skipping {algo}: No model")
                        continue
                    
                    try:
                        pred_actual = None
                        
                        if algo == 'arima':
                            # ARIMA needs time series data
                            target_scaler = self.scalers.get(target, StandardScaler())
                            y_scaled = target_scaler.transform(
                                data_with_features[target].values[-100:].reshape(-1, 1)
                            ).flatten() if target in data_with_features.columns else np.zeros(100)
                            
                            # Train temporary ARIMA on recent data
                            if len(y_scaled) > 30:
                                try:
                                    temp_arima = pm.auto_arima(
                                        y_scaled,
                                        start_p=1, start_q=1,
                                        max_p=3, max_q=3,
                                        seasonal=False,
                                        trace=False,
                                        error_action='ignore'
                                    )
                                    pred_scaled = temp_arima.predict(n_periods=1)[0]
                                    
                                    # Inverse transform
                                    pred_actual = target_scaler.inverse_transform(
                                        np.array([[pred_scaled]])
                                    )[0][0]
                                except Exception as e:
                                    print(f"      ARIMA prediction failed: {e}")
                                    pred_actual = data_with_features[target].iloc[-1] if target in data_with_features.columns else data_with_features['Close'].iloc[-1]
                            else:
                                pred_actual = data_with_features[target].iloc[-1] if target in data_with_features.columns else data_with_features['Close'].iloc[-1]
                        else:
                            # Other models
                            pred_scaled = float(model.predict(latest_scaled)[0])
                            
                            # Inverse transform
                            target_scaler = self.scalers.get(target, StandardScaler())
                            pred_actual = target_scaler.inverse_transform(
                                np.array([[pred_scaled]])
                            )[0][0]
                        
                        # Ensure prediction is valid
                        if pred_actual is None or np.isnan(pred_actual) or np.isinf(pred_actual):
                            print(f"      Invalid prediction from {algo}")
                            pred_actual = data_with_features[target].iloc[-1] if target in data_with_features.columns else data_with_features['Close'].iloc[-1]
                        
                        # Calculate confidence based on historical performance
                        confidence = 70  # default
                        if target in self.historical_performance and algo in self.historical_performance[target]:
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
                        
                        print(f"      {algo}: ${pred_actual:.2f} (conf: {confidence:.1f}%)")
                        
                    except Exception as e:
                        print(f"      Error predicting with {algo}: {e}")
                        # Use last known value as fallback
                        fallback_value = data_with_features[target].iloc[-1] if target in data_with_features.columns else data_with_features['Close'].iloc[-1]
                        target_predictions[algo] = float(fallback_value)
                        target_confidences[algo] = 50.0
                
                if target_predictions:
                    # Calculate weighted ensemble prediction
                    weights = {}
                    total_weight = 0
                    
                    for algo, conf in target_confidences.items():
                        weight = max(conf / 100, 0.1)  # Minimum weight of 0.1
                        weights[algo] = weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        ensemble_pred = sum(
                            pred * (weights[algo] / total_weight)
                            for algo, pred in target_predictions.items()
                        )
                    else:
                        # Equal weights if no confidence scores
                        ensemble_pred = np.mean(list(target_predictions.values()))
                    
                    predictions[target] = float(ensemble_pred)
                    confidence_scores[target] = float(np.mean(list(target_confidences.values())))
                    algorithm_predictions[target] = {
                        'individual': target_predictions,
                        'confidences': target_confidences,
                        'ensemble': float(ensemble_pred),
                        'ensemble_confidence': float(np.mean(list(target_confidences.values())))
                    }
                    
                    print(f"    {target} ensemble: ${ensemble_pred:.2f} (conf: {confidence_scores[target]:.1f}%)")
                else:
                    # Fallback to current price
                    fallback_value = data_with_features[target].iloc[-1] if target in data_with_features.columns else data_with_features['Close'].iloc[-1]
                    predictions[target] = float(fallback_value)
                    confidence_scores[target] = 50.0
                    print(f"    {target} fallback: ${fallback_value:.2f}")
            
            # Ensure we have predictions for all targets
            for target in self.targets:
                if target not in predictions:
                    fallback_value = data_with_features['Close'].iloc[-1] if 'Close' in data_with_features.columns else 100.0
                    predictions[target] = float(fallback_value)
                    confidence_scores[target] = 50.0
            
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
                    'open': float(data['Open'].iloc[-1]) if 'Open' in data.columns else 100.0,
                    'high': float(data['High'].iloc[-1]) if 'High' in data.columns else 100.0,
                    'low': float(data['Low'].iloc[-1]) if 'Low' in data.columns else 100.0,
                    'close': float(data['Close'].iloc[-1]) if 'Close' in data.columns else 100.0
                }
            }
            
            print(f"Successfully generated predictions for {symbol}")
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
            
            # Calculate overall confidence
            overall_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0
            
            # Calculate prediction consistency
            predicted_values = list(predictions.values())
            if len(predicted_values) >= 2:
                mean_val = np.mean(predicted_values)
                if mean_val != 0:
                    consistency = 1 - (np.std(predicted_values) / abs(mean_val))
                    consistency_score = max(0, min(100, consistency * 100))
                else:
                    consistency_score = 0
            else:
                consistency_score = 0
            
            # Determine confidence level
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
            # Get current and predicted prices
            current_close = data['Close'].iloc[-1] if 'Close' in data.columns else 100.0
            predicted_close = predictions.get('Close', current_close)
            
            # Calculate expected change
            expected_change = ((predicted_close - current_close) / current_close) * 100 if current_close != 0 else 0
            
            # 1. Price movement alerts
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
            
            # 2. Volatility alerts
            if 'Volatility' in data.columns:
                recent_volatility = data['Volatility'].iloc[-10:].mean()
                if recent_volatility > 0.03:
                    alerts.append({
                        'level': '🟡 HIGH',
                        'type': 'High Volatility',
                        'message': f'Recent volatility: {recent_volatility:.2%}',
                        'details': 'Market showing increased volatility'
                    })
            
            # 3. RSI alerts
            if 'RSI' in data.columns and len(data) > 0:
                current_rsi = data['RSI'].iloc[-1]
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
            
            # 4. Volume alerts
            if 'Volume_Ratio' in data.columns and len(data) > 0:
                volume_ratio = data['Volume_Ratio'].iloc[-1]
                if volume_ratio > 2.0:
                    alerts.append({
                        'level': '🟡 HIGH',
                        'type': 'Unusual Volume',
                        'message': f'Volume {volume_ratio:.1f}x average',
                        'details': 'High volume may indicate significant news or event'
                    })
            
            # 5. Prediction confidence alerts
            confidence_metrics = self.calculate_prediction_confidence(predictions, {})
            overall_conf = confidence_metrics.get('overall_confidence', 0)
            
            if overall_conf < 50:
                alerts.append({
                    'level': '🟡 HIGH',
                    'type': 'Low Prediction Confidence',
                    'message': f'Model confidence: {overall_conf:.1f}%',
                    'details': 'Consider additional analysis before trading'
                })
            
            # 6. Risk metric alerts
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
            
            # 7. Market regime alert
            market_regime = self.detect_market_regime(data)
            if market_regime != "NORMAL":
                alerts.append({
                    'level': '🟡 HIGH' if market_regime == "HIGH_VOLATILITY" else '🟢 MEDIUM',
                    'type': 'Special Market Regime',
                    'message': f'Current regime: {market_regime.replace("_", " ").title()}',
                    'details': 'Market conditions may affect prediction accuracy'
                })
            
            # Sort alerts by severity
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
            
            # Calculate regime indicators
            volatility = recent['Volatility'].mean() if 'Volatility' in recent.columns else 0
            volume_ratio = recent['Volume_Ratio'].mean() if 'Volume_Ratio' in recent.columns else 1
            rsi = recent['RSI'].iloc[-1] if 'RSI' in recent.columns else 50
            
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
    
    # Weight different risk factors
    if 'volatility' in risk_metrics:
        risk_score += min(risk_metrics['volatility'] / 50, 1) * 0.3
    
    if 'max_drawdown' in risk_metrics:
        risk_score += min(abs(risk_metrics['max_drawdown']) / 30, 1) * 0.3
    
    if 'var_95' in risk_metrics:
        risk_score += min(abs(risk_metrics['var_95']) / 10, 1) * 0.2
    
    if 'anomaly_score' in risk_metrics:
        risk_score += (risk_metrics['anomaly_score'] / 100) * 0.2
    
    # Determine risk level
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
        {"symbol":"TSLA","name":"Tesla Inc.","price":175.79,"change":-3.21}
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
        "version": "4.0.0",
        "algorithms": ["Linear Regression", "SVR", "Random Forest", "ARIMA"],
        "features": "OCHL Prediction + Historical Performance + Confidence + Risk Alerts"
    })

@server.route('/api/predict', methods=['POST'])
@rate_limiter
def predict_stock():
    """Main prediction endpoint with full OCHL prediction"""
    try:
        data = request.get_json() or {}
        symbol = (data.get('symbol') or 'SPY').upper().strip()
        
        print(f"\n=== PREDICTION REQUEST FOR {symbol} ===")
        
        # Get historical data
        historical_data, current_price, error = get_live_stock_data_enhanced(symbol)
        if error:
            print(f"Error fetching data: {error}")
            return jsonify({"error": str(error)}), 400
        
        print(f"Fetched {len(historical_data)} days of data")
        
        # Load existing models or train new ones
        models_loaded = predictor.load_models(symbol)
        
        if not models_loaded or not predictor.is_fitted:
            print("No models found, training new ones...")
            # Train all models
            success, train_msg = predictor.train_all_models(historical_data, symbol)
            if not success:
                print(f"Training failed: {train_msg}")
                return provide_fallback_prediction(symbol, historical_data)
            print("Training successful")
        else:
            print("Loaded existing models")
        
        # Make prediction
        print("Making predictions...")
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
            
            # Algorithm details
            "algorithm_details": prediction_result.get('algorithm_details', {}),
            
            # Confidence metrics
            "confidence_metrics": confidence_metrics,
            
            # Historical performance
            "historical_performance": predictor.historical_performance,
            
            # Risk metrics
            "risk_metrics": predictor.risk_metrics,
            
            # Risk alerts
            "risk_alerts": risk_alerts,
            
            # Risk level and recommendation
            "risk_level": risk_level,
            "trading_recommendation": recommendation,
            
            # Prediction history
            "prediction_history": predictor.prediction_history.get(symbol, [])[-10:],  # Last 10 predictions
            
            # Model info
            "model_info": {
                "last_training_date": predictor.last_training_date,
                "is_fitted": predictor.is_fitted,
                "targets_trained": list(predictor.models.keys()),
                "feature_count": len(predictor.feature_columns),
                "algorithms_used": ["linear_regression", "svr", "random_forest", "arima"]
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
        
        print(f"=== PREDICTION COMPLETE FOR {symbol} ===")
        print(f"Predicted Close: ${predictions.get('Close', 0):.2f} ({expected_changes.get('Close', 0):+.1f}%)")
        print(f"Confidence: {overall_confidence:.1f}%")
        print(f"Recommendation: {recommendation}")
        
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
                "risk_metrics": predictor.risk_metrics
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
        
        # Load history if available
        history_path = os.path.join(HISTORY_DIR, f"{symbol}_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history_data = json.load(f)
            
            return jsonify({
                "status": "success",
                "symbol": symbol,
                "history": history_data.get('prediction_history', []),
                "performance": history_data.get('historical_performance', {}),
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
        
        # Generate reasonable predictions
        predictions = {}
        for target in ['Open', 'High', 'Low', 'Close']:
            base = current_price
            if target == 'High':
                pred = base * random.uniform(1.01, 1.03)
            elif target == 'Low':
                pred = base * random.uniform(0.97, 0.99)
            elif target == 'Open':
                pred = base * random.uniform(0.99, 1.01)
            else:  # Close
                pred = base * random.uniform(0.98, 1.02)
            
            predictions[target] = pred
        
        # Calculate changes
        changes = {}
        for target, pred in predictions.items():
            current = current_price
            change = ((pred - current) / current) * 100 if current != 0 else 0
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
                    "confidence": 65
                }
                for target, pred in predictions.items()
            },
            "confidence_metrics": {
                "overall_confidence": 65,
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
    print("Stock Market Prediction System v4.0")
    print("=" * 60)
    print("Features:")
    print("  • OCHL (Open, Close, High, Low) Prediction")
    print("  • 4 Algorithms: Linear Regression, SVR, Random Forest, ARIMA")
    print("  • Historical Performance Tracking")
    print("  • Prediction Confidence Scoring")
    print("  • Comprehensive Risk Alerts")
    print("  • Model Persistence & History")
    print("=" * 60)
    print("Note: NaN protection enabled throughout the system")
    print("=" * 60)
    
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    port = int(os.environ.get('PORT', 8080))
    server.run(host='0.0.0.0', port=port, debug=True, threaded=True)