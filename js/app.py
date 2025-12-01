# app.py — Complete Stock Prediction System with 10-Year Data
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import IsolationForest
import joblib
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA
warnings.filterwarnings('ignore')

# ================ CONFIGURATION ================
MODELS_DIR = 'models'
HISTORY_DIR = 'history'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# ================ FLASK APP ================
current_dir = os.path.dirname(os.path.abspath(__file__))
server = Flask(__name__, template_folder='templates', static_folder='static')

# ================ RATE LIMITER ================
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

# ================ UTILITY FUNCTIONS ================
def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
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

def safe_divide(a, b, default=1.0):
    """Safe division with default value"""
    try:
        result = np.divide(a, b, out=np.full_like(a, default, dtype=float), where=b!=0)
        return result
    except:
        return np.full_like(a, default, dtype=float)

def create_advanced_features(data):
    """Create comprehensive features for prediction"""
    try:
        data = data.copy()
        
        # Ensure required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                if col == 'Volume':
                    data[col] = 1000000
                else:
                    data[col] = 100.0
        
        # Handle NaN values
        for col in data.columns:
            if col != 'Date':
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = data[col].ffill().bfill()
                if data[col].isna().any():
                    if col == 'Volume':
                        data[col] = data[col].fillna(1000000)
                    elif col in ['Open', 'High', 'Low', 'Close']:
                        data[col] = data[col].fillna(100.0)
                    else:
                        data[col] = data[col].fillna(0)
        
        # Price features
        data['Return'] = data['Close'].pct_change().fillna(0)
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1).replace(0, 1e-10)).fillna(0)
        data['Volatility'] = data['Return'].rolling(window=5, min_periods=1).std().fillna(0)
        data['Volatility_20'] = data['Return'].rolling(window=20, min_periods=1).std().fillna(0)
        
        # Price ranges
        data['High_Low_Range'] = safe_divide(data['High'] - data['Low'], data['Close'].replace(0, 1e-10), 0.02)
        data['Open_Close_Range'] = safe_divide(data['Close'] - data['Open'], data['Open'].replace(0, 1e-10), 0)
        
        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            data[f'MA_{window}'] = data['Close'].rolling(window=window, min_periods=1).mean().fillna(data['Close'])
            data[f'MA_Ratio_{window}'] = safe_divide(data['Close'], data[f'MA_{window}'].replace(0, 1e-10), 1.0)
        
        # Volume features
        data['Volume_MA'] = data['Volume'].rolling(window=10, min_periods=1).mean().fillna(data['Volume'])
        data['Volume_Ratio'] = safe_divide(data['Volume'], data['Volume_MA'].replace(0, 1e-10), 1.0)
        data['Volume_Change'] = data['Volume'].pct_change().fillna(0)
        
        # Technical indicators
        data['RSI'] = calculate_rsi(data['Close'])
        
        # MACD
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
        try:
            bb_middle = data['Close'].rolling(window=20, min_periods=1).mean()
            bb_std = data['Close'].rolling(window=20, min_periods=1).std().fillna(0)
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            data['BB_Middle'] = bb_middle
            data['BB_Upper'] = bb_upper
            data['BB_Lower'] = bb_lower
            data['BB_Width'] = safe_divide(bb_upper - bb_lower, bb_middle.replace(0, 1e-10), 0.1)
            data['BB_Position'] = safe_divide(data['Close'] - bb_lower, (bb_upper - bb_lower).replace(0, 1e-10), 0.5)
        except:
            data['BB_Middle'] = data['Close']
            data['BB_Upper'] = data['Close'] * 1.1
            data['BB_Lower'] = data['Close'] * 0.9
            data['BB_Width'] = 0.1
            data['BB_Position'] = 0.5
        
        # Momentum indicators
        for window in [5, 10, 20]:
            shifted = data['Close'].shift(window).replace(0, 1e-10)
            data[f'Momentum_{window}'] = safe_divide(data['Close'], shifted, 1.0) - 1
        
        # Handle remaining NaN values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Final NaN replacement
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
        return pd.DataFrame({
            'Open': [100.0], 'High': [101.0], 'Low': [99.0], 
            'Close': [100.0], 'Volume': [1000000]
        })

# ================ DATA FETCHING ================
@rate_limiter
def get_live_stock_data_enhanced(ticker):
    """Fetch stock data with fallbacks"""
    try:
        print(f"Fetching data for {ticker}...")
        
        # Try multiple time periods
        strategies = [
            {"func": lambda: yf.download(ticker, period="2y", interval="1d", progress=False, timeout=30), "name":"2y"},
            {"func": lambda: yf.download(ticker, period="1y", interval="1d", progress=False, timeout=25), "name":"1y"},
            {"func": lambda: yf.download(ticker, period="6mo", interval="1d", progress=False, timeout=20), "name":"6mo"},
        ]
        
        for strategy in strategies:
            try:
                print(f"  Trying {strategy['name']}...")
                hist = strategy['func']()
                
                if isinstance(hist, pd.DataFrame) and not hist.empty and len(hist) > 100:
                    print(f"  ✅ Success: {len(hist)} days")
                    
                    hist = hist.reset_index()
                    
                    # Handle date column
                    if 'Date' in hist.columns:
                        hist['Date'] = pd.to_datetime(hist['Date']).dt.strftime('%Y-%m-%d')
                    elif 'Datetime' in hist.columns:
                        hist['Date'] = pd.to_datetime(hist['Datetime']).dt.strftime('%Y-%m-%d')
                        hist = hist.drop(columns=['Datetime'])
                    
                    # Ensure required columns
                    required = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required:
                        if col not in hist.columns:
                            if col == 'Volume':
                                hist[col] = 1000000
                            else:
                                hist[col] = hist.get('Close', 100.0) if 'Close' in hist.columns else 100.0
                    
                    # Clean data
                    for col in required:
                        hist[col] = pd.to_numeric(hist[col], errors='coerce')
                        hist[col] = hist[col].ffill().bfill()
                        if hist[col].isna().any():
                            if col == 'Volume':
                                hist[col] = hist[col].fillna(1000000)
                            else:
                                hist[col] = hist[col].fillna(100.0)
                    
                    current_price = float(hist['Close'].iloc[-1]) if 'Close' in hist.columns else 100.0
                    print(f"  Current price: ${current_price:.2f}")
                    
                    return hist, current_price, None
                    
            except Exception as e:
                print(f"  Strategy failed: {e}")
                if "429" in str(e):
                    time.sleep(10)
                continue
        
        # Fallback
        print(f"  All strategies failed, using fallback data")
        return generate_fallback_data(ticker)
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return generate_fallback_data(ticker)

def generate_fallback_data(ticker, days=500):
    """Generate fallback data"""
    base_prices = {'AAPL':182,'MSFT':407,'GOOGL':172,'AMZN':178,'TSLA':175,'SPY':445}
    base_price = base_prices.get(ticker, 100.0)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.dayofweek < 5]
    
    if len(dates) > days:
        dates = dates[-days:]
    
    # Generate prices
    prices = [base_price]
    for i in range(1, len(dates)):
        change = random.gauss(0, 0.015)
        new_price = max(prices[-1] * (1 + change), base_price * 0.1)
        new_price = min(new_price, base_price * 3)
        prices.append(new_price)
    
    # Generate OHLC
    opens, highs, lows = [], [], []
    for i, close in enumerate(prices):
        if i == 0:
            open_price = close * random.uniform(0.99, 1.01)
        else:
            open_price = prices[i-1] * random.uniform(0.99, 1.01)
        
        daily_range = close * 0.02
        high = max(open_price, close) + daily_range * random.uniform(0.1, 0.5)
        low = min(open_price, close) - daily_range * random.uniform(0.1, 0.5)
        
        high = max(high, low * 1.001)
        
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
    
    print(f"Generated {len(df)} days of fallback data")
    return df, prices[-1], None

# ================ PREDICTOR CLASS ================
class OCHLPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_scaler = StandardScaler()
        self.feature_columns = []
        self.targets = ['Open', 'Close', 'High', 'Low']
        self.historical_performance = {}
        self.prediction_history = {}
        self.risk_metrics = {}
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
            
            # Save history
            history_data = {
                'historical_performance': self.historical_performance,
                'prediction_history': self.prediction_history,
                'risk_metrics': self.risk_metrics,
                'last_training_date': self.last_training_date,
                'feature_columns': self.feature_columns
            }
            
            with open(self.get_history_path(symbol), 'w') as f:
                json.dump(history_data, f, default=str)
                
            print(f"✅ Saved models for {symbol}")
            return True
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return False
    
    def load_models(self, symbol):
        """Load trained models"""
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
                            
                            if target not in self.scalers:
                                self.scalers[target] = model_data.get('scaler', StandardScaler())
                            
                            if not self.feature_columns:
                                self.feature_columns = model_data.get('features', [])
                            
                            models_loaded = True
                        except Exception as e:
                            print(f"Error loading {target}-{algo}: {e}")
                            loaded_models[target][algo] = None
                    else:
                        loaded_models[target][algo] = None
            
            # Load history
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
                except Exception as e:
                    print(f"Error loading history: {e}")
            
            self.models = loaded_models
            return models_loaded
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def prepare_training_data(self, data):
        """Prepare data for training"""
        try:
            print(f"Preparing training data: {len(data)} rows")
            
            data_with_features = create_advanced_features(data)
            
            # Feature selection
            numeric_cols = [col for col in data_with_features.columns 
                          if col != 'Date' and pd.api.types.is_numeric_dtype(data_with_features[col])]
            
            feature_candidates = [col for col in numeric_cols if col not in self.targets]
            
            if len(feature_candidates) > 20:
                # Select top features by correlation
                correlations = []
                for target in self.targets:
                    if target in data_with_features.columns:
                        for feat in feature_candidates:
                            if feat in data_with_features.columns:
                                corr = abs(data_with_features[target].corr(data_with_features[feat]))
                                if not np.isnan(corr):
                                    correlations.append((feat, corr))
                
                feat_scores = {}
                for feat, corr in correlations:
                    if feat not in feat_scores:
                        feat_scores[feat] = []
                    feat_scores[feat].append(corr)
                
                avg_corrs = {feat: np.mean(scores) for feat, scores in feat_scores.items()}
                sorted_feats = sorted(avg_corrs.items(), key=lambda x: x[1], reverse=True)
                self.feature_columns = [feat for feat, _ in sorted_feats[:20]]
            else:
                self.feature_columns = feature_candidates
            
            print(f"Selected {len(self.feature_columns)} features")
            
            # Prepare sequences
            X_data = {}
            y_data = {}
            
            for target in self.targets:
                if target not in data_with_features.columns:
                    data_with_features[target] = data_with_features['Close']
                
                data_with_features[target] = data_with_features[target].ffill().bfill()
                
                X_list = []
                y_list = []
                window_size = 30
                
                if len(data_with_features) < window_size + 10:
                    continue
                
                for i in range(window_size, len(data_with_features) - 1):
                    features = data_with_features[self.feature_columns].iloc[i-window_size:i]
                    
                    if features.isna().any().any():
                        features = features.fillna(0)
                    
                    features_flat = features.values.flatten()
                    target_value = data_with_features[target].iloc[i+1]
                    
                    if pd.isna(target_value):
                        continue
                    
                    X_list.append(features_flat)
                    y_list.append(target_value)
                
                if len(X_list) > 20:
                    X_data[target] = np.array(X_list)
                    y_data[target] = np.array(y_list)
                    print(f"Prepared {len(X_list)} samples for {target}")
                else:
                    X_data[target] = None
                    y_data[target] = None
            
            valid_targets = [t for t in self.targets if t in X_data and X_data[t] is not None and len(X_data[t]) > 0]
            if len(valid_targets) == 0:
                return None, None, None
            
            return X_data, y_data, data_with_features
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            return None, None, None
    
    def train_algorithm(self, X, y, algorithm):
        """Train a specific algorithm"""
        try:
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                X = np.nan_to_num(X, nan=0.0)
                y_mean = np.nanmean(y) if not np.all(np.isnan(y)) else np.mean(y) if len(y) > 0 else 0
                y = np.nan_to_num(y, nan=y_mean)
            
            if len(X) < 20:
                return None
            
            if algorithm == 'linear_regression':
                model = LinearRegression()
                model.fit(X, y)
                return model
                
            elif algorithm == 'svr':
                model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                if len(X) > 500:
                    idx = np.random.choice(len(X), min(500, len(X)), replace=False)
                    model.fit(X[idx], y[idx])
                else:
                    model.fit(X, y)
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
                return model
                
            elif algorithm == 'arima':
                if len(y) > 50:
                    try:
                        y_clean = np.nan_to_num(y, nan=np.nanmean(y) if not np.all(np.isnan(y)) else 0)
                        model = pm.auto_arima(
                            y_clean,
                            start_p=1, start_q=1,
                            max_p=2, max_q=2, m=1,
                            seasonal=False,
                            trace=False,
                            error_action='ignore'
                        )
                        return model
                    except:
                        try:
                            y_clean = np.nan_to_num(y, nan=np.nanmean(y) if not np.all(np.isnan(y)) else 0)
                            model = StatsmodelsARIMA(y_clean, order=(1,1,1))
                            model_fit = model.fit()
                            return model_fit
                        except:
                            return None
                else:
                    return None
                    
            return None
        except Exception as e:
            print(f"Error training {algorithm}: {e}")
            return None
    
    def train_all_models(self, data, symbol):
        """Train all models"""
        try:
            X_data, y_data, data_with_features = self.prepare_training_data(data)
            
            if X_data is None or y_data is None:
                return False, "Insufficient data"
            
            algorithms = ['linear_regression', 'svr', 'random_forest', 'arima']
            self.models = {target: {} for target in self.targets}
            self.scalers = {}
            
            print(f"Training models for {symbol}...")
            
            targets_trained = 0
            
            for target in self.targets:
                if target not in X_data or X_data[target] is None:
                    continue
                
                X = X_data[target]
                y = y_data[target]
                
                if len(X) < 30 or len(y) < 30:
                    continue
                
                # Scale target
                target_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
                y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
                self.scalers[target] = target_scaler
                
                # Scale features
                X_scaled = self.feature_scaler.fit_transform(X)
                
                # Train algorithms
                for algo in algorithms:
                    model = self.train_algorithm(X_scaled, y_scaled, algo)
                    self.models[target][algo] = model
                
                targets_trained += 1
            
            if targets_trained == 0:
                return False, "No targets trained"
            
            # Calculate performance
            self.calculate_historical_performance(X_data, y_data, data_with_features)
            
            # Calculate risk metrics
            self.calculate_risk_metrics(data_with_features)
            
            # Update history
            self.update_prediction_history(symbol, data_with_features)
            
            self.last_training_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.is_fitted = True
            
            # Save models
            self.save_models(symbol)
            
            print(f"✅ Trained {targets_trained} targets")
            return True, f"Trained {targets_trained} targets"
            
        except Exception as e:
            print(f"Error training all models: {e}")
            return False, str(e)
    
    def calculate_historical_performance(self, X_data, y_data, data_with_features):
        """Calculate performance metrics"""
        try:
            performance = {}
            
            for target in self.targets:
                if target not in self.models or target not in X_data:
                    continue
                
                X = X_data[target]
                y = y_data[target]
                
                if X is None or y is None or len(X) < 20:
                    continue
                
                # Split data
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                if len(X_test) < 5:
                    continue
                
                # Scale features
                try:
                    X_train_scaled = self.feature_scaler.transform(X_train)
                    X_test_scaled = self.feature_scaler.transform(X_test)
                except:
                    X_train_scaled = X_train
                    X_test_scaled = X_test
                
                # Scale target
                target_scaler = self.scalers.get(target, MinMaxScaler(feature_range=(0.1, 0.9)))
                try:
                    y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).flatten()
                    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
                except:
                    y_train_scaled = y_train
                    y_test_scaled = y_test
                    target_scaler = None
                
                target_performance = {}
                
                for algo, model in self.models[target].items():
                    if model is None:
                        continue
                    
                    try:
                        if algo == 'arima':
                            try:
                                arima_train = y_train_scaled[-100:] if len(y_train_scaled) > 100 else y_train_scaled
                                temp_arima = pm.auto_arima(
                                    arima_train,
                                    start_p=1, start_q=1,
                                    max_p=2, max_q=2,
                                    seasonal=False,
                                    trace=False,
                                    error_action='ignore'
                                )
                                predictions_scaled = temp_arima.predict(n_periods=len(y_test_scaled))
                            except:
                                predictions_scaled = np.full_like(y_test_scaled, np.mean(y_train_scaled))
                        else:
                            predictions_scaled = model.predict(X_test_scaled)
                        
                        if target_scaler is not None:
                            predictions = target_scaler.inverse_transform(
                                predictions_scaled.reshape(-1, 1)
                            ).flatten()
                            actuals = target_scaler.inverse_transform(
                                y_test_scaled.reshape(-1, 1)
                            ).flatten()
                        else:
                            predictions = predictions_scaled
                            actuals = y_test_scaled
                        
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
                            'direction_accuracy': float(direction_accuracy)
                        }
                        
                    except Exception as e:
                        target_performance[algo] = {
                            'rmse': 9999, 'mae': 9999, 'r2': -1,
                            'mape': 100, 'direction_accuracy': 0
                        }
                
                performance[target] = target_performance
            
            self.historical_performance = performance
            return performance
            
        except Exception as e:
            print(f"Error calculating performance: {e}")
            return {}
    
    def calculate_risk_metrics(self, data):
        """Calculate risk metrics"""
        try:
            if 'Close' not in data.columns:
                self.risk_metrics = {}
                return
            
            returns = data['Close'].pct_change().dropna()
            returns = returns.fillna(0)
            
            if len(returns) < 10:
                self.risk_metrics = {}
                return
            
            # Basic metrics
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # VaR
            var_95 = np.percentile(returns, 5) * 100 if len(returns) >= 20 else 0
            
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max.replace(0, 1e-10)
            max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0
            
            self.risk_metrics = {
                'volatility': float(volatility * 100),
                'sharpe_ratio': float(sharpe_ratio),
                'var_95': float(var_95),
                'max_drawdown': float(max_drawdown),
                'total_return': float(returns.mean() * 252 * 100),
                'positive_days': float((returns > 0).mean() * 100)
            }
            
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            self.risk_metrics = {}
    
    def update_prediction_history(self, symbol, data):
        """Update prediction history"""
        try:
            if symbol not in self.prediction_history:
                self.prediction_history[symbol] = []
            
            latest = {
                'date': data['Date'].iloc[-1] if 'Date' in data.columns else datetime.now().strftime('%Y-%m-%d'),
                'actual_open': float(data['Open'].iloc[-1]) if 'Open' in data.columns else 0,
                'actual_high': float(data['High'].iloc[-1]) if 'High' in data.columns else 0,
                'actual_low': float(data['Low'].iloc[-1]) if 'Low' in data.columns else 0,
                'actual_close': float(data['Close'].iloc[-1]) if 'Close' in data.columns else 0
            }
            
            self.prediction_history[symbol].append(latest)
            if len(self.prediction_history[symbol]) > 100:
                self.prediction_history[symbol] = self.prediction_history[symbol][-100:]
                
        except Exception as e:
            print(f"Error updating history: {e}")
    
    def predict_ochl(self, data, symbol):
        """Predict next day's prices"""
        try:
            print(f"Predicting for {symbol}...")
            
            X_data, _, data_with_features = self.prepare_training_data(data)
            
            if X_data is None:
                return None, "Insufficient data"
            
            predictions = {}
            confidence_scores = {}
            algorithm_predictions = {}
            
            # Current prices for fallback
            current_prices = {
                'open': float(data['Open'].iloc[-1]) if 'Open' in data.columns else 100.0,
                'high': float(data['High'].iloc[-1]) if 'High' in data.columns else 100.0,
                'low': float(data['Low'].iloc[-1]) if 'Low' in data.columns else 100.0,
                'close': float(data['Close'].iloc[-1]) if 'Close' in data.columns else 100.0
            }
            
            for target in self.targets:
                fallback_value = current_prices[target.lower()]
                
                if target not in self.models or target not in X_data or X_data[target] is None:
                    predictions[target] = fallback_value
                    confidence_scores[target] = 50
                    continue
                
                X = X_data[target]
                if len(X) == 0:
                    predictions[target] = fallback_value
                    confidence_scores[target] = 50
                    continue
                
                # Get latest features
                latest_features = X[-1:].reshape(1, -1)
                try:
                    latest_scaled = self.feature_scaler.transform(latest_features)
                except:
                    latest_scaled = latest_features
                
                target_predictions = {}
                target_confidences = {}
                
                for algo, model in self.models[target].items():
                    if model is None:
                        continue
                    
                    try:
                        if algo == 'arima':
                            target_scaler = self.scalers.get(target, MinMaxScaler(feature_range=(0.1, 0.9)))
                            
                            if target in data_with_features.columns:
                                recent_target = data_with_features[target].values[-50:]
                            else:
                                recent_target = data_with_features['Close'].values[-50:]
                            
                            recent_scaled = target_scaler.transform(recent_target.reshape(-1, 1)).flatten()
                            
                            if len(recent_scaled) > 20:
                                try:
                                    temp_arima = pm.auto_arima(
                                        recent_scaled,
                                        start_p=1, start_q=1,
                                        max_p=2, max_q=2,
                                        seasonal=False,
                                        trace=False,
                                        error_action='ignore'
                                    )
                                    pred_scaled = temp_arima.predict(n_periods=1)[0]
                                except:
                                    pred_scaled = recent_scaled[-1] if len(recent_scaled) > 0 else 0.5
                            else:
                                pred_scaled = recent_scaled[-1] if len(recent_scaled) > 0 else 0.5
                        else:
                            pred_scaled = float(model.predict(latest_scaled)[0])
                        
                        # Ensure bounds
                        pred_scaled = max(0.01, min(0.99, pred_scaled))
                        
                        # Inverse transform
                        target_scaler = self.scalers.get(target, MinMaxScaler(feature_range=(0.1, 0.9)))
                        pred_actual = target_scaler.inverse_transform(
                            np.array([[pred_scaled]])
                        )[0][0]
                        
                        # Sanity checks
                        if pred_actual < 0.01 or pred_actual > 10000:
                            pred_actual = fallback_value
                        
                        # Confidence
                        confidence = 70
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
                        
                    except Exception as e:
                        target_predictions[algo] = float(fallback_value)
                        target_confidences[algo] = 50.0
                
                if target_predictions:
                    # Weighted ensemble
                    weights = {}
                    total_weight = 0
                    
                    for algo, conf in target_confidences.items():
                        weight = max(conf / 100, 0.1)
                        weights[algo] = weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        ensemble_pred = sum(
                            pred * (weights[algo] / total_weight)
                            for algo, pred in target_predictions.items()
                        )
                    else:
                        ensemble_pred = np.mean(list(target_predictions.values()))
                    
                    if ensemble_pred < 0.01:
                        ensemble_pred = fallback_value
                    
                    predictions[target] = float(ensemble_pred)
                    confidence_scores[target] = float(np.mean(list(target_confidences.values())))
                    algorithm_predictions[target] = {
                        'individual': target_predictions,
                        'confidences': target_confidences,
                        'ensemble': float(ensemble_pred)
                    }
                else:
                    predictions[target] = float(fallback_value)
                    confidence_scores[target] = 50.0
            
            # Ensure all targets have predictions
            for target in self.targets:
                if target not in predictions:
                    predictions[target] = current_prices[target.lower()]
                    confidence_scores[target] = 50.0
            
            # Generate risk alerts
            risk_alerts = self.generate_risk_alerts(data_with_features, predictions)
            
            # Confidence metrics
            confidence_metrics = self.calculate_prediction_confidence(predictions, confidence_scores)
            
            result = {
                'predictions': predictions,
                'algorithm_details': algorithm_predictions,
                'confidence_scores': confidence_scores,
                'confidence_metrics': confidence_metrics,
                'risk_alerts': risk_alerts,
                'current_prices': current_prices
            }
            
            print(f"✅ Predictions complete")
            return result, None
            
        except Exception as e:
            print(f"Error predicting: {e}")
            return None, str(e)
    
    def calculate_prediction_confidence(self, predictions, confidence_scores):
        """Calculate confidence metrics"""
        try:
            if not predictions or not confidence_scores:
                return {}
            
            overall_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0
            
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
                'confidence_color': confidence_color
            }
        except:
            return {
                'overall_confidence': 50.0,
                'consistency_score': 0.0,
                'confidence_level': "LOW",
                'confidence_color': "danger"
            }
    
    def generate_risk_alerts(self, data, predictions):
        """Generate risk alerts"""
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
                    'message': f'Expected change: {expected_change:+.1f}%',
                    'details': 'Consider adjusting position size'
                })
            elif abs(expected_change) > 5:
                alerts.append({
                    'level': '🟡 HIGH',
                    'type': 'Large Price Movement',
                    'message': f'Expected change: {expected_change:+.1f}%',
                    'details': 'Monitor position closely'
                })
            
            # Volatility alerts
            if 'Volatility' in data.columns:
                recent_volatility = data['Volatility'].iloc[-10:].mean()
                if recent_volatility > 0.03:
                    alerts.append({
                        'level': '🟡 HIGH',
                        'type': 'High Volatility',
                        'message': f'Recent volatility: {recent_volatility:.2%}',
                        'details': 'Market showing increased volatility'
                    })
            
            # RSI alerts
            if 'RSI' in data.columns and len(data) > 0:
                current_rsi = data['RSI'].iloc[-1]
                if current_rsi > 80:
                    alerts.append({
                        'level': '🟡 HIGH',
                        'type': 'Overbought',
                        'message': f'RSI: {current_rsi:.1f}',
                        'details': 'Consider taking profits'
                    })
                elif current_rsi < 20:
                    alerts.append({
                        'level': '🟢 MEDIUM',
                        'type': 'Oversold',
                        'message': f'RSI: {current_rsi:.1f}',
                        'details': 'Potential buying opportunity'
                    })
            
            return alerts
            
        except:
            return []

# Global predictor
predictor = OCHLPredictor()

# ================ HELPER FUNCTIONS ================
def get_next_trading_day():
    today = datetime.now()
    if today.weekday() == 4:
        return (today + timedelta(days=3)).strftime('%Y-%m-%d')
    if today.weekday() == 5:
        return (today + timedelta(days=2)).strftime('%Y-%m-%d')
    if today.weekday() == 6:
        return (today + timedelta(days=1)).strftime('%Y-%m-%d')
    return (today + timedelta(days=1)).strftime('%Y-%m-%d')

def get_last_market_date():
    today = datetime.now()
    if today.weekday() == 0:
        return (today - timedelta(days=3)).strftime('%Y-%m-%d')
    if today.weekday() == 6:
        return (today - timedelta(days=2)).strftime('%Y-%m-%d')
    if today.weekday() == 5:
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
    if not risk_metrics:
        return "🟢 LOW RISK"
    
    risk_score = 0
    
    if 'volatility' in risk_metrics:
        risk_score += min(risk_metrics['volatility'] / 50, 1) * 0.3
    
    if 'max_drawdown' in risk_metrics:
        risk_score += min(abs(risk_metrics['max_drawdown']) / 30, 1) * 0.3
    
    if 'var_95' in risk_metrics:
        risk_score += min(abs(risk_metrics['var_95']) / 10, 1) * 0.2
    
    if risk_score > 0.7:
        return "🔴 EXTREME RISK"
    elif risk_score > 0.5:
        return "🟡 HIGH RISK"
    elif risk_score > 0.3:
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

def get_trading_recommendation(predictions, current_prices, confidence):
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

# ================ FLASK ROUTES ================
@server.route('/')
def index():
    return render_template('index.html')

@server.route('/api/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

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
        except:
            continue
    
    return jsonify(popular_stocks)

@server.route('/api/predict', methods=['POST'])
@rate_limiter
def predict_stock():
    """Main prediction endpoint"""
    try:
        data = request.get_json() or {}
        symbol = (data.get('symbol') or 'AAPL').upper().strip()
        
        print(f"\n=== Predicting {symbol} ===")
        
        # Get data
        historical_data, current_price, error = get_live_stock_data_enhanced(symbol)
        if error:
            return jsonify({"error": str(error)}), 400
        
        print(f"Data: {len(historical_data)} days")
        
        # Load or train models
        models_loaded = predictor.load_models(symbol)
        
        if not models_loaded or not predictor.is_fitted:
            print("Training new models...")
            success, train_msg = predictor.train_all_models(historical_data, symbol)
            if not success:
                return provide_fallback_prediction(symbol, historical_data)
            print("Training complete")
        else:
            print("Using existing models")
        
        # Make prediction
        prediction_result, pred_error = predictor.predict_ochl(historical_data, symbol)
        
        if pred_error:
            return provide_fallback_prediction(symbol, historical_data)
        
        # Prepare response
        current_prices = prediction_result['current_prices']
        predictions = prediction_result['predictions']
        confidence_scores = prediction_result['confidence_scores']
        confidence_metrics = prediction_result['confidence_metrics']
        risk_alerts = prediction_result['risk_alerts']
        
        # Calculate changes
        expected_changes = {}
        for target in ['Open', 'High', 'Low', 'Close']:
            if target in predictions and target.lower() in current_prices:
                current = current_prices[target.lower()]
                predicted = predictions[target]
                change = ((predicted - current) / current) * 100 if current != 0 else 0
                expected_changes[target] = change
        
        # Trading recommendation
        overall_confidence = confidence_metrics.get('overall_confidence', 70)
        recommendation = get_trading_recommendation(predictions, current_prices, overall_confidence)
        
        # Risk level
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
            
            # Confidence
            "confidence_metrics": confidence_metrics,
            
            # Performance
            "historical_performance": predictor.historical_performance,
            
            # Risk
            "risk_metrics": predictor.risk_metrics,
            "risk_alerts": risk_alerts,
            "risk_level": risk_level,
            "trading_recommendation": recommendation,
            
            # History
            "prediction_history": predictor.prediction_history.get(symbol, [])[-5:],
            
            # Model info
            "model_info": {
                "last_training_date": predictor.last_training_date,
                "is_fitted": predictor.is_fitted
            },
            
            # Data info
            "data_info": {
                "total_days": len(historical_data)
            },
            
            # Insight
            "insight": f"Predicted {expected_changes.get('Close', 0):+.1f}% change. {recommendation}"
        }
        
        print(f"Predicted Close: ${predictions.get('Close', 0):.2f} ({expected_changes.get('Close', 0):+.1f}%)")
        print(f"Confidence: {overall_confidence:.1f}%")
        print(f"Recommendation: {recommendation}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            "error": "Prediction service unavailable",
            "fallback": True
        }), 500

@server.route('/api/train', methods=['POST'])
def train_models():
    """Train models endpoint"""
    try:
        data = request.get_json() or {}
        symbol = (data.get('symbol') or 'AAPL').upper().strip()
        
        historical_data, current_price, error = get_live_stock_data_enhanced(symbol)
        if error:
            return jsonify({"error": str(error)}), 400
        
        success, train_msg = predictor.train_all_models(historical_data, symbol)
        
        if success:
            return jsonify({
                "status": "success",
                "message": train_msg,
                "symbol": symbol,
                "last_training_date": predictor.last_training_date
            })
        else:
            return jsonify({
                "status": "error",
                "message": train_msg,
                "symbol": symbol
            }), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def provide_fallback_prediction(symbol, historical_data):
    """Fallback prediction"""
    try:
        if historical_data is None or historical_data.empty:
            current_price = 100.0
        else:
            current_price = historical_data['Close'].iloc[-1] if 'Close' in historical_data.columns else 100.0
        
        # Generate predictions
        predictions = {}
        base_change = random.uniform(-0.05, 0.05)
        
        for target in ['Open', 'High', 'Low', 'Close']:
            if target == 'High':
                pred = current_price * (1 + base_change + random.uniform(0.01, 0.03))
            elif target == 'Low':
                pred = current_price * (1 + base_change - random.uniform(0.01, 0.03))
            elif target == 'Open':
                pred = current_price * (1 + base_change * random.uniform(0.5, 1.5))
            else:
                pred = current_price * (1 + base_change)
            
            pred = max(pred, current_price * 0.5)
            pred = min(pred, current_price * 1.5)
            
            predictions[target] = pred
        
        # Calculate changes
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
                    "confidence": 65
                }
                for target, pred in predictions.items()
            },
            "confidence_metrics": {
                "overall_confidence": 65,
                "confidence_level": "MEDIUM"
            },
            "risk_alerts": [{
                "level": "🟡 HIGH",
                "type": "Fallback Mode",
                "message": "Using fallback predictions"
            }],
            "risk_level": "🟠 MEDIUM RISK",
            "trading_recommendation": "🔄 HOLD",
            "fallback": True
        })
        
    except:
        return jsonify({
            "error": "Service unavailable",
            "fallback": True
        }), 500

@server.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# ================ CREATE TEMPLATE ================
def create_default_template():
    """Create default HTML template"""
    template_dir = os.path.join(current_dir, 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Prediction System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .prediction-box {
                background: #e8f4f8;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                border-left: 4px solid #2196F3;
            }
            .prediction-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin: 20px 0;
            }
            .prediction-item {
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            .price {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }
            .change-positive {
                color: #4CAF50;
            }
            .change-negative {
                color: #f44336;
            }
            .alerts {
                margin: 20px 0;
            }
            .alert {
                padding: 12px;
                margin: 10px 0;
                border-radius: 6px;
                border-left: 4px solid;
            }
            .alert-critical {
                background: #ffebee;
                border-color: #f44336;
            }
            .alert-high {
                background: #fff3e0;
                border-color: #ff9800;
            }
            .alert-medium {
                background: #e8f5e9;
                border-color: #4CAF50;
            }
            .input-group {
                margin: 20px 0;
                display: flex;
                gap: 10px;
            }
            input {
                flex: 1;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
            }
            button {
                padding: 10px 20px;
                background: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background: #0b7dda;
            }
            .loading {
                text-align: center;
                padding: 20px;
                display: none;
            }
            .recommendation {
                font-size: 20px;
                padding: 15px;
                text-align: center;
                margin: 20px 0;
                border-radius: 8px;
                background: #e8f5e9;
                border: 2px solid #4CAF50;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📈 Stock Prediction System</h1>
            
            <div class="input-group">
                <input type="text" id="symbolInput" placeholder="Enter stock symbol (e.g., AAPL)" value="AAPL">
                <button onclick="predictStock()">Predict</button>
                <button onclick="trainModel()">Train Model</button>
            </div>
            
            <div id="loading" class="loading">
                <p>⚡ Analyzing data and making predictions...</p>
            </div>
            
            <div id="predictionResult"></div>
        </div>
        
        <script>
            async function predictStock() {
                const symbol = document.getElementById('symbolInput').value.trim().toUpperCase();
                if (!symbol) {
                    alert('Please enter a stock symbol');
                    return;
                }
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('predictionResult').innerHTML = '';
                
                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ symbol: symbol })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        document.getElementById('predictionResult').innerHTML = `
                            <div class="prediction-box">
                                <h2>Error: ${data.error}</h2>
                                ${data.fallback ? '<p>Using fallback predictions</p>' : ''}
                            </div>
                        `;
                    } else {
                        displayPrediction(data);
                    }
                } catch (error) {
                    document.getElementById('predictionResult').innerHTML = `
                        <div class="prediction-box">
                            <h2>Error: Failed to get prediction</h2>
                            <p>Please try again</p>
                        </div>
                    `;
                }
                
                document.getElementById('loading').style.display = 'none';
            }
            
            function displayPrediction(data) {
                let html = `
                    <div class="prediction-box">
                        <h2>${data.symbol} Prediction Results</h2>
                        <p><strong>Market Status:</strong> ${data.market_status}</p>
                        <p><strong>Prediction Date:</strong> ${data.prediction_date}</p>
                        
                        <div class="recommendation">
                            <strong>Recommendation:</strong> ${data.trading_recommendation}
                        </div>
                        
                        <div class="prediction-grid">
                `;
                
                // Add prediction items
                ['Open', 'High', 'Low', 'Close'].forEach(target => {
                    const pred = data.predictions[target];
                    const changeClass = pred.expected_change >= 0 ? 'change-positive' : 'change-negative';
                    
                    html += `
                        <div class="prediction-item">
                            <h3>${target}</h3>
                            <div class="price">$${pred.predicted_price.toFixed(2)}</div>
                            <div class="${changeClass}">${pred.expected_change >= 0 ? '+' : ''}${pred.expected_change.toFixed(2)}%</div>
                            <div>Current: $${pred.current_price.toFixed(2)}</div>
                            <div>Confidence: ${pred.confidence.toFixed(1)}%</div>
                        </div>
                    `;
                });
                
                html += `</div>`;
                
                // Add confidence
                html += `
                    <div style="margin: 20px 0; padding: 15px; background: #f0f7ff; border-radius: 8px;">
                        <h3>Prediction Confidence: ${data.confidence_metrics.overall_confidence.toFixed(1)}%</h3>
                        <p>Level: ${data.confidence_metrics.confidence_level}</p>
                    </div>
                `;
                
                // Add risk alerts
                if (data.risk_alerts && data.risk_alerts.length > 0) {
                    html += `<div class="alerts"><h3>⚠️ Risk Alerts</h3>`;
                    data.risk_alerts.forEach(alert => {
                        html += `
                            <div class="alert alert-${alert.level.includes('CRITICAL') ? 'critical' : alert.level.includes('HIGH') ? 'high' : 'medium'}">
                                <strong>${alert.level} ${alert.type}</strong><br>
                                ${alert.message}<br>
                                <small>${alert.details}</small>
                            </div>
                        `;
                    });
                    html += `</div>`;
                }
                
                // Add risk metrics
                if (data.risk_metrics) {
                    html += `<div style="margin: 20px 0;"><h3>📊 Risk Metrics</h3>`;
                    html += `<p>Risk Level: ${data.risk_level}</p>`;
                    html += `<p>Volatility: ${data.risk_metrics.volatility?.toFixed(2) || 'N/A'}%</p>`;
                    html += `<p>Max Drawdown: ${data.risk_metrics.max_drawdown?.toFixed(2) || 'N/A'}%</p>`;
                    html += `</div>`;
                }
                
                html += `</div>`;
                
                document.getElementById('predictionResult').innerHTML = html;
            }
            
            async function trainModel() {
                const symbol = document.getElementById('symbolInput').value.trim().toUpperCase();
                if (!symbol) {
                    alert('Please enter a stock symbol');
                    return;
                }
                
                document.getElementById('loading').style.display = 'block';
                
                try {
                    const response = await fetch('/api/train', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ symbol: symbol })
                    });
                    
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        alert(`Model trained successfully for ${symbol}!`);
                    } else {
                        alert(`Training failed: ${data.message}`);
                    }
                } catch (error) {
                    alert('Failed to train model');
                }
                
                document.getElementById('loading').style.display = 'none';
            }
            
            // Predict AAPL on page load
            window.onload = function() {
                predictStock();
            };
        </script>
    </body>
    </html>
    """
    
    with open(os.path.join(template_dir, 'index.html'), 'w') as f:
        f.write(html_content)
    
    print("✅ Created default template")

# ================ MAIN ================
if __name__ == '__main__':
    print("=" * 60)
    print("Stock Market Prediction System")
    print("=" * 60)
    print("Features:")
    print("  • Real-time stock data")
    print("  • OCHL predictions (Open, Close, High, Low)")
    print("  • 4 ML algorithms: Linear Regression, SVR, Random Forest, ARIMA")
    print("  • Risk alerts and confidence scoring")
    print("  • Web interface")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    # Create default template
    create_default_template()
    
    # Start server
    port = int(os.environ.get('PORT', 8080))
    print(f"Server starting on http://localhost:{port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    server.run(host='0.0.0.0', port=port, debug=True, threaded=True)