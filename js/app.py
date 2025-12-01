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
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except:
        return pd.Series([50] * len(prices), index=prices.index)

def calculate_bollinger_bands(data, window=20):
    try:
        sma = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return sma, upper_band, lower_band
    except:
        return data['Close'], data['Close'] * 1.1, data['Close'] * 0.9

def create_advanced_features(data):
    """Create comprehensive features for OCHL prediction"""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            flat = pd.DataFrame()
            for col in data.columns:
                flat[col[0]] = data[col]
            data = flat

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        data = data.copy()
        required_columns = ['Open','High','Low','Close','Volume']
        
        # Ensure all required columns exist
        for col in required_columns:
            if col not in data.columns:
                if col == 'Volume':
                    data[col] = 1000000
                else:
                    data[col] = data['Close'] if 'Close' in data.columns else 100.0
        
        # Convert to numeric
        for col in required_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].fillna(method='ffill').fillna(method='bfill')

        # Price features
        data['Return'] = data['Close'].pct_change().fillna(0)
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1)).fillna(0)
        data['Volatility'] = data['Return'].rolling(window=5, min_periods=1).std().fillna(0)
        data['High_Low_Range'] = (data['High'] - data['Low']) / data['Close'].replace(0, 1)
        data['Open_Close_Range'] = (data['Close'] - data['Open']) / data['Open'].replace(0, 1)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            data[f'MA_{window}'] = data['Close'].rolling(window=window, min_periods=1).mean()
            data[f'MA_Ratio_{window}'] = data['Close'] / data[f'MA_{window}'].replace(0, 1)
        
        # Volume features
        data['Volume_MA'] = data['Volume'].rolling(window=10, min_periods=1).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA'].replace(0, 1)
        data['Volume_Change'] = data['Volume'].pct_change().fillna(0)
        
        # Technical indicators
        data['RSI'] = calculate_rsi(data['Close'])
        data['MACD'] = data['Close'].ewm(span=12, min_periods=1).mean() - data['Close'].ewm(span=26, min_periods=1).mean()
        data['MACD_Signal'] = data['MACD'].ewm(span=9, min_periods=1).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Bollinger Bands
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(data)
        data['BB_Middle'] = bb_middle
        data['BB_Upper'] = bb_upper
        data['BB_Lower'] = bb_lower
        data['BB_Width'] = (bb_upper - bb_lower) / bb_middle.replace(0, 1)
        data['BB_Position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower).replace(0, 1)
        
        # Momentum indicators
        for window in [5, 10, 20]:
            data[f'Momentum_{window}'] = (data['Close'] / data['Close'].shift(window).replace(0, 1)) - 1
        
        # Price patterns
        data['Gap_Up'] = (data['Open'] > data['Close'].shift(1) * 1.01).astype(int)
        data['Gap_Down'] = (data['Open'] < data['Close'].shift(1) * 0.99).astype(int)
        data['New_High'] = (data['Close'] == data['High'].rolling(window=20).max()).astype(int)
        data['New_Low'] = (data['Close'] == data['Low'].rolling(window=20).min()).astype(int)
        
        # Fill NaN values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Final check for any remaining NaN
        for col in data.columns:
            if data[col].isna().any():
                if col in ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram']:
                    data[col] = data[col].fillna(0)
                elif 'Ratio' in col or 'Momentum' in col:
                    data[col] = data[col].fillna(1)
                elif 'Volatility' in col or 'Return' in col:
                    data[col] = data[col].fillna(0)
                else:
                    data[col] = data[col].fillna(100.0 if col != 'Volume' else 1000000)
        
        return data
    except Exception as e:
        print(f"Feature creation error: {e}")
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame({
            'Open':[100.0],'High':[101.0],'Low':[99.0],'Close':[100.0],'Volume':[1000000]
        })

def detect_anomalies(data):
    """Detect anomalies in the data using Isolation Forest"""
    try:
        features = ['Return', 'Volatility', 'Volume_Ratio', 'RSI', 'High_Low_Range']
        available_features = [f for f in features if f in data.columns]
        
        if len(available_features) < 3:
            return pd.Series([0] * len(data), index=data.index)
        
        X = data[available_features].values
        
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
    try:
        strategies = [
            {"func": lambda: yf.download(ticker, period="10y", interval="1d", progress=False, timeout=30), "name":"10y"},
            {"func": lambda: yf.Ticker(ticker).history(period="5y", interval="1d"), "name":"ticker.history"},
            {"func": lambda: yf.download(ticker, period="1y", interval="1d", progress=False, timeout=25), "name":"1y"},
        ]
        
        for strategy in strategies:
            try:
                hist = strategy['func']()
                if isinstance(hist, pd.DataFrame) and (not hist.empty) and len(hist) > 100:
                    hist = hist.reset_index()
                    if 'Date' in hist.columns:
                        hist['Date'] = pd.to_datetime(hist['Date']).dt.strftime('%Y-%m-%d')
                    elif 'Datetime' in hist.columns:
                        hist['Date'] = pd.to_datetime(hist['Datetime']).dt.strftime('%Y-%m-%d')
                        hist = hist.drop(columns=['Datetime'])
                    
                    # Ensure we have all required columns
                    required = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required:
                        if col not in hist.columns:
                            if col == 'Volume':
                                hist[col] = 1000000
                            else:
                                hist[col] = hist['Close']
                    
                    current_price = float(hist['Close'].iloc[-1])
                    return hist, current_price, None
                    
            except Exception as e:
                if "429" in str(e):
                    time.sleep(5)
                continue
        
        # Fallback: generate synthetic data
        return generate_fallback_data(ticker)
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return generate_fallback_data(ticker)

def generate_fallback_data(ticker, days=2520):
    """Generate fallback synthetic data"""
    base_prices = {'AAPL':182,'MSFT':407,'GOOGL':172,'AMZN':178,'TSLA':175,'SPY':445}
    base_price = base_prices.get(ticker, 100.0)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(days*1.4))
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.dayofweek < 5][-days:]
    
    prices = [base_price]
    for _ in range(1, len(dates)):
        change = random.gauss(0, 0.02)
        new_price = max(prices[-1] * (1 + change), base_price * 0.05)
        prices.append(new_price)
    
    df = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'Open': [p * random.uniform(0.99, 1.01) for p in prices],
        'High': [p * random.uniform(1.01, 1.03) for p in prices],
        'Low': [p * random.uniform(0.97, 0.99) for p in prices],
        'Close': prices,
        'Volume': [random.randint(1000000, 10000000) for _ in range(len(prices))]
    })
    
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
                
            return True
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def load_models(self, symbol):
        """Load trained models and history"""
        try:
            loaded_models = {target: {} for target in self.targets}
            algorithms = ['linear_regression', 'svr', 'random_forest', 'arima']
            
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
                    self.is_fitted = True
                except Exception as e:
                    print(f"Error loading history: {e}")
            
            self.models = loaded_models
            return any(any(models.values()) for models in loaded_models.values())
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def prepare_training_data(self, data):
        """Prepare data for OCHL prediction"""
        try:
            data_with_features = create_advanced_features(data)
            
            # Feature selection
            base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
            technical_features = [
                'Return', 'Volatility', 'High_Low_Range', 'Open_Close_Range',
                'MA_5', 'MA_10', 'MA_20', 'MA_50',
                'MA_Ratio_5', 'MA_Ratio_10', 'MA_Ratio_20', 'MA_Ratio_50',
                'Volume_Ratio', 'Volume_Change', 'RSI', 'MACD',
                'MACD_Signal', 'MACD_Histogram', 'BB_Width', 'BB_Position',
                'Momentum_5', 'Momentum_10', 'Momentum_20'
            ]
            
            # Select available features
            all_features = base_features + technical_features
            self.feature_columns = [f for f in all_features if f in data_with_features.columns]
            
            if len(self.feature_columns) < 10:
                # Use all available features if we don't have enough
                self.feature_columns = [col for col in data_with_features.columns 
                                      if col not in ['Date'] and col not in self.targets]
            
            # Prepare X (features) for each target
            X_data = {}
            y_data = {}
            
            for target in self.targets:
                if target not in data_with_features.columns:
                    # Create target if missing (for next day prediction)
                    data_with_features[target] = data_with_features['Close']
                
                # Prepare sequences (use past 30 days to predict next day)
                X_list = []
                y_list = []
                window_size = 30
                
                for i in range(window_size, len(data_with_features)):
                    # Features from window
                    features = data_with_features[self.feature_columns].iloc[i-window_size:i].values.flatten()
                    
                    # Target is next day's value (shift back since we're predicting forward)
                    if i < len(data_with_features) - 1:
                        target_value = data_with_features[target].iloc[i+1]
                    else:
                        # For last point, use current as approximation
                        target_value = data_with_features[target].iloc[i]
                    
                    X_list.append(features)
                    y_list.append(target_value)
                
                X_data[target] = np.array(X_list)
                y_data[target] = np.array(y_list)
            
            return X_data, y_data, data_with_features
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            return None, None, None
    
    def train_algorithm(self, X, y, algorithm):
        """Train a specific algorithm"""
        try:
            if algorithm == 'linear_regression':
                model = LinearRegression()
                model.fit(X, y)
                return model
                
            elif algorithm == 'svr':
                model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                # Use smaller sample for SVR if data is large
                if len(X) > 1000:
                    idx = np.random.choice(len(X), 1000, replace=False)
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
                # ARIMA needs 1D time series
                if len(y) > 50:
                    try:
                        model = pm.auto_arima(
                            y,
                            start_p=1, start_q=1,
                            max_p=3, max_q=3, m=1,
                            seasonal=False,
                            d=1, D=0,
                            trace=False,
                            error_action='ignore',
                            suppress_warnings=True
                        )
                        return model
                    except:
                        # Fallback to simple ARIMA
                        model = StatsmodelsARIMA(y, order=(1,1,1))
                        model_fit = model.fit()
                        return model_fit
                else:
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
                return False, "Insufficient data for training"
            
            algorithms = ['linear_regression', 'svr', 'random_forest', 'arima']
            self.models = {target: {} for target in self.targets}
            self.scalers = {}
            
            print(f"Training OCHL models for {symbol}...")
            
            for target in self.targets:
                print(f"  Training models for {target}...")
                X = X_data[target]
                y = y_data[target]
                
                if len(X) == 0 or len(y) == 0:
                    continue
                
                # Scale features and target
                X_scaled = self.feature_scaler.fit_transform(X)
                
                # Scale target
                target_scaler = StandardScaler()
                y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
                self.scalers[target] = target_scaler
                
                # Train each algorithm
                for algo in algorithms:
                    model = self.train_algorithm(X_scaled, y_scaled, algo)
                    self.models[target][algo] = model
            
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
            
            return True, "All OCHL models trained successfully"
            
        except Exception as e:
            print(f"Error training all models: {e}")
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
                
                if len(X) == 0 or len(y) == 0:
                    continue
                
                # Split for validation (80/20)
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Scale features
                X_train_scaled = self.feature_scaler.transform(X_train)
                X_test_scaled = self.feature_scaler.transform(X_test)
                
                # Scale target
                target_scaler = self.scalers.get(target, StandardScaler())
                y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).flatten()
                y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
                
                target_performance = {}
                
                for algo, model in self.models[target].items():
                    if model is None:
                        continue
                    
                    try:
                        if algo == 'arima':
                            # ARIMA predictions
                            predictions_scaled = model.predict_in_sample()
                            # Align with test data
                            if len(predictions_scaled) > len(y_test_scaled):
                                predictions_scaled = predictions_scaled[-len(y_test_scaled):]
                            elif len(predictions_scaled) < len(y_test_scaled):
                                predictions_scaled = np.pad(predictions_scaled, 
                                                          (0, len(y_test_scaled) - len(predictions_scaled)), 
                                                          'edge')
                        else:
                            # Other models
                            predictions_scaled = model.predict(X_test_scaled)
                        
                        # Inverse transform
                        predictions = target_scaler.inverse_transform(
                            predictions_scaled.reshape(-1, 1)
                        ).flatten()
                        
                        actuals = target_scaler.inverse_transform(
                            y_test_scaled.reshape(-1, 1)
                        ).flatten()
                        
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
            return performance
            
        except Exception as e:
            print(f"Error calculating historical performance: {e}")
            return {}
    
    def calculate_risk_metrics(self, data):
        """Calculate various risk metrics"""
        try:
            returns = data['Close'].pct_change().dropna()
            
            if len(returns) == 0:
                self.risk_metrics = {}
                return
            
            # Basic risk metrics
            volatility = returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5) * 100
            var_99 = np.percentile(returns, 1) * 100
            
            # Expected Shortfall (CVaR)
            cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
            cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100
            
            # Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Skewness and Kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
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
                'actual_open': float(data['Open'].iloc[-1]),
                'actual_high': float(data['High'].iloc[-1]),
                'actual_low': float(data['Low'].iloc[-1]),
                'actual_close': float(data['Close'].iloc[-1]),
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
            X_data, _, data_with_features = self.prepare_training_data(data)
            
            if X_data is None or any(len(X) == 0 for X in X_data.values()):
                return None, "Insufficient data for prediction"
            
            predictions = {}
            confidence_scores = {}
            algorithm_predictions = {}
            
            for target in self.targets:
                if target not in self.models or target not in X_data:
                    continue
                
                X = X_data[target]
                if len(X) == 0:
                    continue
                
                # Get latest features for prediction
                latest_features = X[-1:].reshape(1, -1)
                latest_scaled = self.feature_scaler.transform(latest_features)
                
                target_predictions = {}
                target_confidences = {}
                
                for algo, model in self.models[target].items():
                    if model is None:
                        continue
                    
                    try:
                        if algo == 'arima':
                            # ARIMA needs time series data
                            target_scaler = self.scalers.get(target, StandardScaler())
                            y_scaled = target_scaler.transform(
                                data_with_features[target].values[-100:].reshape(-1, 1)
                            ).flatten()
                            
                            # Train temporary ARIMA on recent data
                            if len(y_scaled) > 30:
                                temp_arima = pm.auto_arima(
                                    y_scaled,
                                    start_p=1, start_q=1,
                                    max_p=3, max_q=3,
                                    seasonal=False,
                                    trace=False,
                                    error_action='ignore'
                                )
                                pred_scaled = temp_arima.predict(n_periods=1)[0]
                            else:
                                pred_scaled = y_scaled[-1] if len(y_scaled) > 0 else 0
                        else:
                            # Other models
                            pred_scaled = float(model.predict(latest_scaled)[0])
                        
                        # Inverse transform
                        target_scaler = self.scalers.get(target, StandardScaler())
                        pred_actual = target_scaler.inverse_transform(
                            np.array([[pred_scaled]])
                        )[0][0]
                        
                        # Calculate confidence based on historical performance
                        confidence = 70  # default
                        if target in self.historical_performance and algo in self.historical_performance[target]:
                            perf = self.historical_performance[target][algo]
                            confidence = (
                                perf.get('direction_accuracy', 0) * 0.4 +
                                (100 - min(perf.get('mape', 100), 100)) * 0.4 +
                                (perf.get('r2', 0) * 100) * 0.2
                            )
                            confidence = min(max(confidence, 0), 100)
                        
                        target_predictions[algo] = pred_actual
                        target_confidences[algo] = confidence
                        
                    except Exception as e:
                        print(f"Error predicting {target} with {algo}: {e}")
                        continue
                
                if target_predictions:
                    # Calculate weighted ensemble prediction
                    weights = {}
                    total_weight = 0
                    
                    for algo, conf in target_confidences.items():
                        weight = conf / 100  # Convert confidence to weight
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
                    
                    predictions[target] = ensemble_pred
                    confidence_scores[target] = np.mean(list(target_confidences.values()))
                    algorithm_predictions[target] = {
                        'individual': target_predictions,
                        'confidences': target_confidences,
                        'ensemble': ensemble_pred,
                        'ensemble_confidence': np.mean(list(target_confidences.values()))
                    }
            
            # Generate risk alerts
            risk_alerts = self.generate_risk_alerts(data_with_features, predictions)
            
            # Calculate prediction confidence metrics
            confidence_metrics = self.calculate_prediction_confidence(predictions, confidence_scores)
            
            return {
                'predictions': predictions,
                'algorithm_details': algorithm_predictions,
                'confidence_scores': confidence_scores,
                'confidence_metrics': confidence_metrics,
                'risk_alerts': risk_alerts,
                'current_prices': {
                    'open': float(data['Open'].iloc[-1]),
                    'high': float(data['High'].iloc[-1]),
                    'low': float(data['Low'].iloc[-1]),
                    'close': float(data['Close'].iloc[-1])
                }
            }, None
            
        except Exception as e:
            print(f"Error predicting OCHL: {e}")
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
                consistency = 1 - (np.std(predicted_values) / np.mean(predicted_values)) if np.mean(predicted_values) != 0 else 0
                consistency_score = max(0, min(100, consistency * 100))
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
            current_close = data['Close'].iloc[-1]
            predicted_close = predictions.get('Close', current_close)
            
            # Calculate expected change
            expected_change = ((predicted_close - current_close) / current_close) * 100
            
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
            recent_volatility = data['Volatility'].iloc[-10:].mean() if 'Volatility' in data.columns else 0
            if recent_volatility > 0.03:
                alerts.append({
                    'level': '🟡 HIGH',
                    'type': 'High Volatility',
                    'message': f'Recent volatility: {recent_volatility:.2%}',
                    'details': 'Market showing increased volatility'
                })
            
            # 3. RSI alerts
            if 'RSI' in data.columns:
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
            if 'Volume_Ratio' in data.columns:
                volume_ratio = data['Volume_Ratio'].iloc[-1]
                if volume_ratio > 2.0:
                    alerts.append({
                        'level': '🟡 HIGH',
                        'type': 'Unusual Volume',
                        'message': f'Volume {volume_ratio:.1f}x average',
                        'details': 'High volume may indicate significant news or event'
                    })
            
            # 5. Prediction confidence alerts
            if 'Close' in predictions:
                close_pred = predictions['Close']
                confidence = self.calculate_prediction_confidence(predictions, {})
                overall_conf = confidence.get('overall_confidence', 0)
                
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
        
        # Get historical data
        historical_data, current_price, error = get_live_stock_data_enhanced(symbol)
        if error:
            return jsonify({"error": str(error)}), 400
        
        # Load existing models or train new ones
        models_loaded = predictor.load_models(symbol)
        
        if not models_loaded or not predictor.is_fitted:
            # Train all models
            success, train_msg = predictor.train_all_models(historical_data, symbol)
            if not success:
                return provide_fallback_prediction(symbol, historical_data)
        
        # Make prediction
        prediction_result, pred_error = predictor.predict_ochl(historical_data, symbol)
        
        if pred_error:
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
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction endpoint error: {e}")
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
        current_price = historical_data['Close'].iloc[-1] if not historical_data.empty else 100.0
        
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
    
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    port = int(os.environ.get('PORT', 8080))
    server.run(host='0.0.0.0', port=port, debug=True, threaded=True)