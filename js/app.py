# app.py — Flask only, with model persistence (joblib) and prediction.html wiring
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
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
import joblib  # persist models
import pmdarima as pm  # For ARIMA
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA
import warnings
warnings.filterwarnings('ignore')

# ---------------- Config ----------------
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

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
        return pd.Series([50] * len(prices), index=prices.index if hasattr(prices, 'index') else range(len(prices)))

def create_advanced_features(data):
    """Robust feature creation with safe fallbacks"""
    try:
        # Convert multiindex -> single level if needed
        if isinstance(data.columns, pd.MultiIndex):
            flat = pd.DataFrame()
            for col in data.columns:
                flat[col[0]] = data[col]
            data = flat

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        data = data.copy()
        required_columns = ['Open','High','Low','Close','Volume']
        for col in required_columns:
            if col not in data.columns:
                if col == 'Volume':
                    data[col] = 1000000
                else:
                    data[col] = data['Close'] if 'Close' in data.columns else 100.0
        for col in required_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            if data[col].isna().all():
                data[col] = 1000000 if col=='Volume' else 100.0
            data[col] = data[col].fillna(method='ffill').fillna(method='bfill')

        # pad to minimum length if needed so indicators compute properly
        if len(data) < 60:
            while len(data) < 60:
                new_row = {}
                for col in required_columns:
                    if col == 'Volume':
                        new_row[col] = 1000000
                    else:
                        base = data[col].iloc[-1] if len(data)>0 else 100.0
                        new_row[col] = base * random.uniform(0.995,1.005)
                data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

        data['Return'] = data['Close'].pct_change().fillna(0)
        data['Volatility'] = data['Return'].rolling(window=5, min_periods=1).std().fillna(0)
        data['MA_5'] = data['Close'].rolling(window=5, min_periods=1).mean()
        data['MA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        data['MA_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
        data['Close_Ratio_5'] = data['Close'] / data['MA_5'].replace(0,1)
        data['Close_Ratio_20'] = data['Close'] / data['MA_20'].replace(0,1)
        data['Close_Ratio_50'] = data['Close'] / data['MA_50'].replace(0,1)
        data['Volume_MA'] = data['Volume'].rolling(window=5, min_periods=1).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA'].replace(0,1)
        data['Price_Range'] = (data['High'] - data['Low']) / data['Close'].replace(0,1)
        data['HL_Ratio'] = data['High'] / data['Low'].replace(0,1)
        data['OC_Ratio'] = data['Open'] / data['Close'].replace(0,1)
        data['RSI'] = calculate_rsi(data['Close'])
        data['MACD'] = data['Close'].ewm(span=12, min_periods=1).mean() - data['Close'].ewm(span=26, min_periods=1).mean()
        data['MACD_Signal'] = data['MACD'].ewm(span=9, min_periods=1).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        data['Momentum_5'] = (data['Close'] / data['Close'].shift(5).replace(0,1)) - 1
        data['Momentum_10'] = (data['Close'] / data['Close'].shift(10).replace(0,1)) - 1

        data = data.fillna(method='ffill').fillna(method='bfill')
        # final fallback fills
        for col in data.columns:
            if data[col].isna().any():
                if col in ['RSI','MACD','MACD_Signal','MACD_Histogram']:
                    data[col] = data[col].fillna(0)
                elif 'Ratio' in col or 'Momentum' in col:
                    data[col] = data[col].fillna(1)
                elif 'Volatility' in col or 'Return' in col:
                    data[col] = data[col].fillna(0)
                else:
                    data[col] = data[col].fillna(100.0 if col!='Volume' else 1000000)
        return data
    except Exception as e:
        print("Feature creation error:", e)
        return pd.DataFrame({
            'Open':[100.0],'High':[101.0],'Low':[99.0],'Close':[100.0],'Volume':[1000000]
        })

def detect_crises(data, threshold=0.05):
    try:
        data = data.copy()
        data['Crisis'] = (data['Return'].abs()>threshold).astype(int)
        return data
    except:
        data['Crisis'] = 0
        return data

# ---------------- Fallback data generator ----------------
def generate_fallback_data(ticker, base_price=None, days=2520):
    base_prices = {'AAPL':189.5,'MSFT':330.45,'GOOGL':142.3,'AMZN':178.2,'TSLA':248.5,'SPY':445.2}
    if base_price is None:
        base_price = base_prices.get(ticker, 100.0)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(days*1.4))
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.dayofweek < 5]
    dates = dates[-days:]
    prices = [base_price*random.uniform(0.9,1.1)]
    volumes = [random.randint(5_000_000,50_000_000)]
    volatility = 0.02
    for i in range(1,len(dates)):
        change = random.gauss(0,volatility)
        if random.random() < 0.01:
            change += random.gauss(0,0.05)
        new_price = max(prices[-1]*(1+change), base_price*0.05)
        prices.append(new_price)
        volumes.append(int(max(min(volumes[-1]*(1+random.gauss(0,0.2)),200_000_000),1_000_000)))
    opens,highs,lows = [],[],[]
    for i,close in enumerate(prices):
        open_price = close * random.uniform(0.98,1.02)
        daily_range = volatility*close
        high = max(open_price,close) + daily_range*random.uniform(0.2,0.8)
        low = min(open_price,close) - daily_range*random.uniform(0.2,0.8)
        highs.append(high); lows.append(low); opens.append(open_price)
    df = pd.DataFrame({'Date':dates.strftime('%Y-%m-%d'),'Open':opens,'High':highs,'Low':lows,'Close':prices,'Volume':volumes})
    return df, prices[-1], None

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
                if isinstance(hist, pd.DataFrame) and (not hist.empty) and len(hist)>50:
                    hist = hist.reset_index()
                    if 'Date' in hist.columns:
                        hist['Date'] = pd.to_datetime(hist['Date']).dt.strftime('%Y-%m-%d')
                    elif 'Datetime' in hist.columns:
                        hist['Date'] = pd.to_datetime(hist['Datetime']).dt.strftime('%Y-%m-%d')
                        hist = hist.drop(columns=['Datetime'])
                    current_price = float(hist['Close'].iloc[-1])
                    return hist, current_price, None
            except Exception as e:
                # handle throttling
                if "429" in str(e):
                    time.sleep(5)
                continue
        return generate_fallback_data(ticker)
    except Exception as e:
        return generate_fallback_data(ticker)

# ---------------- MULTI-ALGORITHM PREDICTOR CLASS ----------------
class MultiAlgorithmPredictor:
    def __init__(self):
        self.models = {}  # Will store models for each algorithm
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.is_fitted = False
        self.feature_columns = []
        self.historical_performance = {}
        self.prediction_history = {}
        self.algorithm_weights = {}  # Weights based on past performance
        self.target = 'Close'  # Main target for prediction
        
    def model_file(self, symbol, algorithm):
        safe = "".join([c for c in symbol if c.isalnum() or c in "-_"]).upper()
        return os.path.join(MODELS_DIR, f"{safe}_{algorithm}_model.joblib")
    
    def save_models(self, symbol):
        """Save all trained models for a symbol"""
        for algo in self.models.keys():
            try:
                model_data = {
                    'model': self.models[algo],
                    'feature_columns': self.feature_columns,
                    'scaler_features': self.scaler_features,
                    'scaler_target': self.scaler_target
                }
                joblib.dump(model_data, self.model_file(symbol, algo))
            except Exception as e:
                print(f"Error saving {algo} model: {e}")
    
    def load_models(self, symbol):
        """Load all trained models for a symbol"""
        loaded_models = {}
        algorithms = ['linear_regression', 'svr', 'random_forest', 'arima']
        
        for algo in algorithms:
            try:
                path = self.model_file(symbol, algo)
                if os.path.exists(path):
                    model_data = joblib.load(path)
                    loaded_models[algo] = model_data['model']
                    if algo == 'linear_regression':  # Load scalers from first model
                        self.feature_columns = model_data.get('feature_columns', [])
                        self.scaler_features = model_data.get('scaler_features', StandardScaler())
                        self.scaler_target = model_data.get('scaler_target', StandardScaler())
                else:
                    loaded_models[algo] = None
            except Exception as e:
                print(f"Error loading {algo} model: {e}")
                loaded_models[algo] = None
        
        return loaded_models
    
    def prepare_data(self, data, forecast_days=1):
        """Prepare data for training and prediction"""
        try:
            # Create features
            data_with_features = create_advanced_features(data)
            
            # Feature selection
            base_features = ['Open', 'High', 'Low', 'Volume']
            technical_features = [
                'Return', 'Volatility', 'MA_5', 'MA_20', 'MA_50',
                'Close_Ratio_5', 'Close_Ratio_20', 'Close_Ratio_50',
                'Volume_MA', 'Volume_Ratio', 'Price_Range', 'HL_Ratio', 'OC_Ratio',
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 
                'Momentum_5', 'Momentum_10'
            ]
            
            all_features = base_features + technical_features
            available_features = [f for f in all_features if f in data_with_features.columns]
            
            if len(available_features) < 5:
                available_features = [f for f in base_features if f in data_with_features.columns]
            
            self.feature_columns = available_features
            
            # Prepare features and target
            X = data_with_features[available_features].values
            y = data_with_features[self.target].values
            
            # Scale features and target
            X_scaled = self.scaler_features.fit_transform(X) if len(X) > 0 else X
            y_scaled = self.scaler_target.fit_transform(y.reshape(-1, 1)).flatten() if len(y) > 0 else y
            
            return X_scaled, y_scaled, data_with_features
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None, None, None
    
    def train_linear_regression(self, X, y):
        """Train Linear Regression model"""
        try:
            model = LinearRegression()
            model.fit(X, y)
            return model
        except Exception as e:
            print(f"Linear Regression training error: {e}")
            return None
    
    def train_svr(self, X, y):
        """Train Support Vector Regression model"""
        try:
            # Use smaller dataset for SVR due to computational complexity
            if len(X) > 1000:
                X_sample = X[:1000]
                y_sample = y[:1000]
            else:
                X_sample, y_sample = X, y
            
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            model.fit(X_sample, y_sample)
            return model
        except Exception as e:
            print(f"SVR training error: {e}")
            return None
    
    def train_random_forest(self, X, y):
        """Train Random Forest model"""
        try:
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
        except Exception as e:
            print(f"Random Forest training error: {e}")
            return None
    
    def train_arima(self, y):
        """Train ARIMA model using auto_arima for parameter selection"""
        try:
            # Use pmdarima's auto_arima to find best parameters
            model = pm.auto_arima(
                y,
                start_p=1, start_q=1,
                max_p=3, max_q=3, m=1,
                start_P=0, seasonal=False,
                d=1, D=0, trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            return model
        except Exception as e:
            print(f"ARIMA training error: {e}")
            # Fallback to simple ARIMA
            try:
                model = StatsmodelsARIMA(y, order=(1,1,1))
                model_fit = model.fit()
                return model_fit
            except:
                return None
    
    def train_all_models(self, data, symbol):
        """Train all four algorithms"""
        try:
            X, y, data_with_features = self.prepare_data(data)
            
            if X is None or y is None:
                return False, "Insufficient data for training"
            
            print(f"Training models for {symbol} with {len(X)} samples...")
            
            # Train Linear Regression
            lr_model = self.train_linear_regression(X, y)
            
            # Train SVR
            svr_model = self.train_svr(X, y)
            
            # Train Random Forest
            rf_model = self.train_random_forest(X, y)
            
            # Train ARIMA (uses only target values)
            arima_model = self.train_arima(y)
            
            # Store all models
            self.models = {
                'linear_regression': lr_model,
                'svr': svr_model,
                'random_forest': rf_model,
                'arima': arima_model
            }
            
            # Calculate initial performance metrics
            self.calculate_model_performance(X, y, data_with_features)
            
            # Set initial weights based on performance
            self.update_algorithm_weights()
            
            self.is_fitted = True
            
            # Save models
            self.save_models(symbol)
            
            return True, "All models trained successfully"
        except Exception as e:
            print(f"Error training all models: {e}")
            return False, str(e)
    
    def calculate_model_performance(self, X, y, data_with_features):
        """Calculate performance metrics for all trained models"""
        try:
            # Split data for validation (80% train, 20% test)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            performance = {}
            
            for algo, model in self.models.items():
                if model is None:
                    performance[algo] = {
                        'rmse': 9999,
                        'mae': 9999,
                        'r2': -1,
                        'mape': 100,
                        'direction_accuracy': 0
                    }
                    continue
                
                try:
                    if algo == 'arima':
                        # ARIMA specific prediction
                        predictions = model.predict_in_sample()
                        if len(predictions) > len(y_test):
                            predictions = predictions[-len(y_test):]
                        elif len(predictions) < len(y_test):
                            predictions = np.pad(predictions, (0, len(y_test) - len(predictions)), 'edge')
                    else:
                        # For other models
                        if len(X_test) > 0:
                            predictions = model.predict(X_test)
                        else:
                            predictions = model.predict(X_train[-len(y_test):]) if len(X_train) >= len(y_test) else model.predict(X_train)
                    
                    # Ensure predictions and y_test have same length
                    min_len = min(len(predictions), len(y_test))
                    predictions = predictions[:min_len]
                    y_test_eval = y_test[:min_len]
                    
                    if min_len > 0:
                        # Calculate metrics
                        rmse = np.sqrt(mean_squared_error(y_test_eval, predictions))
                        mae = mean_absolute_error(y_test_eval, predictions)
                        r2 = r2_score(y_test_eval, predictions) if len(y_test_eval) > 1 else 0
                        
                        # Calculate MAPE
                        mape = np.mean(np.abs((y_test_eval - predictions) / (y_test_eval + 1e-10))) * 100
                        
                        # Calculate direction accuracy
                        actual_direction = np.diff(y_test_eval) > 0
                        predicted_direction = np.diff(predictions) > 0
                        if len(actual_direction) > 0 and len(predicted_direction) > 0:
                            min_dir_len = min(len(actual_direction), len(predicted_direction))
                            direction_accuracy = accuracy_score(
                                actual_direction[:min_dir_len], 
                                predicted_direction[:min_dir_len]
                            ) * 100
                        else:
                            direction_accuracy = 0
                        
                        performance[algo] = {
                            'rmse': float(rmse),
                            'mae': float(mae),
                            'r2': float(r2),
                            'mape': float(mape),
                            'direction_accuracy': float(direction_accuracy)
                        }
                    else:
                        performance[algo] = {
                            'rmse': 9999,
                            'mae': 9999,
                            'r2': -1,
                            'mape': 100,
                            'direction_accuracy': 0
                        }
                        
                except Exception as e:
                    print(f"Error calculating performance for {algo}: {e}")
                    performance[algo] = {
                        'rmse': 9999,
                        'mae': 9999,
                        'r2': -1,
                        'mape': 100,
                        'direction_accuracy': 0
                    }
            
            self.historical_performance = performance
            return performance
            
        except Exception as e:
            print(f"Error in calculate_model_performance: {e}")
            return {}
    
    def update_algorithm_weights(self):
        """Update weights for ensemble prediction based on performance"""
        try:
            if not self.historical_performance:
                # Default equal weights
                self.algorithm_weights = {
                    'linear_regression': 0.25,
                    'svr': 0.25,
                    'random_forest': 0.25,
                    'arima': 0.25
                }
                return
            
            # Calculate weights based on R² score (higher is better)
            weights = {}
            total_score = 0
            
            for algo, metrics in self.historical_performance.items():
                # Use direction accuracy as weight factor
                score = max(metrics.get('direction_accuracy', 0), 50)  # Minimum 50% weight
                weights[algo] = score
                total_score += score
            
            # Normalize weights
            if total_score > 0:
                for algo in weights:
                    weights[algo] = weights[algo] / total_score
            else:
                # Equal weights if no valid scores
                weights = {algo: 0.25 for algo in self.historical_performance.keys()}
            
            self.algorithm_weights = weights
            
        except Exception as e:
            print(f"Error updating algorithm weights: {e}")
            self.algorithm_weights = {
                'linear_regression': 0.25,
                'svr': 0.25,
                'random_forest': 0.25,
                'arima': 0.25
            }
    
    def predict_with_all_models(self, data, symbol):
        """Get predictions from all four algorithms"""
        try:
            # Prepare data
            X, y, data_with_features = self.prepare_data(data)
            
            if X is None or y is None:
                return None, "Insufficient data for prediction"
            
            # Get latest features for prediction
            latest_features = X[-1:].reshape(1, -1) if len(X) > 0 else None
            
            predictions = {}
            confidence_scores = {}
            
            # Get prediction from each model
            for algo, model in self.models.items():
                if model is None:
                    predictions[algo] = float(y[-1]) if len(y) > 0 else 100.0
                    confidence_scores[algo] = 50
                    continue
                
                try:
                    if algo == 'arima':
                        # ARIMA prediction
                        forecast = model.predict(n_periods=1)
                        pred_scaled = float(forecast[0])
                    else:
                        # Other models
                        if latest_features is not None:
                            pred_scaled = float(model.predict(latest_features)[0])
                        else:
                            pred_scaled = float(y[-1]) if len(y) > 0 else 100.0
                    
                    # Inverse transform to get actual price
                    pred_actual = self.scaler_target.inverse_transform(
                        np.array([[pred_scaled]])
                    )[0][0]
                    
                    predictions[algo] = pred_actual
                    
                    # Calculate confidence score based on historical performance
                    if algo in self.historical_performance:
                        perf = self.historical_performance[algo]
                        confidence = (perf.get('direction_accuracy', 0) * 0.6 + 
                                    (100 - min(perf.get('mape', 100), 100)) * 0.4)
                        confidence_scores[algo] = min(max(confidence, 0), 100)
                    else:
                        confidence_scores[algo] = 70  # Default confidence
                        
                except Exception as e:
                    print(f"Error predicting with {algo}: {e}")
                    predictions[algo] = float(y[-1]) if len(y) > 0 else 100.0
                    confidence_scores[algo] = 50
            
            # Calculate weighted ensemble prediction
            ensemble_prediction = 0
            total_weight = 0
            
            for algo, pred in predictions.items():
                weight = self.algorithm_weights.get(algo, 0.25)
                ensemble_prediction += pred * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_prediction /= total_weight
            
            # Calculate ensemble confidence
            ensemble_confidence = np.mean(list(confidence_scores.values()))
            
            result = {
                'predictions': predictions,
                'confidence_scores': confidence_scores,
                'ensemble_prediction': ensemble_prediction,
                'ensemble_confidence': ensemble_confidence,
                'weights': self.algorithm_weights,
                'performance': self.historical_performance,
                'current_price': float(data['Close'].iloc[-1]) if 'Close' in data.columns else 100.0
            }
            
            return result, None
            
        except Exception as e:
            print(f"Error in predict_with_all_models: {e}")
            return None, str(e)

# Global predictor instance
predictor = MultiAlgorithmPredictor()

# ---------------- Small helpers ----------------
def get_next_trading_day():
    today = datetime.now()
    if today.weekday()==4: return (today+timedelta(days=3)).strftime('%Y-%m-%d')
    if today.weekday()==5: return (today+timedelta(days=2)).strftime('%Y-%m-%d')
    if today.weekday()==6: return (today+timedelta(days=1)).strftime('%Y-%m-%d')
    return (today+timedelta(days=1)).strftime('%Y-%m-%d')

def get_last_market_date():
    today = datetime.now()
    if today.weekday()==0: return (today-timedelta(days=3)).strftime('%Y-%m-%d')
    if today.weekday()==6: return (today-timedelta(days=2)).strftime('%Y-%m-%d')
    if today.weekday()==5: return (today-timedelta(days=1)).strftime('%Y-%m-%d')
    return (today-timedelta(days=1)).strftime('%Y-%m-%d')

def get_market_status():
    today = datetime.now()
    if today.weekday()>=5: return "closed","Market is closed on weekends"
    current_time = datetime.now().time()
    market_open = datetime.strptime('09:30','%H:%M').time()
    market_close = datetime.strptime('16:00','%H:%M').time()
    if current_time < market_open: return "pre_market","Pre-market hours"
    if current_time > market_close: return "after_hours","After-hours trading"
    return "open","Market is open"

def get_risk_level(change_percent, confidence):
    risk_score = (abs(change_percent)/20*0.6) + ((100-confidence)/100*0.4)
    if risk_score>0.7: return "🔴 EXTREME RISK"
    if risk_score>0.5: return "🟡 HIGH RISK"
    if risk_score>0.3: return "🟠 MEDIUM RISK"
    return "🟢 LOW RISK"

def get_trading_recommendation(change_percent, confidence):
    if confidence < 50:
        return "🚨 LOW CONFIDENCE - WAIT"
    if change_percent > 5:
        return "✅ STRONG BUY"
    if change_percent > 2:
        return "📈 BUY"
    if change_percent < -5:
        return "💼 STRONG SELL"
    if change_percent < -2:
        return "📉 SELL"
    return "🔄 HOLD"

def get_algorithm_description(algorithm):
    descriptions = {
        'linear_regression': 'Simple linear relationship model between features and target',
        'svr': 'Support Vector Regression - good for non-linear patterns',
        'random_forest': 'Ensemble of decision trees, robust to overfitting',
        'arima': 'Time series model using autocorrelation'
    }
    return descriptions.get(algorithm, 'Machine learning algorithm')

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

# ---------------- API endpoints ----------------
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
                stock['price'] = round(current,2); stock['change']=round(change,2)
        except Exception:
            continue
    return jsonify(popular_stocks)

@server.route('/api/health')
def health_check():
    return jsonify({"status":"healthy","timestamp":datetime.now().isoformat(),"version":"3.0.0"})

@server.route('/api/predict', methods=['POST'])
@rate_limiter
def predict_stock():
    try:
        data = request.get_json() or {}
        symbol = (data.get('symbol') or 'SPY').upper().strip()
        
        # Get historical data
        historical_data, current_price, error = get_live_stock_data_enhanced(symbol)
        if error:
            return jsonify({"error": str(error)}), 400
        
        # Try to load existing models
        loaded_models = predictor.load_models(symbol)
        predictor.models = loaded_models
        
        # Check if we have at least one model loaded
        has_models = any(model is not None for model in loaded_models.values())
        
        if not has_models:
            # Train all models if none are loaded
            success, train_msg = predictor.train_all_models(historical_data, symbol)
            if not success:
                return provide_fallback_prediction(symbol, current_price, historical_data)
        else:
            # Update performance metrics with loaded models
            X, y, data_with_features = predictor.prepare_data(historical_data)
            if X is not None and y is not None:
                predictor.calculate_model_performance(X, y, data_with_features)
                predictor.update_algorithm_weights()
                predictor.is_fitted = True
        
        # Get predictions from all models
        prediction_result, pred_error = predictor.predict_with_all_models(historical_data, symbol)
        
        if pred_error:
            return provide_fallback_prediction(symbol, current_price, historical_data)
        
        # Prepare response
        current_price_val = float(current_price)
        ensemble_pred = prediction_result['ensemble_prediction']
        change_percent = ((ensemble_pred - current_price_val) / current_price_val) * 100
        
        response = {
            "symbol": symbol,
            "current_price": round(current_price_val, 2),
            "prediction_date": get_next_trading_day(),
            "last_trading_day": get_last_market_date(),
            
            # Individual algorithm predictions
            "algorithm_predictions": {
                algo: {
                    "predicted_price": round(pred, 2),
                    "confidence": round(prediction_result['confidence_scores'][algo], 1),
                    "change_percent": round(((pred - current_price_val) / current_price_val) * 100, 2),
                    "description": get_algorithm_description(algo)
                }
                for algo, pred in prediction_result['predictions'].items()
            },
            
            # Ensemble prediction (main prediction)
            "ensemble_prediction": {
                "predicted_price": round(ensemble_pred, 2),
                "confidence": round(prediction_result['ensemble_confidence'], 1),
                "change_percent": round(change_percent, 2),
                "weights_used": prediction_result['weights']
            },
            
            # Model performance metrics
            "model_performance": {
                algo: {
                    "rmse": round(metrics.get('rmse', 0), 4),
                    "r2": round(metrics.get('r2', 0), 3),
                    "mape": round(metrics.get('mape', 0), 2),
                    "direction_accuracy": round(metrics.get('direction_accuracy', 0), 1)
                }
                for algo, metrics in prediction_result['performance'].items()
            },
            
            # Trading recommendations
            "change_percent": round(change_percent, 2),
            "risk_level": get_risk_level(change_percent, prediction_result['ensemble_confidence']),
            "recommendation": get_trading_recommendation(change_percent, prediction_result['ensemble_confidence']),
            
            # Additional info
            "market_status": get_market_status()[1],
            "best_algorithm": max(
                prediction_result['performance'].items(),
                key=lambda x: x[1].get('direction_accuracy', 0)
            )[0] if prediction_result['performance'] else "ensemble",
            
            "data_analysis": {
                "total_data_points": len(historical_data),
                "features_used": len(predictor.feature_columns),
                "algorithms_used": len([m for m in predictor.models.values() if m is not None])
            },
            
            "insight": f"AI ensemble predicts {change_percent:+.2f}% movement for {symbol}. Best performing algorithm: {max(prediction_result['performance'].items(), key=lambda x: x[1].get('direction_accuracy', 0))[0] if prediction_result['performance'] else 'ensemble'}."
        }
        
        return jsonify(response)
        
    except Exception as e:
        print("Predict endpoint error:", e)
        return provide_fallback_prediction((request.get_json() or {}).get('symbol', 'SPY').upper(), 100.0, None)

@server.route('/api/train', methods=['POST'])
@rate_limiter
def train_models():
    """API endpoint to explicitly train models for a symbol"""
    try:
        data = request.get_json() or {}
        symbol = (data.get('symbol') or 'SPY').upper().strip()
        
        # Get historical data
        historical_data, current_price, error = get_live_stock_data_enhanced(symbol)
        if error:
            return jsonify({"error": str(error)}), 400
        
        # Train all models
        success, train_msg = predictor.train_all_models(historical_data, symbol)
        
        if success:
            return jsonify({
                "status": "success",
                "message": train_msg,
                "symbol": symbol,
                "models_trained": list(predictor.models.keys()),
                "performance": predictor.historical_performance
            })
        else:
            return jsonify({
                "status": "error",
                "message": train_msg,
                "symbol": symbol
            }), 400
            
    except Exception as e:
        print("Train endpoint error:", e)
        return jsonify({
            "status": "error",
            "message": str(e),
            "symbol": symbol if 'symbol' in locals() else 'UNKNOWN'
        }), 500

@server.route('/api/compare_models', methods=['POST'])
@rate_limiter
def compare_models():
    """API endpoint to compare performance of all models"""
    try:
        data = request.get_json() or {}
        symbol = (data.get('symbol') or 'SPY').upper().strip()
        
        # Get historical data
        historical_data, current_price, error = get_live_stock_data_enhanced(symbol)
        if error:
            return jsonify({"error": str(error)}), 400
        
        # Prepare data for performance comparison
        X, y, data_with_features = predictor.prepare_data(historical_data)
        
        if X is None or y is None:
            return jsonify({"error": "Insufficient data for comparison"}), 400
        
        # Train temporary models for comparison
        comparison_results = {}
        algorithms = ['linear_regression', 'svr', 'random_forest', 'arima']
        
        for algo in algorithms:
            try:
                if algo == 'linear_regression':
                    model = predictor.train_linear_regression(X, y)
                elif algo == 'svr':
                    model = predictor.train_svr(X, y)
                elif algo == 'random_forest':
                    model = predictor.train_random_forest(X, y)
                elif algo == 'arima':
                    model = predictor.train_arima(y)
                
                # Calculate performance
                split_idx = int(len(X) * 0.8)
                X_test = X[split_idx:]
                y_test = y[split_idx:]
                
                if algo == 'arima':
                    predictions = model.predict_in_sample()[-len(y_test):]
                else:
                    predictions = model.predict(X_test)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-10))) * 100
                
                comparison_results[algo] = {
                    "rmse": round(float(rmse), 4),
                    "mae": round(float(mae), 4),
                    "r2": round(float(r2), 3),
                    "mape": round(float(mape), 2),
                    "description": get_algorithm_description(algo),
                    "status": "success"
                }
                
            except Exception as e:
                comparison_results[algo] = {
                    "status": "error",
                    "message": str(e),
                    "description": get_algorithm_description(algo)
                }
        
        return jsonify({
            "symbol": symbol,
            "comparison": comparison_results,
            "best_model": min(
                [(algo, metrics.get('rmse', 9999)) for algo, metrics in comparison_results.items() 
                 if metrics.get('status') == 'success'],
                key=lambda x: x[1]
            )[0] if any(m.get('status') == 'success' for m in comparison_results.values()) else "none"
        })
        
    except Exception as e:
        print("Compare models error:", e)
        return jsonify({"error": str(e)}), 500

def provide_fallback_prediction(symbol, current_price, historical_data):
    """Provide fallback prediction when models fail"""
    try:
        if current_price is None:
            base_prices = {'AAPL':182,'MSFT':407,'GOOGL':172,'AMZN':178,'TSLA':175,'SPY':445}
            current_price = base_prices.get(symbol, 100.0)
        
        change_percent = (random.random()-0.5)*8
        predicted_price = current_price * (1 + change_percent/100)
        
        # Simulate different algorithm predictions
        algorithms = {
            'linear_regression': predicted_price * random.uniform(0.98, 1.02),
            'svr': predicted_price * random.uniform(0.97, 1.03),
            'random_forest': predicted_price * random.uniform(0.99, 1.01),
            'arima': predicted_price * random.uniform(0.96, 1.04)
        }
        
        algorithm_predictions = {}
        for algo, pred in algorithms.items():
            algorithm_predictions[algo] = {
                "predicted_price": round(pred, 2),
                "confidence": random.randint(60, 80),
                "change_percent": round(((pred - current_price) / current_price) * 100, 2),
                "description": get_algorithm_description(algo)
            }
        
        return jsonify({
            "symbol": symbol,
            "current_price": round(float(current_price), 2),
            "prediction_date": get_next_trading_day(),
            
            "algorithm_predictions": algorithm_predictions,
            
            "ensemble_prediction": {
                "predicted_price": round(predicted_price, 2),
                "confidence": 65,
                "change_percent": round(change_percent, 2),
                "weights_used": {"linear_regression":0.25,"svr":0.25,"random_forest":0.25,"arima":0.25}
            },
            
            "change_percent": round(change_percent, 2),
            "risk_level": get_risk_level(change_percent, 65),
            "recommendation": get_trading_recommendation(change_percent, 65),
            "market_status": get_market_status()[1],
            "fallback": True,
            "message": "Using fallback prediction engine"
        })
    except Exception as e:
        print("Fallback error:", e)
        return jsonify({"error":"Prediction temporarily unavailable","fallback":True,"symbol":symbol}), 500

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
    print("Starting Flask app with multi-algorithm prediction system...")
    print("Algorithms available: Linear Regression, SVR, Random Forest, ARIMA")
    
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    port = int(os.environ.get('PORT', 8080))
    server.run(host='0.0.0.0', port=port, debug=True, threaded=True)