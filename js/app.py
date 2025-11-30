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

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
import joblib  # persist models

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
        return rsi
    except:
        return pd.Series([50] * len(prices), index=prices.index)

def create_advanced_features(data):
    """(same robust feature creation as before)"""
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

        if len(data) < 30:
            while len(data) < 30:
                new_row = {}
                for col in required_columns:
                    if col == 'Volume':
                        new_row[col] = 1000000
                    else:
                        base = data[col].iloc[-1] if len(data)>0 else 100.0
                        new_row[col] = base * random.uniform(0.99,1.01)
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
    cycle_period = max(1,len(dates)//4)
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
        open_price = close * random.uniform(0.98,1.02) if i>0 else close*random.uniform(0.98,1.02)
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
            {"func": lambda: yf.download(ticker, period="2y", interval="1d", progress=False, timeout=30), "name":"2y"},
            {"func": lambda: yf.Ticker(ticker).history(period="2y", interval="1d"), "name":"ticker.history"},
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
                    current_price = hist['Close'].iloc[-1]
                    return hist, current_price, None
            except Exception as e:
                if "429" in str(e): time.sleep(10)
                continue
        return generate_fallback_data(ticker)
    except Exception as e:
        return generate_fallback_data(ticker)

# ---------------- Predictor with persistence ----------------
class AdvancedMultiTargetPredictor:
    def __init__(self):
        self.models = {}
        self.crisis_model = None
        self.scaler_features = StandardScaler()
        self.scaler_targets = {}
        self.is_fitted = False
        self.feature_columns = []
        self.crisis_features = []
        self.model_health_metrics = {}
        self.targets = ['Open','High','Low','Close']
        self.historical_performance = {}
        self.prediction_confidence = {}
        self.market_regime = "NORMAL"

    def model_file(self, symbol):
        safe = "".join([c for c in symbol if c.isalnum() or c in "-_"]).upper()
        return os.path.join(MODELS_DIR, f"{safe}_predictor.joblib")

    def save_to_disk(self, symbol):
        try:
            payload = {
                'models': self.models,
                'crisis_model': self.crisis_model,
                'scaler_features': self.scaler_features,
                'scaler_targets': self.scaler_targets,
                'is_fitted': self.is_fitted,
                'feature_columns': self.feature_columns,
                'crisis_features': self.crisis_features,
                'model_health_metrics': self.model_health_metrics,
                'targets': self.targets,
                'historical_performance': self.historical_performance,
                'prediction_confidence': self.prediction_confidence,
                'market_regime': self.market_regime
            }
            joblib.dump(payload, self.model_file(symbol))
            print(f"✅ Saved model to {self.model_file(symbol)}")
            return True
        except Exception as e:
            print("Save failure:", e)
            return False

    def load_from_disk(self, symbol):
        try:
            path = self.model_file(symbol)
            if not os.path.exists(path):
                return False
            payload = joblib.load(path)
            # restore
            self.models = payload.get('models', {})
            self.crisis_model = payload.get('crisis_model', None)
            self.scaler_features = payload.get('scaler_features', StandardScaler())
            self.scaler_targets = payload.get('scaler_targets', {})
            self.is_fitted = payload.get('is_fitted', False)
            self.feature_columns = payload.get('feature_columns', [])
            self.crisis_features = payload.get('crisis_features', [])
            self.model_health_metrics = payload.get('model_health_metrics', {})
            self.targets = payload.get('targets', ['Open','High','Low','Close'])
            self.historical_performance = payload.get('historical_performance', {})
            self.prediction_confidence = payload.get('prediction_confidence', {})
            self.market_regime = payload.get('market_regime', "NORMAL")
            print(f"✅ Loaded model from {path}")
            return True
        except Exception as e:
            print("Load failure:", e)
            return False

    def prepare_multi_target_data(self, data, target_days=1):
        try:
            data_with_features = create_advanced_features(data)
            base_features = ['Open','High','Low','Volume']
            technical_features = ['Return','Volatility','MA_5','MA_20','MA_50',
                                  'Close_Ratio_5','Close_Ratio_20','Close_Ratio_50',
                                  'Volume_MA','Volume_Ratio','Price_Range','HL_Ratio','OC_Ratio',
                                  'RSI','MACD','MACD_Signal','MACD_Histogram','Momentum_5','Momentum_10']
            available_columns = data_with_features.columns.tolist()
            features = [f for f in (base_features+technical_features) if f in available_columns]
            if len(features) < 4:
                features = [f for f in base_features if f in available_columns]
                if len(features) == 0:
                    return None, None, None, None, None
            targets = self.targets
            data_scaled = data_with_features.copy()
            try:
                data_scaled[features] = self.scaler_features.fit_transform(data_with_features[features])
            except Exception:
                data_scaled[features] = data_with_features[features]
            self.scaler_targets = {}
            for target in targets:
                try:
                    self.scaler_targets[target] = StandardScaler()
                    if target in data_with_features.columns:
                        td = data_with_features[[target]].values.reshape(-1,1)
                        data_scaled[target] = self.scaler_targets[target].fit_transform(td).flatten()
                    else:
                        data_scaled[target] = data_scaled['Close']
                except Exception:
                    data_scaled[target] = data_scaled['Close']
            X,y_arrays = self.create_sequences_multi_target(data_scaled, features, targets, window_size=30)
            if X is None or len(X)==0:
                return None, None, None, None, None
            return X, y_arrays, features, targets, data_scaled
        except Exception as e:
            print("Prep error:", e)
            return None, None, None, None, None

    def create_sequences_multi_target(self, data, features, targets, window_size=30):
        try:
            if len(data) <= window_size:
                return None, None
            X, y_dict = [], {t:[] for t in targets}
            for i in range(window_size, len(data)):
                try:
                    seq = data[features].iloc[i-window_size:i].values.flatten()
                    X.append(seq)
                    for t in targets:
                        if t in data.columns:
                            y_dict[t].append(data[t].iloc[i])
                        else:
                            y_dict[t].append(data['Close'].iloc[i])
                except Exception:
                    continue
            if len(X)==0:
                return None, None
            X = np.array(X)
            y_arrays = {t: np.array(y_dict[t]) for t in targets}
            return X, y_arrays
        except Exception as e:
            print("Seq error:", e)
            return None, None

    def calculate_historical_performance(self, X_train, y_train, X_test, y_test):
        """Calculate comprehensive historical performance metrics"""
        performance = {}
        for target in self.targets:
            if target in self.models and target in y_train and target in y_test:
                try:
                    model = self.models[target]
                    # Training performance
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    train_rmse = np.sqrt(mean_squared_error(y_train[target], y_train_pred))
                    test_rmse = np.sqrt(mean_squared_error(y_test[target], y_test_pred))
                    train_mae = mean_absolute_error(y_train[target], y_train_pred)
                    test_mae = mean_absolute_error(y_test[target], y_test_pred)
                    train_r2 = r2_score(y_train[target], y_train_pred)
                    test_r2 = r2_score(y_test[target], y_test_pred)
                    
                    # Direction accuracy
                    train_dir_acc = np.mean((np.diff(y_train[target]) * np.diff(y_train_pred)) > 0)
                    test_dir_acc = np.mean((np.diff(y_test[target]) * np.diff(y_test_pred)) > 0)
                    
                    performance[target] = {
                        'train_rmse': float(train_rmse),
                        'test_rmse': float(test_rmse),
                        'train_mae': float(train_mae),
                        'test_mae': float(test_mae),
                        'train_r2': float(train_r2),
                        'test_r2': float(test_r2),
                        'train_direction_accuracy': float(train_dir_acc),
                        'test_direction_accuracy': float(test_dir_acc),
                        'overfitting_ratio': float(test_rmse / train_rmse) if train_rmse > 0 else 1.0
                    }
                except Exception as e:
                    print(f"Performance calculation error for {target}: {e}")
                    performance[target] = {
                        'train_rmse': 0.0, 'test_rmse': 0.0, 'train_mae': 0.0, 'test_mae': 0.0,
                        'train_r2': 0.0, 'test_r2': 0.0, 'train_direction_accuracy': 0.0,
                        'test_direction_accuracy': 0.0, 'overfitting_ratio': 1.0
                    }
        return performance

    def calculate_prediction_confidence(self, X_pred):
        """Calculate confidence scores for predictions"""
        confidence = {}
        for target in self.targets:
            if target in self.models:
                try:
                    model = self.models[target]
                    if hasattr(model, 'estimators_'):  # Random Forest
                        predictions = []
                        for estimator in model.estimators_:
                            pred = estimator.predict(X_pred)
                            predictions.append(pred[0])
                        
                        mean_pred = np.mean(predictions)
                        std_pred = np.std(predictions)
                        confidence_interval = 1.96 * std_pred
                        
                        # Confidence score based on coefficient of variation
                        if mean_pred != 0:
                            cv = std_pred / abs(mean_pred)
                            confidence_score = max(0, 1 - cv) * 100
                        else:
                            confidence_score = 50.0
                            
                        confidence[target] = {
                            'confidence_score': float(confidence_score),
                            'mean_prediction': float(mean_pred),
                            'std_deviation': float(std_pred),
                            'confidence_interval': float(confidence_interval),
                            'predictions_count': len(predictions)
                        }
                    else:
                        # For non-ensemble models, use a default confidence
                        pred = model.predict(X_pred)[0]
                        confidence[target] = {
                            'confidence_score': 75.0,
                            'mean_prediction': float(pred),
                            'std_deviation': 0.0,
                            'confidence_interval': float(pred * 0.05),
                            'predictions_count': 1
                        }
                except Exception as e:
                    print(f"Confidence calculation error for {target}: {e}")
                    confidence[target] = {
                        'confidence_score': 50.0,
                        'mean_prediction': 0.0,
                        'std_deviation': 0.0,
                        'confidence_interval': 0.0,
                        'predictions_count': 0
                    }
        return confidence

    def detect_market_regime(self, data):
        """Detect current market regime based on technical indicators"""
        try:
            if len(data) < 20:
                return "NORMAL"
            
            recent_data = data.tail(20)
            
            # Calculate regime indicators
            volatility = recent_data['Volatility'].mean()
            rsi = recent_data['RSI'].iloc[-1]
            momentum = recent_data['Momentum_5'].iloc[-1]
            volume_ratio = recent_data['Volume_Ratio'].iloc[-1]
            
            # Crisis probability from crisis model
            crisis_prob = 0.1
            if self.crisis_model and len(self.crisis_features) > 0:
                available_features = [f for f in self.crisis_features if f in data.columns]
                if available_features:
                    latest_features = data[available_features].iloc[-1:].values
                    try:
                        crisis_probs = self.crisis_model.predict_proba(latest_features)
                        if crisis_probs.shape[1] > 1:
                            crisis_prob = crisis_probs[0, 1]
                    except:
                        pass
            
            # Determine regime
            if crisis_prob > 0.7 or volatility > 0.03:
                return "HIGH_VOLATILITY"
            elif crisis_prob > 0.4 or volatility > 0.02:
                return "ELEVATED_RISK"
            elif rsi > 70 or rsi < 30:
                return "EXTREME_SENTIMENT"
            elif abs(momentum) > 0.05:
                return "STRONG_TREND"
            elif volume_ratio > 1.5:
                return "HIGH_VOLUME"
            else:
                return "NORMAL"
                
        except Exception as e:
            print(f"Market regime detection error: {e}")
            return "NORMAL"

    def train_multi_target_models(self, data, target_days=1):
        try:
            X, y_arrays, features, targets, data_scaled = self.prepare_multi_target_data(data, target_days)
            if X is None:
                return None, "Insufficient data/features"
            self.feature_columns = features
            self.targets = targets
            
            # Split data
            min_test = 2
            if len(X) <= min_test + 5:
                X_train, X_test = X, X[-1:]
                y_train = {t: y_arrays[t][:-1] for t in targets}
                y_test = {t: y_arrays[t][-1:] for t in targets}
            else:
                split = len(X) - min_test
                X_train, X_test = X[:split], X[split:]
                y_train = {t: y_arrays[t][:split] for t in targets}
                y_test = {t: y_arrays[t][split:] for t in targets}
            
            models = {}
            rmse_scores = {}
            for t in targets:
                try:
                    m = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
                    if len(X_train)>0 and len(y_train[t])>0:
                        m.fit(X_train, y_train[t])
                        models[t] = m
                        if len(X_test)>0 and len(y_test[t])>0:
                            pred = m.predict(X_test)
                            rmse_scores[t] = float(np.sqrt(mean_squared_error(y_test[t], pred)))
                    else:
                        fallback = DummyRegressor(strategy="mean")
                        if len(X_train)>0:
                            fallback.fit(X_train, np.ones(len(X_train))*data['Close'].mean())
                        models[t] = fallback
                except Exception as e:
                    print("Train target error:", t, e)
                    fallback = DummyRegressor(strategy="mean")
                    if len(X_train)>0:
                        fallback.fit(X_train, np.ones(len(X_train))*data['Close'].mean())
                    models[t] = fallback
            
            self.models = models
            self.rmse_scores = rmse_scores
            
            # Calculate historical performance
            self.historical_performance = self.calculate_historical_performance(X_train, y_train, X_test, y_test)
            
            # Calculate initial prediction confidence
            X_pred = X[-1:] if len(X) > 0 else X_train[-1:] if len(X_train) > 0 else None
            if X_pred is not None:
                self.prediction_confidence = self.calculate_prediction_confidence(X_pred)
            
            # Detect market regime
            self.market_regime = self.detect_market_regime(data)
            
            # crisis model
            try:
                crisis_data = detect_crises(data)
                crisis_feats = [f for f in features+['Return','Volatility','Momentum_5'] if f in crisis_data.columns and f in data_scaled.columns]
                crisis_feats = list(dict.fromkeys(crisis_feats))
                self.crisis_features = crisis_feats
                if len(crisis_feats)>0:
                    cd = crisis_data[crisis_feats + ['Crisis']].dropna()
                    if len(cd)>10:
                        Xc = cd[crisis_feats].values
                        yc = cd['Crisis'].values
                        splitc = int(len(Xc)*0.8)
                        if splitc>5:
                            Xc_train = Xc[:splitc]; yc_train = yc[:splitc]
                            crm = RandomForestClassifier(n_estimators=50, random_state=42)
                            crm.fit(Xc_train, yc_train)
                            self.crisis_model = crm
            except Exception as e:
                print("Crisis train failed:", e)
                self.crisis_model = None
            
            self.model_health_metrics = {
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_points': len(X),
                'targets_trained': len([t for t in targets if t in self.models]),
                'crisis_detection_trained': self.crisis_model is not None,
                'feature_count': len(features),
                'crisis_feature_count': len(self.crisis_features),
                'data_range': f"{data['Date'].iloc[0] if 'Date' in data.columns else 'N/A'} to {data['Date'].iloc[-1] if 'Date' in data.columns else 'N/A'}",
                'total_days': len(data),
                'market_regime': self.market_regime,
                'average_confidence': np.mean([c['confidence_score'] for c in self.prediction_confidence.values()]) if self.prediction_confidence else 0.0
            }
            self.is_fitted = True
            return rmse_scores, None
        except Exception as e:
            print("Train error:", e)
            return None, str(e)

    def predict_next_day_prices(self, data):
        if not self.is_fitted:
            return None, None, None, None, "Model not fitted"
        try:
            X, y_arrays, features, targets, data_scaled = self.prepare_multi_target_data(data)
            if X is None or len(X)==0:
                return None, None, None, None, "Insufficient data for prediction"
            X_pred = X[-1:]
            
            # Update prediction confidence
            self.prediction_confidence = self.calculate_prediction_confidence(X_pred)
            
            # Update market regime
            self.market_regime = self.detect_market_regime(data)
            
            predictions_scaled = {}
            for t in targets:
                if t in self.models:
                    try:
                        predictions_scaled[t] = self.models[t].predict(X_pred)
                    except Exception:
                        predictions_scaled[t] = np.array([data[t].iloc[-1] if t in data.columns else data['Close'].iloc[-1]])
                else:
                    predictions_scaled[t] = np.array([data['Close'].iloc[-1]])
            
            # inverse transform where possible
            predictions_actual = {}
            for t in targets:
                if t in predictions_scaled and t in self.scaler_targets:
                    try:
                        predictions_actual[t] = self.scaler_targets[t].inverse_transform(predictions_scaled[t].reshape(-1,1)).flatten()
                    except Exception:
                        predictions_actual[t] = predictions_scaled[t]
                else:
                    predictions_actual[t] = predictions_scaled[t]
            
            confidence_data = self.calculate_confidence_bands_multi(X_pred)
            crisis_probs = self.predict_crisis_probability(data_scaled)
            current_close = data['Close'].iloc[-1] if 'Close' in data.columns else 100.0
            scenarios = self.generate_multi_scenarios(predictions_actual, current_close)
            
            return predictions_actual, confidence_data, scenarios, crisis_probs, None
        except Exception as e:
            print("Predict error:", e)
            return None, None, None, None, str(e)

    def calculate_confidence_bands_multi(self, X, confidence=0.95):
        """Enhanced confidence bands with prediction intervals"""
        confidence_data = {}
        for target in self.targets:
            if target in self.models and target in self.prediction_confidence:
                conf = self.prediction_confidence[target]
                confidence_data[target] = {
                    'mean': conf['mean_prediction'],
                    'lower': conf['mean_prediction'] - conf['confidence_interval'],
                    'upper': conf['mean_prediction'] + conf['confidence_interval'],
                    'std': conf['std_deviation'],
                    'confidence_score': conf['confidence_score'],
                    'prediction_interval': f"±{conf['confidence_interval']:.4f}",
                    'quality': "HIGH" if conf['confidence_score'] > 80 else "MEDIUM" if conf['confidence_score'] > 60 else "LOW"
                }
        return confidence_data

    def predict_crisis_probability(self, data_scaled):
        try:
            if self.crisis_model is None or len(self.crisis_features)==0:
                return [0.1]
            feats = [f for f in self.crisis_features if f in data_scaled.columns]
            if not feats:
                return [0.1]
            latest = data_scaled[feats].iloc[-1:].values
            try:
                probs = self.crisis_model.predict_proba(latest)
                if probs.shape[1]>1:
                    return probs[:,1].tolist()
                else:
                    return [probs[0,0]]
            except Exception:
                return [0.1]
        except Exception:
            return [0.1]

    def get_risk_alerts(self, predictions, current_price, crisis_probs):
        alerts = []
        try:
            changes = []
            for t in ['Open','High','Low','Close']:
                if t in predictions and len(predictions[t])>0:
                    pv = predictions[t]
                    if hasattr(pv,'item'):
                        pv = pv.item() if pv.size==1 else pv[0]
                    elif hasattr(pv,'iloc'):
                        pv = pv.iloc[0]
                    current_val = current_price
                    try:
                        changes.append(((float(pv)-float(current_val))/float(current_val))*100)
                    except Exception:
                        continue
            if changes:
                avg = np.mean(changes)
                if abs(avg)>15:
                    alerts.append({'level':'🔴 CRITICAL','type':'Extreme Movement','message':f'Expected {avg:+.1f}% move'})
                elif abs(avg)>8:
                    alerts.append({'level':'🟡 HIGH','type':'Large Movement','message':f'Expected {avg:+.1f}% move'})
            
            # Enhanced crisis alerts based on market regime
            if crisis_probs:
                avgc = float(np.mean(crisis_probs))
                if self.market_regime == "HIGH_VOLATILITY":
                    alerts.append({'level':'🔴 CRITICAL','type':'High Volatility Regime','message':'Market in high volatility regime'})
                elif self.market_regime == "ELEVATED_RISK":
                    alerts.append({'level':'🟡 HIGH','type':'Elevated Risk Regime','message':'Market in elevated risk regime'})
                
                if avgc>0.7:
                    alerts.append({'level':'🔴 CRITICAL','type':'High Crisis Prob','message':f'Crisis probability: {avgc:.1%}'})
                elif avgc>0.4:
                    alerts.append({'level':'🟡 HIGH','type':'Elevated Crisis','message':f'Crisis probability: {avgc:.1%}'})
            
            # Add confidence-based alerts
            avg_confidence = np.mean([c['confidence_score'] for c in self.prediction_confidence.values()]) if self.prediction_confidence else 0
            if avg_confidence < 50:
                alerts.append({'level':'🟡 HIGH','type':'Low Prediction Confidence','message':f'Model confidence: {avg_confidence:.1f}%'})
                
        except Exception as e:
            alerts.append({'level':'🟡 HIGH','type':'System','message':'Risk assessment unavailable'})
        return alerts

    def generate_multi_scenarios(self, predictions, current_price):
        try:
            base_change = 0
            if 'Close' in predictions and len(predictions['Close'])>0:
                pv = predictions['Close']
                if hasattr(pv,'item'):
                    pv = pv.item() if pv.size==1 else pv[0]
                current_val = current_price
                try:
                    base_change = ((float(pv)-float(current_val))/float(current_val))*100
                except Exception:
                    base_change = 0
            
            # Adjust scenarios based on market regime
            regime_multipliers = {
                "HIGH_VOLATILITY": 1.5,
                "ELEVATED_RISK": 1.3,
                "EXTREME_SENTIMENT": 1.2,
                "STRONG_TREND": 1.1,
                "HIGH_VOLUME": 1.1,
                "NORMAL": 1.0
            }
            
            multiplier = regime_multipliers.get(self.market_regime, 1.0)
            
            return {
                'base':{'probability':50,'price_change':base_change,'description':self.get_scenario_description(base_change)},
                'bullish':{'probability':25,'price_change':base_change*1.3*multiplier,'description':self.get_scenario_description(base_change*1.3*multiplier)},
                'bearish':{'probability':15,'price_change':base_change*0.7*multiplier,'description':self.get_scenario_description(base_change*0.7*multiplier)},
                'sideways':{'probability':10,'price_change':base_change*0.3,'description':self.get_scenario_description(base_change*0.3)}
            }
        except Exception:
            return {'base':{'probability':100,'price_change':0,'description':'Unavailable'}}

    def get_scenario_description(self, change):
        try:
            if change>10: return "STRONG BULLISH"
            if change>5: return "BULLISH"
            if change>2: return "SLIGHTLY BULLISH"
            if change<-10: return "STRONG BEARISH"
            if change<-5: return "BEARISH"
            if change<-2: return "SLIGHTLY BEARISH"
            return "STABLE SIDEWAYS"
        except:
            return "NORMAL"

# global predictor instance — we will load per-symbol into it when needed
predictor = AdvancedMultiTargetPredictor()

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

def get_risk_level(change_percent, crisis_prob):
    risk_score = (abs(change_percent)/20*0.6) + (crisis_prob*0.4)
    if risk_score>0.7: return "🔴 EXTREME RISK"
    if risk_score>0.5: return "🟡 HIGH RISK"
    if risk_score>0.3: return "🟠 MEDIUM RISK"
    return "🟢 LOW RISK"

def get_trading_recommendation(change_percent, risk_level, crisis_prob):
    if risk_level=="🔴 EXTREME RISK":
        return "🚨 EXTREME CAUTION"
    if risk_level=="🟡 HIGH RISK":
        return "📈 CAUTIOUS"
    if risk_level=="🟠 MEDIUM RISK":
        return "🔄 HOLD / CONSIDER"
    if change_percent>2: return "✅ STRONG BUY"
    if change_percent < -1: return "💼 CAUTIOUS SELL"
    return "🔄 HOLD"

# ---------------- NAVIGATION MAP ----------------
NAVIGATION_MAP = {
    'index':'/','jeet':'/jeet','portfolio':'/portfolio','mystock':'/mystock','deposit':'/deposit',
    'insight':'/insight','prediction':'/prediction','news':'/news','videos':'/videos','superstars':'/Superstars',
    'alerts':'/alerts','help':'/help','profile':'/profile'
}

# ---------------- Dynamic routes for NAVIGATION_MAP ----------------
def _find_template_for_page(page_key):
    """
    Try possible template filenames for given page_key and return the first that exists.
    Order: <page>.html, <Page>.html, <page_key_lower>.html, <page_key_title>.html
    If none exist, returns 'jeet.html' as safe fallback.
    """
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
    # minimal: return popular list + try small yfinance fetch
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
    return jsonify({"status":"healthy","timestamp":datetime.now().isoformat(),"version":"2.1.0"})

@server.route('/api/predict', methods=['POST'])
@rate_limiter
def predict_stock():
    try:
        data = request.get_json() or {}
        symbol = (data.get('symbol') or 'SPY').upper().strip()
        # try to load existing saved model first
        loaded = predictor.load_from_disk(symbol)
        historical_data, current_price, error = get_live_stock_data_enhanced(symbol)
        if error:
            return jsonify({"error":str(error)}), 400

        # If model loaded and fitted, use it. Otherwise train and save.
        if not (loaded and predictor.is_fitted):
            rmse, train_err = predictor.train_multi_target_models(historical_data)
            if train_err:
                # fallback response
                return provide_fallback_prediction(symbol, current_price, historical_data)
            # save
            predictor.save_to_disk(symbol)

        # Predict
        predictions, confidence_data, scenarios, crisis_probs, pred_err = predictor.predict_next_day_prices(historical_data)
        if pred_err:
            return provide_fallback_prediction(symbol, current_price, historical_data)

        predicted_close = float(predictions['Close'][0]) if 'Close' in predictions else float(current_price)
        change_percent = ((predicted_close - float(current_price)) / float(current_price)) * 100 if float(current_price) != 0 else 0.0
        avg_crisis_prob = float(np.mean(crisis_probs)) if crisis_probs is not None else 0.1
        risk_level = get_risk_level(change_percent, avg_crisis_prob)
        recommendation = get_trading_recommendation(change_percent, risk_level, avg_crisis_prob)
        risk_alerts = predictor.get_risk_alerts(predictions, current_price, crisis_probs)

        response = {
            "symbol": symbol,
            "current_price": round(float(current_price),2),
            "prediction_date": get_next_trading_day(),
            "last_trading_day": get_last_market_date(),
            "predicted_prices": {
                "open": round(float(predictions['Open'][0]),2) if 'Open' in predictions else round(float(current_price),2),
                "high": round(float(predictions['High'][0]),2) if 'High' in predictions else round(float(current_price)*1.02,2),
                "low": round(float(predictions['Low'][0]),2) if 'Low' in predictions else round(float(current_price)*0.98,2),
                "close": round(predicted_close,2)
            },
            "change_percent": round(change_percent,2),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "crisis_probability": round(avg_crisis_prob,3),
            "confidence_data": confidence_data,
            "scenarios": scenarios,
            "risk_alerts": risk_alerts,
            "model_health": predictor.model_health_metrics,
            "historical_performance": predictor.historical_performance,
            "prediction_confidence": predictor.prediction_confidence,
            "market_regime": predictor.market_regime,
            "market_status": get_market_status()[1],
            "data_analysis": {
                "total_data_points": len(historical_data),
                "features_used": len(predictor.feature_columns),
                "targets_predicted": len(predictor.targets) if hasattr(predictor,'targets') else 4
            },
            "insight": f"AI predicts {change_percent:+.2f}% movement for {symbol}. {recommendation}"
        }
        return jsonify(response)
    except Exception as e:
        print("Predict endpoint error:", e)
        return provide_fallback_prediction((request.get_json() or {}).get('symbol','SPY').upper(), 100.0, None)

def provide_fallback_prediction(symbol, current_price, historical_data):
    try:
        if current_price is None:
            base_prices = {'AAPL':182,'MSFT':407,'GOOGL':172,'AMZN':178,'TSLA':175,'SPY':445}
            current_price = base_prices.get(symbol, 100.0)
        has_large_history = False
        try:
            if historical_data is not None and hasattr(historical_data,'__len__'):
                has_large_history = len(historical_data) > 100
        except:
            has_large_history = False
        change_percent = (random.random()-0.5)*8
        predicted_price = current_price * (1 + change_percent/100)
        confidence = 75 if has_large_history else 65
        return jsonify({
            "symbol": symbol,
            "current_price": round(float(current_price),2),
            "prediction_date": get_next_trading_day(),
            "predicted_prices":{
                "open": round(current_price*(1+(random.random()-0.5)*0.01),2),
                "high": round(current_price*(1+random.random()*0.03),2),
                "low": round(current_price*(1-random.random()*0.02),2),
                "close": round(predicted_price,2)
            },
            "change_percent": round(change_percent,2),
            "confidence": {"Open":confidence-5,"High":confidence-8,"Low":confidence-6,"Close":confidence},
            "confidence_level": confidence,
            "risk_level": get_risk_level(change_percent,0.1),
            "recommendation": get_trading_recommendation(change_percent, get_risk_level(change_percent,0.1), 0.1),
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
def not_found(error): return jsonify({"error":"Page not found"}), 404
@server.errorhandler(500)
def internal_error(error): return jsonify({"error":"Internal server error"}), 500

# ---------------- Run ----------------
if __name__ == '__main__':
    print("Starting Flask app with model persistence...")
    os.makedirs('templates', exist_ok=True); os.makedirs('static', exist_ok=True)
    port = int(os.environ.get('PORT', 8080))
    server.run(host='0.0.0.0', port=port, debug=True, threaded=True)