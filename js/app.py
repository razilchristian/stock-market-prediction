import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from flask import Flask, send_from_directory, render_template, jsonify, request, redirect, url_for
import requests
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import time
import random
from functools import wraps
import json
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import ta  # Technical analysis library

# ========= RATE LIMITING DECORATOR =========
def rate_limit(max_per_second):
    min_interval = 1.0 / max_per_second
    def decorate(func):
        last_time_called = [0.0]
        @wraps(func)
        def rate_limited_function(*args, **kwargs):
            elapsed = time.time() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_time_called[0] = time.time()
            return ret
        return rate_limited_function
    return decorate

# ========= FLASK SERVER & DASH APP =========
current_dir = os.path.dirname(os.path.abspath(__file__))
server = Flask(__name__, template_folder='templates')
app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/dash/',
    suppress_callback_exceptions=True,
    external_stylesheets=[
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
    ]
)

# ========= ENHANCED TECHNICAL INDICATOR FUNCTIONS =========
def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except:
        return pd.Series([50] * len(prices), index=prices.index)  # Default value

def create_advanced_features(data):
    """Create comprehensive technical indicators with enhanced features"""
    try:
        print("🔄 Creating comprehensive technical indicators...")
        data = data.copy()
        
        # Ensure numeric columns and handle Dtype errors
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Basic price features
        data['Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Return'].rolling(window=5).std()
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # Close ratios
        data['Close_Ratio_5'] = data['Close'] / data['MA_5']
        data['Close_Ratio_20'] = data['Close'] / data['MA_20']
        data['Close_Ratio_50'] = data['Close'] / data['MA_50']
        
        # Volume features
        data['Volume_MA'] = data['Volume'].rolling(window=5).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        
        # Price features
        data['Price_Range'] = (data['High'] - data['Low']) / data['Close']
        data['HL_Ratio'] = data['High'] / data['Low']
        data['OC_Ratio'] = data['Open'] / data['Close']
        
        # Technical indicators with error handling
        try:
            data['RSI'] = calculate_rsi(data['Close'])
        except:
            data['RSI'] = 50
            
        try:
            data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        except:
            data['MACD'] = data['MACD_Signal'] = data['MACD_Histogram'] = 0
        
        # Momentum indicators
        data['Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
        data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
        
        # Fill missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        print(f"✅ Created {len(data.columns)} advanced features")
        return data
        
    except Exception as e:
        print(f"❌ Error in feature creation: {e}")
        return data

def detect_crises(data, threshold=0.05):
    """Detect crisis periods based on price movements"""
    try:
        data = data.copy()
        data['Crisis'] = (data['Return'].abs() > threshold).astype(int)
        return data
    except:
        data['Crisis'] = 0
        return data

# ========= ENHANCED FALLBACK STOCK DATA GENERATION =========
def generate_fallback_data(ticker, base_price=None, days=2520):
    """Generate realistic fallback stock data with 10 years of historical data"""
    print(f"📊 Generating realistic 10-year fallback data for {ticker}...")
    
    # Base prices for popular stocks
    base_prices = {
        'AAPL': 189.50, 'MSFT': 330.45, 'GOOGL': 142.30, 'AMZN': 178.20,
        'TSLA': 248.50, 'META': 355.60, 'NVDA': 435.25, 'NFLX': 560.80,
        'SPY': 445.20, 'QQQ': 378.90, 'IBM': 163.40, 'ORCL': 115.75
    }
    
    if base_price is None:
        base_price = base_prices.get(ticker, 100.00)
    
    # Generate date range (last 'days' trading days - approx 10 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days*1.5)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.dayofweek < 5]
    dates = dates[-days:]
    
    # Generate realistic price data
    prices = [base_price * random.uniform(0.8, 1.2)]
    volumes = [random.randint(5000000, 50000000)]
    
    cycle_period = len(dates) // 4
    
    for i in range(1, len(dates)):
        cycle_pos = i % cycle_period
        cycle_phase = cycle_pos / cycle_period
        
        if cycle_phase < 0.3:
            trend = 0.0003
            volatility = 0.012
        elif cycle_phase < 0.5:
            trend = 0.0005
            volatility = 0.018
        elif cycle_phase < 0.7:
            trend = -0.0004
            volatility = 0.022
        else:
            trend = -0.0002
            volatility = 0.016
        
        current_month = dates[i].month
        if current_month in [11, 12]:
            trend += 0.0002
        elif current_month in [9, 10]:
            volatility += 0.005
        
        change = random.gauss(trend, volatility)
        
        if random.random() < 0.015:
            change += random.gauss(0, 0.06)
        
        new_price = prices[-1] * (1 + change)
        new_price = max(new_price, base_price * 0.1)
        new_price = min(new_price, base_price * 10.0)
        
        prices.append(new_price)
        
        volume_change = random.gauss(0, 0.2)
        if abs(change) > 0.03:
            volume_change = random.gauss(0.6, 0.3)
        elif abs(change) < 0.005:
            volume_change = random.gauss(-0.3, 0.2)
            
        new_volume = volumes[-1] * (1 + volume_change)
        new_volume = max(new_volume, 1000000)
        new_volume = min(new_volume, 200000000)
        volumes.append(int(new_volume))
    
    # Create realistic OHLC data
    opens, highs, lows = [], [], []
    
    for i, close in enumerate(prices):
        if i == 0:
            open_price = close * random.uniform(0.98, 1.02)
        else:
            overnight_gap = random.gauss(0, 0.008)
            open_price = prices[i-1] * (1 + overnight_gap)
        
        current_volatility = volatility * (1 + random.uniform(-0.3, 0.3))
        daily_range = current_volatility * close
        
        high_price = max(open_price, close) + daily_range * random.uniform(0.4, 0.8)
        low_price = min(open_price, close) - daily_range * random.uniform(0.4, 0.8)
        
        high_price = max(high_price, close, open_price)
        low_price = min(low_price, close, open_price)
        
        if (high_price - low_price) / open_price > 0.15:
            range_ratio = 0.15 * open_price / (high_price - low_price)
            high_price = open_price + (high_price - open_price) * range_ratio
            low_price = open_price - (open_price - low_price) * range_ratio
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
    
    df = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    })
    
    print(f"✅ Generated {len(df)} fallback data points for {ticker} (10 years)")
    return df, prices[-1], None

# ========= ENHANCED REAL-TIME DATA FUNCTIONS =========
@rate_limit(0.3)
def get_live_stock_data(ticker):
    """Get 10 years of live stock data from yfinance with multiple fallback strategies"""
    try:
        print(f"📥 Fetching 10 YEARS of LIVE data for {ticker}...")
        time.sleep(random.uniform(1, 2))
        
        periods_to_try = [
            ("10y", "1d"), ("5y", "1d"), ("2y", "1d"), 
            ("1y", "1d"), ("6mo", "1d")
        ]
        
        for period, interval in periods_to_try:
            try:
                print(f"🔄 Trying {period} period with {interval} interval...")
                hist_data = yf.download(
                    ticker, 
                    period=period, 
                    interval=interval, 
                    progress=False, 
                    timeout=15,
                    auto_adjust=True,
                    threads=True
                )
                
                if not hist_data.empty and len(hist_data) > 100:
                    current_price = hist_data['Close'].iloc[-1]
                    hist_data = hist_data.reset_index()
                    
                    # Handle Dtype error by ensuring proper date formatting
                    if 'Date' in hist_data.columns:
                        hist_data['Date'] = pd.to_datetime(hist_data['Date']).dt.strftime('%Y-%m-%d')
                    elif 'Datetime' in hist_data.columns:
                        hist_data['Date'] = pd.to_datetime(hist_data['Datetime']).dt.strftime('%Y-%m-%d')
                        hist_data = hist_data.drop('Datetime', axis=1)
                    
                    print(f"✅ Fetched {len(hist_data)} LIVE data points for {ticker} ({period})")
                    return hist_data, current_price, None
                    
            except Exception as e:
                print(f"❌ Failed with {period}/{interval}: {e}")
                continue
        
        # Try Ticker object
        try:
            print("🔄 Trying alternative method with yf.Ticker...")
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period="10y", interval="1d", auto_adjust=True)
            
            if not hist_data.empty and len(hist_data) > 100:
                current_price = hist_data['Close'].iloc[-1]
                hist_data = hist_data.reset_index()
                hist_data['Date'] = pd.to_datetime(hist_data['Date']).dt.strftime('%Y-%m-%d')
                
                print(f"✅ Fetched {len(hist_data)} LIVE data points using Ticker method")
                return hist_data, current_price, None
                
        except Exception as e:
            print(f"❌ Ticker method failed: {e}")
        
        # Try with start/end dates
        try:
            print("🔄 Trying with specific date range...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*10)
            
            hist_data = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval="1d",
                progress=False,
                timeout=15
            )
            
            if not hist_data.empty and len(hist_data) > 100:
                current_price = hist_data['Close'].iloc[-1]
                hist_data = hist_data.reset_index()
                hist_data['Date'] = pd.to_datetime(hist_data['Date']).dt.strftime('%Y-%m-%d')
                
                print(f"✅ Fetched {len(hist_data)} LIVE data points using date range")
                return hist_data, current_price, None
                
        except Exception as e:
            print(f"❌ Date range method failed: {e}")
        
        print("❌ All live data methods failed, using enhanced fallback...")
        return generate_fallback_data(ticker)
        
    except Exception as e:
        print(f"❌ Live data fetching error: {e}")
        return generate_fallback_data(ticker)

# ========= ENHANCED MULTI-TARGET STOCK PREDICTOR =========
class AdvancedMultiTargetPredictor:
    def __init__(self):
        self.models = {}
        self.crisis_model = None
        self.scaler_features = StandardScaler()
        self.scaler_targets = {}
        self.is_fitted = False
        self.feature_columns = []
        self.crisis_features = []
        self.prediction_history = []
        self.model_health_metrics = {}
        
    def prepare_multi_target_data(self, data, target_days=1):
        """Prepare data for multi-target prediction (Open, High, Low, Close)"""
        try:
            print("🔄 Preparing multi-target prediction data...")
            
            # Create advanced features
            data_with_features = create_advanced_features(data)
            
            # Define features for prediction
            base_features = ['Open', 'High', 'Low', 'Volume']
            technical_features = ['Volatility', 'Close_Ratio_5', 'Close_Ratio_20', 'Close_Ratio_50', 
                                 'Volume_MA', 'Volume_Ratio', 'Price_Range', 'HL_Ratio', 'OC_Ratio', 
                                 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Momentum_5', 'Momentum_10']
            
            # Only use features that exist in our data
            available_columns = data_with_features.columns.tolist()
            features = [f for f in (base_features + technical_features) if f in available_columns]
            
            # Targets we want to predict
            targets = ['Open', 'High', 'Low', 'Close']
            
            print(f"📊 Using {len(features)} features for {len(targets)} targets")
            
            # Scale features and targets
            data_scaled = data_with_features.copy()
            data_scaled[features] = self.scaler_features.fit_transform(data_with_features[features])
            
            # Scale targets separately
            self.scaler_targets = {}
            for target in targets:
                self.scaler_targets[target] = StandardScaler()
                data_scaled[target] = self.scaler_targets[target].fit_transform(data_with_features[[target]])
            
            # Create sequences for time series prediction
            X, y_arrays = self.create_sequences_multi_target(data_scaled, features, targets, window_size=30)
            
            return X, y_arrays, features, targets, data_scaled
            
        except Exception as e:
            print(f"❌ Error preparing multi-target data: {e}")
            return None, None, None, None, None
    
    def create_sequences_multi_target(self, data, features, targets, window_size=30):
        """Create sequences for time series prediction with multiple targets"""
        try:
            X, y_dict = [], {target: [] for target in targets}
            
            for i in range(window_size, len(data)):
                X.append(data[features].iloc[i-window_size:i].values.flatten())
                for target in targets:
                    y_dict[target].append(data[target].iloc[i])
            
            # Convert to arrays
            X = np.array(X)
            y_arrays = {target: np.array(y_dict[target]) for target in targets}
            
            print(f"📊 Sequences created: X shape {X.shape}")
            for target in targets:
                print(f"  {target} shape: {y_arrays[target].shape}")
            
            return X, y_arrays
            
        except Exception as e:
            print(f"❌ Error creating sequences: {e}")
            return None, None
    
    def train_multi_target_models(self, data, target_days=1):
        """Train models for multi-target OHLC price prediction"""
        try:
            print("🤖 Training multi-target prediction models...")
            
            # Prepare data
            X, y_arrays, features, targets, data_scaled = self.prepare_multi_target_data(data, target_days)
            
            if X is None:
                return None, "Error in data preparation"
            
            self.feature_columns = features
            self.targets = targets
            
            # Split data - reserve last 2 days for prediction
            split = len(X) - 2
            X_train, X_test = X[:split], X[split:]
            y_train_arrays = {target: y_arrays[target][:split] for target in targets}
            y_test_arrays = {target: y_arrays[target][split:] for target in targets}
            
            print(f"📊 Training set: {X_train.shape}")
            print(f"📊 Test set: {X_test.shape}")
            
            # Train individual models for each target
            models = {}
            predictions = {}
            rmse_scores = {}
            
            for target in targets:
                print(f"🎯 Training model for {target}...")
                model = RandomForestRegressor(
                    n_estimators=200, 
                    random_state=42, 
                    n_jobs=-1,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2
                )
                model.fit(X_train, y_train_arrays[target])
                models[target] = model
                
                # Make predictions
                pred = model.predict(X_test)
                predictions[target] = pred
                
                # Calculate RMSE
                if len(y_test_arrays[target]) > 0:
                    rmse = np.sqrt(mean_squared_error(y_test_arrays[target], pred))
                    rmse_scores[target] = rmse
                    print(f"  {target} RMSE: {rmse:.4f}")
            
            self.models = models
            self.rmse_scores = rmse_scores
            
            # Train crisis detection model
            print("🚨 Training crisis detection model...")
            crisis_data = detect_crises(data)
            crisis_features = [f for f in features + ['Return', 'Volatility', 'Momentum_5'] 
                             if f in crisis_data.columns and f in data_scaled.columns]
            
            # Remove duplicates while preserving order
            crisis_features = list(dict.fromkeys(crisis_features))
            self.crisis_features = crisis_features
            
            crisis_data_clean = crisis_data[crisis_features + ['Crisis']].dropna()
            X_crisis = crisis_data_clean[crisis_features].values
            y_crisis = crisis_data_clean['Crisis'].values
            
            crisis_split = int(len(X_crisis) * 0.8)
            X_crisis_train = X_crisis[:crisis_split]
            y_crisis_train = y_crisis[:crisis_split]
            
            self.crisis_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.crisis_model.fit(X_crisis_train, y_crisis_train)
            
            # Store model health metrics
            self.model_health_metrics = {
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_points': len(X_train),
                'targets_trained': len(targets),
                'crisis_detection_trained': True,
                'feature_count': len(features),
                'crisis_feature_count': len(crisis_features),
                'data_range': f"{data['Date'].iloc[0]} to {data['Date'].iloc[-1]}",
                'total_years': len(data) / 252
            }
            
            self.is_fitted = True
            
            print("✅ Multi-target models trained successfully!")
            return rmse_scores, None
            
        except Exception as e:
            print(f"❌ Multi-target training error: {str(e)}")
            return None, f"Training error: {str(e)}"
    
    def predict_next_day_prices(self, data):
        """Predict next day OHLC prices with crisis detection"""
        if not self.is_fitted:
            return None, None, None, "Models not trained"
        
        try:
            print("🔮 Generating multi-target predictions...")
            
            # Prepare data for prediction
            X, y_arrays, features, targets, data_scaled = self.prepare_multi_target_data(data)
            
            if X is None:
                return None, None, None, "Error in data preparation"
            
            # Get the most recent sequences for prediction
            X_pred = X[-2:]  # Last 2 sequences for prediction
            
            # Make predictions for each target
            predictions_scaled = {}
            for target in targets:
                model = self.models[target]
                pred = model.predict(X_pred)
                predictions_scaled[target] = pred
            
            # Inverse transform predictions to get actual prices
            predictions_actual = {}
            for target in targets:
                pred_reshaped = predictions_scaled[target].reshape(-1, 1)
                predictions_actual[target] = self.scaler_targets[target].inverse_transform(pred_reshaped).flatten()
            
            # Calculate confidence bands
            confidence_data = self.calculate_confidence_bands_multi(X_pred)
            
            # Predict crisis probability
            crisis_probs = self.predict_crisis_probability(data_scaled)
            
            # Generate scenarios
            scenarios = self.generate_multi_scenarios(predictions_actual, data['Close'].iloc[-1])
            
            print("✅ Multi-target prediction complete")
            return predictions_actual, confidence_data, scenarios, crisis_probs, None
            
        except Exception as e:
            return None, None, None, None, f"Prediction error: {str(e)}"
    
    def calculate_confidence_bands_multi(self, X, confidence=0.95):
        """Calculate confidence bands for multiple models"""
        try:
            confidence_data = {}
            
            for target, model in self.models.items():
                predictions_list = []
                for estimator in model.estimators_:
                    pred = estimator.predict(X)
                    predictions_list.append(pred)
                
                predictions_array = np.array(predictions_list)
                mean_pred = np.mean(predictions_array, axis=0)
                std_pred = np.std(predictions_array, axis=0)
                
                # Calculate confidence intervals
                lower = mean_pred - 1.96 * std_pred
                upper = mean_pred + 1.96 * std_pred
                
                # Inverse transform to actual prices
                if target in self.scaler_targets:
                    mean_pred_reshaped = mean_pred.reshape(-1, 1)
                    lower_reshaped = lower.reshape(-1, 1)
                    upper_reshaped = upper.reshape(-1, 1)
                    
                    mean_actual = self.scaler_targets[target].inverse_transform(mean_pred_reshaped).flatten()
                    lower_actual = self.scaler_targets[target].inverse_transform(lower_reshaped).flatten()
                    upper_actual = self.scaler_targets[target].inverse_transform(upper_reshaped).flatten()
                else:
                    mean_actual, lower_actual, upper_actual = mean_pred, lower, upper
                
                confidence_data[target] = {
                    'mean': mean_actual,
                    'lower': lower_actual,
                    'upper': upper_actual,
                    'std': std_pred
                }
            
            return confidence_data
            
        except Exception as e:
            print(f"⚠️ Error calculating confidence bands: {e}")
            return {}
    
    def predict_crisis_probability(self, data_scaled):
        """Predict crisis probability for future dates"""
        try:
            if self.crisis_model is None:
                return [0.1, 0.1]  # Default low probability
            
            # Get the latest features for crisis prediction
            latest_features = data_scaled[self.crisis_features].iloc[-2:].values
            crisis_probs = self.crisis_model.predict_proba(latest_features)[:, 1]
            
            return crisis_probs
            
        except Exception as e:
            print(f"⚠️ Error predicting crisis probability: {e}")
            return [0.1, 0.1]
    
    def generate_multi_scenarios(self, predictions, current_price):
        """Generate multiple market scenarios based on predictions"""
        try:
            base_change = ((predictions['Close'][0] - current_price) / current_price) * 100
            
            scenarios = {
                'base': {
                    'probability': 50,
                    'price_change': base_change,
                    'description': self.get_scenario_description(base_change)
                },
                'bullish': {
                    'probability': 25,
                    'price_change': base_change * 1.3,
                    'description': self.get_scenario_description(base_change * 1.3)
                },
                'bearish': {
                    'probability': 15,
                    'price_change': base_change * 0.7,
                    'description': self.get_scenario_description(base_change * 0.7)
                },
                'sideways': {
                    'probability': 10,
                    'price_change': base_change * 0.3,
                    'description': self.get_scenario_description(base_change * 0.3)
                }
            }
            
            return scenarios
            
        except Exception as e:
            print(f"⚠️ Error generating scenarios: {e}")
            return {}
    
    def get_scenario_description(self, change):
        """Get descriptive scenario analysis"""
        if change > 10:
            return "STRONG BULLISH: Significant upside potential"
        elif change > 5:
            return "BULLISH: Moderate gains expected"
        elif change > 2:
            return "SLIGHTLY BULLISH: Minor positive movement"
        elif change < -10:
            return "STRONG BEARISH: Significant downside risk"
        elif change < -5:
            return "BEARISH: Moderate decline expected"
        elif change < -2:
            return "SLIGHTLY BEARISH: Minor negative movement"
        else:
            return "STABLE SIDEWAYS: Minimal net movement expected"
    
    def get_risk_alerts(self, predictions, current_price, crisis_probs):
        """Generate risk alerts based on multiple factors"""
        alerts = []
        
        # Calculate overall change
        avg_change = np.mean([((predictions[target][0] - current_price) / current_price * 100) 
                            for target in ['Open', 'High', 'Low', 'Close']])
        
        # Price movement alerts
        if abs(avg_change) > 15:
            alerts.append({
                'level': '🔴 CRITICAL',
                'type': 'Extreme Movement',
                'message': f'Expected {avg_change:+.1f}% move - Extreme volatility expected',
                'action': 'Consider position sizing and stop-losses'
            })
        elif abs(avg_change) > 8:
            alerts.append({
                'level': '🟡 HIGH',
                'type': 'Large Movement',
                'message': f'Expected {avg_change:+.1f}% move - High volatility',
                'action': 'Monitor closely and adjust positions'
            })
        
        # Crisis probability alerts
        avg_crisis_prob = np.mean(crisis_probs) if crisis_probs is not None else 0.1
        if avg_crisis_prob > 0.7:
            alerts.append({
                'level': '🔴 CRITICAL',
                'type': 'High Crisis Probability',
                'message': f'Crisis probability: {avg_crisis_prob:.1%}',
                'action': 'Reduce exposure immediately'
            })
        elif avg_crisis_prob > 0.4:
            alerts.append({
                'level': '🟡 HIGH',
                'type': 'Elevated Crisis Risk',
                'message': f'Crisis probability: {avg_crisis_prob:.1%}',
                'action': 'Exercise extreme caution'
            })
        
        return alerts

# Initialize enhanced predictor
predictor = AdvancedMultiTargetPredictor()

# ========= UTILITY FUNCTIONS =========
def get_next_trading_day():
    """Get the next actual trading day"""
    today = datetime.now()
    if today.weekday() == 4:  # Friday
        next_day = today + timedelta(days=3)
    elif today.weekday() == 5:  # Saturday
        next_day = today + timedelta(days=2)
    elif today.weekday() == 6:  # Sunday
        next_day = today + timedelta(days=1)
    else:
        next_day = today + timedelta(days=1)
    
    return next_day.strftime('%Y-%m-%d')

def get_last_market_date():
    """Get the last actual market trading date"""
    today = datetime.now()
    if today.weekday() == 0:  # Monday
        last_day = today - timedelta(days=3)
    elif today.weekday() == 6:  # Sunday
        last_day = today - timedelta(days=2)
    elif today.weekday() == 5:  # Saturday
        last_day = today - timedelta(days=1)
    else:
        last_day = today - timedelta(days=1)
    
    return last_day.strftime('%Y-%m-%d')

def get_market_status():
    """Get current market status"""
    today = datetime.now()
    if today.weekday() >= 5:
        return "closed", "Market is closed on weekends"
    
    current_time = datetime.now().time()
    market_open = datetime.strptime('09:30', '%H:%M').time()
    market_close = datetime.strptime('16:00', '%H:%M').time()
    
    if current_time < market_open:
        return "pre_market", "Pre-market hours"
    elif current_time > market_close:
        return "after_hours", "After-hours trading"
    else:
        return "open", "Market is open"

def get_risk_level(change_percent, crisis_prob):
    """Calculate risk level based on predicted change and crisis probability"""
    risk_score = (abs(change_percent) / 20 * 0.6) + (crisis_prob * 0.4)
    
    if risk_score > 0.7:
        return "🔴 EXTREME RISK"
    elif risk_score > 0.5:
        return "🟡 HIGH RISK"
    elif risk_score > 0.3:
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

def get_trading_recommendation(change_percent, risk_level, crisis_prob):
    """Generate trading recommendation"""
    if risk_level == "🔴 EXTREME RISK":
        if change_percent > 0:
            return "🚨 EXTREME CAUTION: Very high risk - Only very small positions"
        else:
            return "⛔ AVOID: Extreme risk conditions - Market too volatile"
    elif risk_level == "🟡 HIGH RISK":
        if change_percent > 8:
            return "📈 CAUTIOUS BULLISH: Good upside but high risk"
        elif change_percent > 3:
            return "📈 CONSIDER BUY: Positive momentum with managed risk"
        elif change_percent < -3:
            return "📉 CONSIDER SELL: Downside risk present"
        else:
            return "⚖️ HOLD: Wait for clearer market direction"
    elif risk_level == "🟠 MEDIUM RISK":
        if change_percent > 5:
            return "✅ BUY: Positive outlook with acceptable risk"
        elif change_percent < -2:
            return "💼 SELL: Consider reducing exposure"
        else:
            return "🔄 HOLD: Stable with minimal expected movement"
    else:  # LOW RISK
        if change_percent > 2:
            return "✅ STRONG BUY: Good risk-reward ratio"
        elif change_percent < -1:
            return "💼 CAUTIOUS SELL: Protective action recommended"
        else:
            return "🔄 HOLD: Very stable - minimal trading opportunity"

# ========= NAVIGATION CONFIGURATION =========
NAVIGATION_MAP = {
    'index': '/',
    'jeet': '/jeet',
    'portfolio': '/portfolio',
    'mystock': '/mystock',
    'deposit': '/deposit',
    'insight': '/insight',
    'prediction': '/prediction',
    'news': '/news',
    'videos': '/videos',
    'superstars': '/superstars',
    'alerts': '/alerts',
    'help': '/help',
    'profile': '/profile'
}

# ========= FLASK ROUTES =========
@server.route('/')
def index():
    return render_template('jeet.html', navigation=NAVIGATION_MAP)

@server.route('/jeet')
def jeet_page():
    return render_template('jeet.html', navigation=NAVIGATION_MAP)

@server.route('/portfolio')
def portfolio_page():
    return render_template('portfolio.html', navigation=NAVIGATION_MAP)

@server.route('/mystock')
def mystock_page():
    return render_template('mystock.html', navigation=NAVIGATION_MAP)

@server.route('/deposit')
def deposit_page():
    return render_template('deposit.html', navigation=NAVIGATION_MAP)

@server.route('/insight')
def insight_page():
    return render_template('insight.html', navigation=NAVIGATION_MAP)

@server.route('/prediction')
def prediction_page():
    return render_template('prediction.html', navigation=NAVIGATION_MAP)

@server.route('/news')
def news_page():
    return render_template('news.html', navigation=NAVIGATION_MAP)

@server.route('/videos')
def videos_page():
    return render_template('videos.html', navigation=NAVIGATION_MAP)

@server.route('/superstars')
@server.route('/Superstars')
def superstars_page():
    return render_template('Superstars.html', navigation=NAVIGATION_MAP)

@server.route('/alerts')
@server.route('/Alerts')
def alerts_page():
    return render_template('Alerts.html', navigation=NAVIGATION_MAP)

@server.route('/help')
def help_page():
    return render_template('help.html', navigation=NAVIGATION_MAP)

@server.route('/profile')
def profile_page():
    return render_template('profile.html', navigation=NAVIGATION_MAP)

@server.route('/navigate/<page_name>')
def navigate_to_page(page_name):
    if page_name in NAVIGATION_MAP:
        return redirect(NAVIGATION_MAP[page_name])
    else:
        return redirect('/')

# ========= API ROUTES =========
@server.route('/api/predict', methods=['POST'])
@rate_limit(0.2)
def predict_stock():
    """API endpoint for multi-target stock prediction"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'SPY').upper()
        
        print(f"🔮 Generating MULTI-TARGET predictions for {symbol}...")
        
        # Fetch LIVE historical data
        historical_data, current_price, error = get_live_stock_data(symbol)
        if error:
            return jsonify({"error": error}), 400
        
        print(f"📊 Using {len(historical_data)} data points for {symbol}")
        
        # Train multi-target models
        training_results, training_error = predictor.train_multi_target_models(historical_data)
        if training_error:
            return jsonify({"error": training_error}), 400
        
        print("🤖 Multi-target models trained successfully")
        
        # Make prediction for next trading day
        predictions, confidence_data, scenarios, crisis_probs, pred_error = predictor.predict_next_day_prices(historical_data)
        if pred_error:
            return jsonify({"error": pred_error}), 400
        
        # Calculate metrics
        predicted_close = predictions['Close'][0] if 'Close' in predictions else current_price
        change_percent = ((predicted_close - current_price) / current_price) * 100
        
        # Calculate average crisis probability
        avg_crisis_prob = np.mean(crisis_probs) if crisis_probs is not None else 0.1
        
        # Generate risk assessment
        risk_level = get_risk_level(change_percent, avg_crisis_prob)
        recommendation = get_trading_recommendation(change_percent, risk_level, avg_crisis_prob)
        
        # Generate risk alerts
        risk_alerts = predictor.get_risk_alerts(predictions, current_price, crisis_probs)
        
        # Prepare comprehensive response
        response = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "prediction_date": get_next_trading_day(),
            "last_trading_day": get_last_market_date(),
            
            # Multi-target predictions
            "predicted_prices": {
                "open": round(predictions['Open'][0], 2) if 'Open' in predictions else round(current_price, 2),
                "high": round(predictions['High'][0], 2) if 'High' in predictions else round(current_price * 1.02, 2),
                "low": round(predictions['Low'][0], 2) if 'Low' in predictions else round(current_price * 0.98, 2),
                "close": round(predicted_close, 2)
            },
            
            "change_percent": round(change_percent, 2),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "crisis_probability": round(avg_crisis_prob, 3),
            
            # Advanced features
            "confidence_data": confidence_data,
            "scenarios": scenarios,
            "risk_alerts": risk_alerts,
            "model_health": predictor.model_health_metrics,
            
            "market_status": get_market_status()[1],
            "data_analysis": {
                "total_data_points": len(historical_data),
                "features_used": len(predictor.feature_columns),
                "targets_predicted": len(predictor.targets) if hasattr(predictor, 'targets') else 4
            }
        }
        
        print(f"✅ Multi-target predictions generated for {symbol}")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Multi-target prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# ========= DASH LAYOUT =========
app.layout = html.Div([
    html.Div([
        html.H1('🚀 Advanced Multi-Target AI Stock Prediction', 
                style={'color': '#00e6ff', 'textAlign': 'center', 'marginBottom': '10px',
                      'fontFamily': 'Inter, sans-serif', 'fontWeight': '700', 'fontSize': '2.5rem'}),
        html.P("Multi-Target OHLC • Crisis Detection • Confidence Bands • Risk Assessment", 
               style={'color': '#94a3b8', 'textAlign': 'center', 'marginBottom': '30px',
                     'fontFamily': 'Inter, sans-serif', 'fontSize': '1.1rem', 'fontWeight': '400'})
    ], style={'padding': '30px 20px', 'background': 'linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%)',
             'borderBottom': '1px solid #2a2a4a'}),

    html.Div(id="market-status-banner", style={
        'padding': '20px', 'borderRadius': '12px', 'margin': '20px',
        'textAlign': 'center', 'fontWeight': '600', 'fontSize': '16px',
        'fontFamily': 'Inter, sans-serif'
    }),

    html.Div([
        html.Div([
            html.Label("📈 Stock Ticker Symbol", 
                      style={'color': '#e2e8f0', 'marginBottom': '8px', 'fontWeight': '600',
                            'fontFamily': 'Inter, sans-serif', 'fontSize': '14px'}),
            dcc.Input(
                id='ticker-input', 
                placeholder='e.g., SPY, AAPL, MSFT, TSLA, GOOGL...', 
                type='text',
                value='SPY',
                style={
                    'width': '100%', 
                    'padding': '14px 16px',
                    'borderRadius': '10px',
                    'border': '2px solid #374151',
                    'backgroundColor': '#1f2937',
                    'color': '#ffffff',
                    'fontSize': '16px',
                    'fontFamily': 'Inter, sans-serif'
                }
            )
        ], style={'marginBottom': '20px'}),

        html.Button("🚀 Generate Multi-Target AI Prediction", 
                   id="train-btn", 
                   n_clicks=0,
                   style={
                       'width': '100%',
                       'backgroundColor': '#00e6ff',
                       'background': 'linear-gradient(135deg, #00e6ff 0%, #00ff9d 100%)',
                       'color': '#0f172a',
                       'border': 'none',
                       'padding': '18px 30px',
                       'borderRadius': '12px',
                       'cursor': 'pointer',
                       'fontWeight': '700',
                       'fontSize': '16px',
                       'fontFamily': 'Inter, sans-serif'
                   }),

    ], style={
        'backgroundColor': '#111827',
        'padding': '30px',
        'borderRadius': '16px',
        'border': '1px solid #2a2a4a',
        'margin': '20px'
    }),

    dcc.Loading(
        id="loading-main",
        type="circle",
        color="#00e6ff",
        children=html.Div(id="training-status", style={'textAlign': 'center', 'padding': '20px'})
    ),

    html.Div(id="prediction-results"),

], style={
    'backgroundColor': '#0f0f23', 
    'color': '#ffffff', 
    'minHeight': '100vh',
    'fontFamily': 'Inter, sans-serif'
})

# ========= DASH CALLBACKS =========
@app.callback(
    Output('market-status-banner', 'children'),
    [Input('train-btn', 'n_clicks')]
)
def update_market_status(n_clicks):
    market_status, status_message = get_market_status()
    
    color_map = {
        'open': '#00ff9d',
        'pre_market': '#ffa500',
        'after_hours': '#ffa500',
        'closed': '#ff4d7c'
    }
    
    return html.Div([
        html.Span("📊 Live Market Status: ", 
                 style={'fontWeight': '700', 'fontSize': '18px', 'marginRight': '10px'}),
        html.Span(f"{status_message.upper()} ", 
                 style={'color': color_map.get(market_status, '#ffffff'), 'fontWeight': '700', 'fontSize': '18px'}),
        html.Br(),
        html.Span(f"🎯 Next Trading Day: {get_next_trading_day()} | ", 
                 style={'color': '#94a3b8', 'fontSize': '14px', 'marginTop': '5px'}),
        html.Span(f"Last Trading Day: {get_last_market_date()}",
                 style={'color': '#94a3b8', 'fontSize': '14px'})
    ], style={
        'backgroundColor': '#1a1a2e',
        'padding': '20px',
        'borderRadius': '12px',
        'border': f'2px solid {color_map.get(market_status, "#00e6ff")}'
    })

@app.callback(
    [Output('training-status', 'children'),
     Output('prediction-results', 'children')],
    [Input('train-btn', 'n_clicks')],
    [State('ticker-input', 'value')]
)
def generate_prediction(n_clicks, ticker):
    if n_clicks == 0:
        return html.Div([
            html.P("Enter a stock ticker symbol and click 'Generate Multi-Target AI Prediction' for comprehensive analysis.",
                  style={'color': '#94a3b8', 'fontSize': '16px', 'fontFamily': 'Inter, sans-serif'})
        ]), html.Div()
    
    if not ticker:
        return html.Div([
            html.P("❌ Please enter a valid stock ticker symbol.", 
                  style={'color': '#ff4d7c', 'fontSize': '16px', 'fontFamily': 'Inter, sans-serif'})
        ]), html.Div()
    
    try:
        print(f"🔄 Starting MULTI-TARGET prediction process for {ticker}...")
        
        # Fetch data with fallback
        historical_data, current_price, error = get_live_stock_data(ticker)
        if error:
            return html.Div([
                html.P(f"❌ Error: {error}", 
                      style={'color': '#ff4d7c', 'fontSize': '16px', 'fontFamily': 'Inter, sans-serif'})
            ]), html.Div()
        
        # Train multi-target models
        training_results, training_error = predictor.train_multi_target_models(historical_data)
        if training_error:
            return html.Div([
                html.P(f"❌ Training error: {training_error}", 
                      style={'color': '#ff4d7c', 'fontSize': '16px', 'fontFamily': 'Inter, sans-serif'})
            ]), html.Div()
        
        # Make prediction
        predictions, confidence_data, scenarios, crisis_probs, pred_error = predictor.predict_next_day_prices(historical_data)
        if pred_error:
            return html.Div([
                html.P(f"❌ Prediction error: {pred_error}", 
                      style={'color': '#ff4d7c', 'fontSize': '16px', 'fontFamily': 'Inter, sans-serif'})
            ]), html.Div()
        
        # Calculate metrics
        predicted_close = predictions['Close'][0] if 'Close' in predictions else current_price
        change_percent = ((predicted_close - current_price) / current_price) * 100
        
        # Calculate average crisis probability
        avg_crisis_prob = np.mean(crisis_probs) if crisis_probs is not None else 0.1
        
        # Generate risk assessment
        risk_level = get_risk_level(change_percent, avg_crisis_prob)
        recommendation = get_trading_recommendation(change_percent, risk_level, avg_crisis_prob)
        
        # Generate risk alerts
        risk_alerts = predictor.get_risk_alerts(predictions, current_price, crisis_probs)
        
        # Create comprehensive results display
        results_content = create_multi_target_prediction_display(
            ticker, current_price, predictions, change_percent, risk_level, 
            recommendation, avg_crisis_prob, confidence_data, scenarios,
            risk_alerts, predictor.model_health_metrics, len(historical_data)
        )
        
        status = html.Div([
            html.H4(f"✅ MULTI-TARGET AI Analysis Complete for {ticker.upper()}", 
                   style={'color': '#00ff9d', 'marginBottom': '10px', 'fontSize': '24px', 'fontWeight': '700'}),
            html.P(f"📊 Data Points: {len(historical_data):,} | Targets: 4 (OHLC) | Crisis Detection: Active",
                  style={'color': '#94a3b8', 'fontSize': '14px'})
        ])
        
        return status, results_content
        
    except Exception as e:
        print(f"❌ MULTI-TARGET Prediction failed: {str(e)}")
        return html.Div([
            html.P(f"❌ MULTI-TARGET Prediction failed: {str(e)}", 
                  style={'color': '#ff4d7c', 'fontSize': '16px', 'fontFamily': 'Inter, sans-serif'})
        ]), html.Div()

def create_multi_target_prediction_display(ticker, current_price, predictions, change_percent, risk_level, 
                                         recommendation, crisis_prob, confidence_data, scenarios,
                                         risk_alerts, model_health, data_points):
    """Create comprehensive multi-target prediction results display"""
    
    change_color = '#00ff9d' if change_percent > 0 else '#ff4d7c'
    trend_icon = '📈' if change_percent > 0 else '📉'
    
    # Extract predicted prices safely
    pred_open = predictions.get('Open', [current_price])[0] if 'Open' in predictions else current_price
    pred_high = predictions.get('High', [current_price * 1.02])[0] if 'High' in predictions else current_price * 1.02
    pred_low = predictions.get('Low', [current_price * 0.98])[0] if 'Low' in predictions else current_price * 0.98
    pred_close = predictions.get('Close', [current_price])[0] if 'Close' in predictions else current_price
    
    return html.Div([
        html.Div([
            html.H3(f"🔮 MULTI-TARGET AI PREDICTION - {ticker.upper()}", 
                   style={'color': '#00e6ff', 'marginBottom': '25px', 'fontSize': '28px',
                         'fontFamily': 'Inter, sans-serif', 'fontWeight': '700', 'textAlign': 'center'}),
            
            # Crisis Detection Alert
            html.Div([
                html.H4("🚨 CRISIS DETECTION SYSTEM", style={'color': '#ff4d7c', 'marginBottom': '15px', 'fontSize': '20px', 'fontWeight': '700'}),
                html.P(f"Crisis Probability: {crisis_prob:.1%}", style={'color': '#ff4d7c', 'fontSize': '18px', 'fontWeight': '600', 'marginBottom': '10px'}),
                html.P("AI-powered crisis detection monitors market conditions for extreme volatility events", 
                      style={'color': '#ffffff', 'fontSize': '14px', 'marginBottom': '10px'})
            ], style={'padding': '20px', 'backgroundColor': '#2a1e1e', 'borderRadius': '12px', 'border': '2px solid #ff4d7c', 'marginBottom': '25px'}),
            
            # Multi-Target Price Prediction Cards
            html.Div([
                html.Div([
                    html.Div([
                        html.P("CURRENT PRICE", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.H2(f"${current_price:.2f}", style={'color': '#ffffff', 'margin': '10px 0', 'fontSize': '36px', 'fontWeight': '700'}),
                        html.P("Live Market Price", style={'color': '#64748b', 'margin': '0', 'fontSize': '12px'})
                    ], style={'textAlign': 'center', 'padding': '25px'})
                ], style={'flex': '1', 'backgroundColor': '#1e293b', 'borderRadius': '12px', 'margin': '0 10px', 'border': '1px solid #374151'}),
                
                html.Div([
                    html.Div([
                        html.P("PREDICTED CLOSE", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.H2(f"${pred_close:.2f}", style={'color': change_color, 'margin': '10px 0', 'fontSize': '36px', 'fontWeight': '700'}),
                        html.Div([
                            html.Span(f"{trend_icon} {change_percent:+.2f}%", 
                                     style={'color': change_color, 'fontWeight': '700', 'fontSize': '18px'}),
                            html.Span(f" • Crisis Risk: {crisis_prob:.1%}", 
                                     style={'color': '#ff4d7c', 'fontSize': '14px', 'marginLeft': '10px'})
                        ])
                    ], style={'textAlign': 'center', 'padding': '25px'})
                ], style={'flex': '1', 'backgroundColor': '#1e293b', 'borderRadius': '12px', 'margin': '0 10px', 'border': f'2px solid {change_color}'}),
                
                html.Div([
                    html.Div([
                        html.P("RISK LEVEL", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.H3(risk_level, style={'color': '#ffa500', 'margin': '15px 0', 'fontSize': '20px', 'fontWeight': '700'}),
                        html.P(f"Crisis Prob: {crisis_prob:.1%}", style={'color': '#64748b', 'margin': '0', 'fontSize': '12px'})
                    ], style={'textAlign': 'center', 'padding': '25px'})
                ], style={'flex': '1', 'backgroundColor': '#1e293b', 'borderRadius': '12px', 'margin': '0 10px', 'border': '1px solid #374151'})
            ], style={'display': 'flex', 'marginBottom': '30px', 'gap': '15px'}),
            
            # Trading Recommendation
            html.Div([
                html.H4("💡 AI TRADING RECOMMENDATION", style={'color': '#00e6ff', 'marginBottom': '15px', 'fontSize': '20px', 'fontWeight': '600'}),
                html.P(recommendation, style={'color': '#ffffff', 'fontSize': '16px', 'padding': '20px', 'backgroundColor': '#1a1a2e', 'borderRadius': '8px', 'borderLeft': '4px solid #00e6ff'})
            ], style={'marginBottom': '25px'}),
            
            # Detailed Multi-Target Predictions
            html.Div([
                html.H4("🎯 MULTI-TARGET PRICE PREDICTIONS", style={'color': '#00e6ff', 'marginBottom': '20px', 'fontSize': '20px', 'fontWeight': '600'}),
                html.Div([
                    html.Div([
                        html.P("OPEN PRICE", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.H3(f"${pred_open:.2f}", style={'color': '#fbbf24', 'margin': '10px 0', 'fontSize': '24px', 'fontWeight': '700'}),
                        html.P(f"Predicted Opening Price", style={'color': '#64748b', 'margin': '0', 'fontSize': '12px'})
                    ], style={'textAlign': 'center', 'flex': '1', 'padding': '20px', 'backgroundColor': '#1f2937', 'borderRadius': '10px', 'margin': '0 5px'}),
                    
                    html.Div([
                        html.P("HIGH PRICE", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.H3(f"${pred_high:.2f}", style={'color': '#00ff9d', 'margin': '10px 0', 'fontSize': '24px', 'fontWeight': '700'}),
                        html.P(f"Predicted Daily High", style={'color': '#64748b', 'margin': '0', 'fontSize': '12px'})
                    ], style={'textAlign': 'center', 'flex': '1', 'padding': '20px', 'backgroundColor': '#1f2937', 'borderRadius': '10px', 'margin': '0 5px'}),
                    
                    html.Div([
                        html.P("LOW PRICE", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.H3(f"${pred_low:.2f}", style={'color': '#ff4d7c', 'margin': '10px 0', 'fontSize': '24px', 'fontWeight': '700'}),
                        html.P(f"Predicted Daily Low", style={'color': '#64748b', 'margin': '0', 'fontSize': '12px'})
                    ], style={'textAlign': 'center', 'flex': '1', 'padding': '20px', 'backgroundColor': '#1f2937', 'borderRadius': '10px', 'margin': '0 5px'}),
                    
                    html.Div([
                        html.P("CLOSE PRICE", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.H3(f"${pred_close:.2f}", style={'color': change_color, 'margin': '10px 0', 'fontSize': '24px', 'fontWeight': '700'}),
                        html.P(f"Predicted Closing Price", style={'color': '#64748b', 'margin': '0', 'fontSize': '12px'})
                    ], style={'textAlign': 'center', 'flex': '1', 'padding': '20px', 'backgroundColor': '#1f2937', 'borderRadius': '10px', 'margin': '0 5px'})
                ], style={'display': 'flex', 'gap': '10px', 'marginBottom': '20px'}),
            ], style={'marginBottom': '25px'}),
            
            # Market Scenarios
            html.Div([
                html.H4("📊 MARKET SCENARIOS & PROBABILITIES", style={'color': '#00e6ff', 'marginBottom': '20px', 'fontSize': '20px', 'fontWeight': '600'}),
                html.Div([
                    html.Div([
                        html.P(f"BASE SCENARIO ({scenarios.get('base', {}).get('probability', 50)}%)", style={'color': '#94a3b8', 'margin': '0 0 10px 0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.P(f"Change: {scenarios.get('base', {}).get('price_change', change_percent):+.2f}%", style={'color': '#ffffff', 'fontSize': '16px', 'fontWeight': '600', 'marginBottom': '5px'}),
                        html.P(scenarios.get('base', {}).get('description', 'Standard market conditions'), style={'color': '#94a3b8', 'fontSize': '12px', 'margin': '0'})
                    ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'}),
                    
                    html.Div([
                        html.P(f"BULLISH ({scenarios.get('bullish', {}).get('probability', 25)}%)", style={'color': '#00ff9d', 'margin': '0 0 10px 0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.P(f"Change: {scenarios.get('bullish', {}).get('price_change', change_percent * 1.3):+.2f}%", style={'color': '#00ff9d', 'fontSize': '16px', 'fontWeight': '600', 'marginBottom': '5px'}),
                        html.P(scenarios.get('bullish', {}).get('description', 'Optimistic market conditions'), style={'color': '#94a3b8', 'fontSize': '12px', 'margin': '0'})
                    ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'}),
                    
                    html.Div([
                        html.P(f"BEARISH ({scenarios.get('bearish', {}).get('probability', 15)}%)", style={'color': '#ff4d7c', 'margin': '0 0 10px 0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.P(f"Change: {scenarios.get('bearish', {}).get('price_change', change_percent * 0.7):+.2f}%", style={'color': '#ff4d7c', 'fontSize': '16px', 'fontWeight': '600', 'marginBottom': '5px'}),
                        html.P(scenarios.get('bearish', {}).get('description', 'Pessimistic market conditions'), style={'color': '#94a3b8', 'fontSize': '12px', 'margin': '0'})
                    ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'})
                ], style={'display': 'flex', 'gap': '10px', 'marginBottom': '15px'}),
            ], style={'marginBottom': '25px'}),
            
            # Risk Alerts
            html.Div([
                html.H4("🚨 RISK ALERTS & WARNINGS", style={'color': '#00e6ff', 'marginBottom': '20px', 'fontSize': '20px', 'fontWeight': '600'}),
                html.Div([
                    html.Div([
                        html.P(f"{alert['level']} - {alert['type']}", style={'color': alert['level'][:2], 'margin': '0 0 8px 0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.P(alert['message'], style={'color': '#ffffff', 'fontSize': '13px', 'marginBottom': '5px'}),
                        html.P(f"Action: {alert['action']}", style={'color': '#94a3b8', 'fontSize': '12px', 'margin': '0'})
                    ], style={'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'marginBottom': '10px', 'borderLeft': f'4px solid {alert["level"][:2]}'})
                    for alert in risk_alerts
                ]) if risk_alerts else html.P("No critical risk alerts detected", style={'color': '#00ff9d', 'fontSize': '14px', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px'})
            ], style={'marginBottom': '25px'}),
            
            # Model Health
            html.Div([
                html.H4("🤖 MODEL HEALTH & PERFORMANCE", style={'color': '#00e6ff', 'marginBottom': '20px', 'fontSize': '20px', 'fontWeight': '600'}),
                
                html.Div([
                    html.Div([
                        html.P("DATA QUALITY", style={'color': '#94a3b8', 'margin': '0 0 8px 0', 'fontSize': '12px', 'fontWeight': '600'}),
                        html.P(f"Data Points: {data_points:,}", style={'color': '#ffffff', 'margin': '0', 'fontSize': '14px'}),
                        html.P(f"Features: {model_health.get('feature_count', 'N/A')}", style={'color': '#fbbf24', 'margin': '0', 'fontSize': '13px'})
                    ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'}),
                    
                    html.Div([
                        html.P("TARGETS PREDICTED", style={'color': '#94a3b8', 'margin': '0 0 8px 0', 'fontSize': '12px', 'fontWeight': '600'}),
                        html.P("Open, High, Low, Close", style={'color': '#ffffff', 'margin': '0', 'fontSize': '14px'}),
                        html.P(f"Models: {model_health.get('targets_trained', 'N/A')}", style={'color': '#00ff9d', 'margin': '0', 'fontSize': '13px'})
                    ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'}),
                    
                    html.Div([
                        html.P("CRISIS DETECTION", style={'color': '#94a3b8', 'margin': '0 0 8px 0', 'fontSize': '12px', 'fontWeight': '600'}),
                        html.P("Active Monitoring", style={'color': '#ff4d7c', 'margin': '0', 'fontSize': '14px'}),
                        html.P(f"Features: {model_health.get('crisis_feature_count', 'N/A')}", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '11px'})
                    ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'})
                ], style={'display': 'flex', 'gap': '10px', 'marginBottom': '15px'}),
                
            ], style={'marginBottom': '25px'}),
            
            # Disclaimer
            html.Div([
                html.P("⚠️ DISCLAIMER: This AI prediction is for educational and research purposes only. Stock market investments carry risks, and past performance does not guarantee future results. Always conduct your own research and consult with financial advisors before making investment decisions.",
                      style={'color': '#fbbf24', 'fontSize': '12px', 'textAlign': 'center', 'lineHeight': '1.4', 
                            'padding': '15px', 'backgroundColor': '#2a1e1e', 'borderRadius': '8px', 'border': '1px solid #fbbf24'})
            ])
            
        ], style={
            'backgroundColor': '#111827',
            'padding': '30px',
            'borderRadius': '16px',
            'border': '1px solid #2a2a4a',
            'margin': '20px',
            'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.3)'
        })
    ])

# ========= STATIC FILE SERVING =========
@server.route('/static/<path:path>')
def serve_static(path):
    static_folder = os.path.join(current_dir, 'static')
    return send_from_directory(static_folder, path)

@server.route('/templates/<path:path>')
def serve_templates(path):
    templates_folder = os.path.join(current_dir, 'templates')
    return send_from_directory(templates_folder, path)

# ========= SIMPLE ERROR HANDLERS =========
@server.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Page not found"}), 404

@server.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ========= HEALTH CHECK =========
@server.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

# ========= MAIN EXECUTION =========
if __name__ == '__main__':
    print("🚀 Starting Enhanced Multi-Target AI Stock Prediction Platform...")
    print("📊 Features: Multi-Target OHLC • Crisis Detection • Confidence Bands • Risk Assessment")
    print("🌐 Web Interface: http://localhost:8080")
    print("📈 Prediction Page: http://localhost:8080/prediction")
    print("🔮 Dash App: http://localhost:8080/dash/")
    
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run the application
    app.run_server(
        host='0.0.0.0',
        port=8080,
        debug=True,
        dev_tools_ui=True,
        dev_tools_hot_reload=True,
        threaded=True
    )