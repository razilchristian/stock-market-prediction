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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import ta  # Technical analysis library
import shap

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

# ========= ADVANCED STOCK PREDICTOR WITH EXPLAINABILITY =========
class AdvancedStockPredictor:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR()
        }
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns = []
        self.best_model = None
        self.feature_importance = {}
        self.prediction_history = []
        self.model_health_metrics = {}
        
    def create_advanced_features(self, df):
        """Create comprehensive technical indicators and features with market regimes"""
        try:
            print("🔄 Creating advanced technical indicators...")
            
            # Ensure numeric columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Basic price features
            df['Price_Range'] = (df['High'] - df['Low']) / df['Open']
            df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Open']
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Close_Open_Ratio'] = df['Close'] / df['Open']
            
            # Returns and momentum
            for lag in [1, 2, 3, 5, 10]:
                df[f'Return_{lag}'] = df['Close'].pct_change(lag)
                df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            
            # Volatility features (multiple timeframes)
            for window in [5, 10, 20, 50]:
                df[f'Volatility_{window}'] = df['Return_1'].rolling(window).std()
                df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window).mean()
                df[f'Rolling_Min_{window}'] = df['Close'].rolling(window).min()
                df[f'Rolling_Max_{window}'] = df['Close'].rolling(window).max()
            
            # RSI multiple timeframes
            for period in [6, 14, 21]:
                df[f'RSI_{period}'] = ta.momentum.RSIIndicator(df['Close'], window=period).rsi()
            
            # MACD with multiple configurations
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Histogram'] = macd.macd_diff()
            
            # Moving averages and crossovers
            for window in [5, 10, 20, 50, 100, 200]:
                df[f'SMA_{window}'] = ta.trend.SMAIndicator(df['Close'], window=window).sma_indicator()
                df[f'EMA_{window}'] = ta.trend.EMAIndicator(df['Close'], window=window).ema_indicator()
                df[f'Price_vs_SMA_{window}'] = df['Close'] / df[f'SMA_{window}']
                df[f'Price_vs_EMA_{window}'] = df['Close'] / df[f'EMA_{window}']
            
            # Market regime indicators
            df['Above_SMA_200'] = (df['Close'] > df['SMA_200']).astype(int)
            df['Golden_Cross'] = ((df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))).astype(int)
            df['Death_Cross'] = ((df['SMA_50'] < df['SMA_200']) & (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))).astype(int)
            
            # Volume indicators
            df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
            df['Volume_Change'] = df['Volume'].pct_change()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'], window=20)
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # ATR for volatility
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # Time-based features
            df['Date'] = pd.to_datetime(df['Date'])
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
            df['Is_Month_Start'] = df['Date'].dt.is_month_start.astype(int)
            
            # Market stress indicators
            df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
            df['Price_VWAP_Ratio'] = df['Close'] / df['VWAP']
            
            # Drop NaN values created by indicators
            initial_count = len(df)
            df = df.dropna()
            final_count = len(df)
            print(f"✅ Created {len(df.columns)} features. Data points: {initial_count} → {final_count} after cleaning")
            
            return df
            
        except Exception as e:
            print(f"❌ Error creating features: {e}")
            return df
    
    def prepare_advanced_data(self, df, target_days=1):
        """Prepare data for next-day prediction with multiple targets"""
        try:
            # Create targets for next trading day
            df['Target_Close'] = df['Close'].shift(-target_days)
            df['Target_High'] = df['High'].shift(-target_days)
            df['Target_Low'] = df['Low'].shift(-target_days)
            
            # Feature columns (exclude date and targets)
            exclude_cols = ['Date', 'Target_Close', 'Target_High', 'Target_Low', 'Open', 'High', 'Low', 'Close', 'Volume']
            self.feature_columns = [col for col in df.columns if col not in exclude_cols]
            
            # Remove rows where target is NaN
            df_clean = df.dropna(subset=['Target_Close', 'Target_High', 'Target_Low'])
            
            X = df_clean[self.feature_columns]
            y_close = df_clean['Target_Close']
            y_high = df_clean['Target_High']
            y_low = df_clean['Target_Low']
            
            print(f"📊 Prepared data: {len(X)} samples, {len(self.feature_columns)} features")
            return X, y_close, y_high, y_low
            
        except Exception as e:
            print(f"❌ Error preparing data: {e}")
            return None, None, None, None
    
    def train_advanced_models(self, df, target_days=1):
        """Train models for next-day price prediction with explainability"""
        try:
            print("🔄 Creating advanced features...")
            # Create features
            df_with_features = self.create_advanced_features(df)
            
            if len(df_with_features) < 100:
                return None, "Insufficient data for training (minimum 100 data points required)"
            
            print(f"📈 Training models on {len(df_with_features)} data points...")
            # Prepare data
            X, y_close, y_high, y_low = self.prepare_advanced_data(df_with_features, target_days)
            
            if X is None:
                return None, "Error in data preparation"
            
            # Split data
            X_train, X_test, y_close_train, y_close_test, y_high_train, y_high_test, y_low_train, y_low_test = train_test_split(
                X, y_close, y_high, y_low, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            results = {}
            models_trained = {}
            
            # Train models for each target
            for name, model in self.models.items():
                print(f"🤖 Training {name}...")
                
                # Train for close price
                if name == 'SVR':
                    model.fit(X_train_scaled, y_close_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_close_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_close_test, y_pred)
                mse = mean_squared_error(y_close_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_close_test, y_pred)
                
                results[name] = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2,
                    'Predictions': y_pred.tolist()
                }
                
                models_trained[name] = model
            
            # Select best model based on R2 score
            best_model_name = max(results.keys(), key=lambda x: results[x]['R2'])
            self.best_model = models_trained[best_model_name]
            self.best_model_name = best_model_name
            
            # Calculate feature importance
            self.calculate_feature_importance(X_train, y_close_train)
            
            # Store model health metrics
            self.model_health_metrics = {
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_points': len(X_train),
                'best_model': best_model_name,
                'best_r2': results[best_model_name]['R2'],
                'feature_count': len(self.feature_columns)
            }
            
            self.is_fitted = True
            self.trained_models = models_trained
            
            print(f"✅ All models trained successfully! Best model: {best_model_name}")
            return results, None
            
        except Exception as e:
            return None, f"Training error: {str(e)}"
    
    def calculate_feature_importance(self, X, y):
        """Calculate feature importance for model explainability"""
        try:
            if self.best_model_name == 'SVR':
                # For linear models, use coefficients
                if hasattr(self.best_model, 'coef_'):
                    importance = abs(self.best_model.coef_[0])
                else:
                    importance = np.ones(len(self.feature_columns))
            else:
                # For tree-based models, use built-in feature importance
                importance = self.best_model.feature_importances_
            
            self.feature_importance = dict(zip(self.feature_columns, importance))
            self.feature_importance = dict(sorted(self.feature_importance.items(), 
                                                key=lambda x: x[1], reverse=True)[:10])  # Top 10 features
            
        except Exception as e:
            print(f"⚠️ Could not calculate feature importance: {e}")
            self.feature_importance = {}
    
    def predict_next_day_prices(self, df):
        """Predict next day prices with confidence bands and scenarios"""
        if not self.is_fitted:
            return None, None, None, "Models not trained"
        
        try:
            # Create features for the latest data
            df_with_features = self.create_advanced_features(df)
            
            if len(df_with_features) == 0:
                return None, None, None, "No data available for prediction"
            
            # Get the most recent data point for prediction
            latest_data = df_with_features.iloc[-1:]
            current_price = latest_data['Close'].iloc[0]
            
            # Predict close price using best model
            if self.best_model_name == 'SVR':
                X_scaled = self.scaler.transform(latest_data[self.feature_columns])
                predicted_close = self.best_model.predict(X_scaled)[0]
            else:
                predicted_close = self.best_model.predict(latest_data[self.feature_columns])[0]
            
            # Calculate other prices based on historical patterns
            recent_data = df_with_features.tail(50)
            volatility = recent_data['Return_1'].std()
            current_volatility = recent_data['Volatility_20'].iloc[-1]
            
            # Predict other prices using typical ratios with volatility adjustment
            open_ratio = (recent_data['Open'] / recent_data['Close'].shift(1)).mean()
            high_ratio = (recent_data['High'] / recent_data['Close']).mean()
            low_ratio = (recent_data['Low'] / recent_data['Close']).mean()
            
            predicted_open = predicted_close * open_ratio
            predicted_high = predicted_close * high_ratio * (1 + volatility)
            predicted_low = predicted_close * low_ratio * (1 - volatility)
            
            # Ensure logical price relationships
            predicted_high = max(predicted_high, predicted_close, predicted_open)
            predicted_low = min(predicted_low, predicted_close, predicted_open)
            
            predictions = {
                'Open': predicted_open,
                'High': predicted_high,
                'Low': predicted_low,
                'Close': predicted_close
            }
            
            # Calculate confidence intervals based on multiple factors
            model_confidence = min(0.95, max(0.60, self.model_health_metrics.get('best_r2', 0.7)))
            volatility_penalty = current_volatility * 2
            confidence_score = max(60, (model_confidence - volatility_penalty) * 100)
            
            # Create confidence bands
            confidence_bands = {}
            range_multiplier = 1.5 + (current_volatility * 2)  # Dynamic range based on volatility
            
            for price_type in ['Open', 'High', 'Low', 'Close']:
                price = predictions[price_type]
                range_pct = volatility * range_multiplier
                
                confidence_bands[price_type] = {
                    'confidence': round(confidence_score, 1),
                    'lower_bound': price * (1 - range_pct),
                    'upper_bound': price * (1 + range_pct),
                    'range_pct': range_pct * 100
                }
            
            # Generate scenarios
            scenarios = self.generate_scenarios(predictions, current_price, volatility, recent_data)
            
            # Store prediction history for drift detection
            prediction_record = {
                'timestamp': datetime.now(),
                'symbol': 'N/A',  # Will be set by caller
                'current_price': current_price,
                'predicted_close': predicted_close,
                'confidence': confidence_score,
                'volatility': volatility
            }
            self.prediction_history.append(prediction_record)
            
            return predictions, confidence_bands, scenarios, None
            
        except Exception as e:
            return None, None, None, f"Prediction error: {str(e)}"
    
    def generate_scenarios(self, predictions, current_price, volatility, recent_data):
        """Generate multiple market scenarios"""
        base_change = ((predictions['Close'] - current_price) / current_price) * 100
        
        # Bullish scenario (lower volatility, higher gains)
        bull_volatility = volatility * 0.7
        bull_change = base_change * 1.3
        
        # Bearish scenario (higher volatility, larger losses)
        bear_volatility = volatility * 1.5
        bear_change = base_change * 0.7
        
        # Sideways scenario (low movement)
        side_volatility = volatility * 0.8
        side_change = base_change * 0.3
        
        scenarios = {
            'base': {
                'probability': 50,
                'price_change': base_change,
                'volatility': volatility,
                'description': self.get_scenario_description(base_change, volatility)
            },
            'bullish': {
                'probability': 25,
                'price_change': bull_change,
                'volatility': bull_volatility,
                'description': self.get_scenario_description(bull_change, bull_volatility)
            },
            'bearish': {
                'probability': 15,
                'price_change': bear_change,
                'volatility': bear_volatility,
                'description': self.get_scenario_description(bear_change, bear_volatility)
            },
            'sideways': {
                'probability': 10,
                'price_change': side_change,
                'volatility': side_volatility,
                'description': self.get_scenario_description(side_change, side_volatility)
            }
        }
        
        return scenarios
    
    def get_scenario_description(self, change, volatility):
        """Get descriptive scenario analysis"""
        if change > 10:
            return "STRONG BULLISH: Significant upside potential with momentum"
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
            if volatility > 0.03:
                return "VOLATILE SIDEWAYS: High volatility with minimal net movement"
            else:
                return "STABLE SIDEWAYS: Low volatility, range-bound trading"
    
    def get_risk_alerts(self, predictions, current_price, volatility, confidence):
        """Generate risk alerts based on multiple factors"""
        alerts = []
        change_pct = ((predictions['Close'] - current_price) / current_price) * 100
        
        # Price movement alerts
        if abs(change_pct) > 15:
            alerts.append({
                'level': '🔴 CRITICAL',
                'type': 'Extreme Movement',
                'message': f'Expected {change_pct:+.1f}% move - Extreme volatility expected',
                'action': 'Consider position sizing and stop-losses'
            })
        elif abs(change_pct) > 8:
            alerts.append({
                'level': '🟡 HIGH',
                'type': 'Large Movement',
                'message': f'Expected {change_pct:+.1f}% move - High volatility',
                'action': 'Monitor closely and adjust positions'
            })
        
        # Volatility alerts
        if volatility > 0.05:
            alerts.append({
                'level': '🔴 CRITICAL',
                'type': 'High Volatility',
                'message': f'Extreme market volatility detected ({volatility:.3f})',
                'action': 'Reduce position size, use tight stop-losses'
            })
        elif volatility > 0.03:
            alerts.append({
                'level': '🟡 HIGH',
                'type': 'Elevated Volatility',
                'message': f'Above-average volatility ({volatility:.3f})',
                'action': 'Exercise caution with new positions'
            })
        
        # Confidence alerts
        if confidence < 60:
            alerts.append({
                'level': '🟡 HIGH',
                'type': 'Low Confidence',
                'message': f'Prediction confidence below threshold ({confidence:.1f}%)',
                'action': 'Verify with additional analysis before trading'
            })
        
        # Model health alerts
        if len(self.prediction_history) > 10:
            recent_errors = [abs(p['predicted_close'] - p['current_price']) / p['current_price'] 
                           for p in self.prediction_history[-10:]]
            avg_error = np.mean(recent_errors)
            if avg_error > 0.05:
                alerts.append({
                    'level': '🟠 MEDIUM',
                    'type': 'Model Performance',
                    'message': f'Recent prediction accuracy declining',
                    'action': 'Consider retraining model with newer data'
                })
        
        return alerts
    
    def get_early_warning_tiers(self, change_pct, volatility, confidence):
        """Generate early warning tiers"""
        risk_score = (abs(change_pct) / 20 * 0.4) + (volatility / 0.05 * 0.4) + ((100 - confidence) / 100 * 0.2)
        
        if risk_score > 0.7:
            return {
                'tier': '🔴 TIER 3 - CRITICAL',
                'color': '#ff4444',
                'description': 'Extreme risk conditions - Avoid new positions',
                'actions': ['Close risky positions', 'Use tight stop-losses', 'Reduce exposure']
            }
        elif risk_score > 0.5:
            return {
                'tier': '🟡 TIER 2 - HIGH',
                'color': '#ffaa00',
                'description': 'Elevated risk - Exercise caution',
                'actions': ['Monitor positions closely', 'Set stop-losses', 'Consider hedging']
            }
        elif risk_score > 0.3:
            return {
                'tier': '🟠 TIER 1 - MEDIUM',
                'color': '#ff8800',
                'description': 'Moderate risk - Normal precautions',
                'actions': ['Standard position sizing', 'Monitor market conditions']
            }
        else:
            return {
                'tier': '🟢 TIER 0 - LOW',
                'color': '#00cc66',
                'description': 'Low risk - Normal trading conditions',
                'actions': ['Standard trading strategies', 'Regular monitoring']
            }
    
    def check_model_drift(self):
        """Check for model performance drift"""
        if len(self.prediction_history) < 20:
            return {'status': 'INSUFFICIENT_DATA', 'message': 'Need more prediction history'}
        
        recent = self.prediction_history[-10:]
        older = self.prediction_history[-20:-10]
        
        recent_errors = [abs(p['predicted_close'] - p['current_price']) / p['current_price'] for p in recent]
        older_errors = [abs(p['predicted_close'] - p['current_price']) / p['current_price'] for p in older]
        
        recent_avg = np.mean(recent_errors)
        older_avg = np.mean(older_errors)
        
        drift_ratio = recent_avg / older_avg if older_avg > 0 else 1.0
        
        if drift_ratio > 1.3:
            return {
                'status': '🔴 HIGH_DRIFT',
                'message': f'Model performance degraded by {(drift_ratio-1)*100:.1f}%',
                'recommendation': 'Retrain model with current market data'
            }
        elif drift_ratio > 1.1:
            return {
                'status': '🟡 MODERATE_DRIFT',
                'message': f'Minor performance drift detected ({(drift_ratio-1)*100:.1f}%)',
                'recommendation': 'Monitor closely, consider retraining soon'
            }
        else:
            return {
                'status': '🟢 STABLE',
                'message': 'Model performance stable',
                'recommendation': 'Continue monitoring'
            }

# Initialize predictor
predictor = AdvancedStockPredictor()

# ========= REAL-TIME DATA FUNCTIONS =========
@rate_limit(0.3)  # Rate limit: 1 request every 3 seconds
def get_live_stock_data(ticker):
    """Get live stock data from yfinance with proper error handling"""
    try:
        print(f"📥 Fetching LIVE data for {ticker}...")
        
        # Add random delay to avoid rate limiting
        time.sleep(random.uniform(2, 4))
        
        # Use yf.download for more reliable data fetching
        try:
            hist_data = yf.download(ticker, period="3mo", progress=False)
            if hist_data.empty:
                # Fallback to shorter period
                hist_data = yf.download(ticker, period="1mo", progress=False)
        except Exception as e:
            print(f"   Download failed: {e}")
            return None, None, f"Failed to download data for {ticker}"
        
        if hist_data.empty or len(hist_data) < 20:
            return None, None, f"Insufficient data for {ticker}"
        
        # Get current price from the latest data
        current_price = hist_data['Close'].iloc[-1]
        
        # Reset index to get Date as column
        hist_data = hist_data.reset_index()
        hist_data['Date'] = hist_data['Date'].dt.strftime('%Y-%m-%d')
        
        print(f"✅ Fetched {len(hist_data)} LIVE data points for {ticker}")
        print(f"💰 Current LIVE price: ${current_price:.2f}")
        
        return hist_data, current_price, None
        
    except Exception as e:
        print(f"❌ Live data fetching error: {e}")
        return None, None, f"Error fetching live data: {str(e)}"

def get_next_trading_day():
    """Get the next actual trading day"""
    today = datetime.now()
    
    # If today is Friday, next trading day is Monday
    if today.weekday() == 4:  # Friday
        next_day = today + timedelta(days=3)
    # If today is Saturday, next trading day is Monday
    elif today.weekday() == 5:  # Saturday
        next_day = today + timedelta(days=2)
    # If today is Sunday, next trading day is Monday
    elif today.weekday() == 6:  # Sunday
        next_day = today + timedelta(days=1)
    # For Monday-Thursday, next trading day is tomorrow
    else:
        next_day = today + timedelta(days=1)
    
    return next_day.strftime('%Y-%m-%d')

def get_last_market_date():
    """Get the last actual market trading date"""
    today = datetime.now()
    
    # If today is Monday, last trading day was Friday
    if today.weekday() == 0:  # Monday
        last_day = today - timedelta(days=3)
    # If today is Sunday, last trading day was Friday
    elif today.weekday() == 6:  # Sunday
        last_day = today - timedelta(days=2)
    # If today is Saturday, last trading day was Friday
    elif today.weekday() == 5:  # Saturday
        last_day = today - timedelta(days=1)
    # For Tuesday-Friday, last trading day was yesterday
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

def get_risk_level(change_percent, volatility):
    """Calculate risk level based on predicted change and volatility"""
    risk_score = (abs(change_percent) / 20 * 0.6) + (volatility / 0.05 * 0.4)
    
    if risk_score > 0.7:
        return "🔴 EXTREME RISK"
    elif risk_score > 0.5:
        return "🟡 HIGH RISK"
    elif risk_score > 0.3:
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

def get_trading_recommendation(change_percent, risk_level, volatility, confidence):
    """Generate trading recommendation with confidence consideration"""
    if risk_level == "🔴 EXTREME RISK":
        if change_percent > 0:
            return "🚨 EXTREME CAUTION: Very high volatility - Only very small positions with tight stop-loss"
        else:
            return "⛔ AVOID: Extreme risk conditions - Market too volatile for safe trading"
    elif risk_level == "🟡 HIGH RISK":
        if change_percent > 8:
            return "📈 CAUTIOUS BULLISH: Good upside but high risk - Use stop-losses"
        elif change_percent > 3:
            return "📈 CONSIDER BUY: Positive momentum with managed risk"
        elif change_percent < -3:
            return "📉 CONSIDER SELL: Downside risk present - Reduce exposure"
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
    """API endpoint for stock prediction using LIVE data"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL').upper()
        
        print(f"🔮 Generating LIVE predictions for {symbol}...")
        
        # Fetch LIVE historical data
        historical_data, current_price, error = get_live_stock_data(symbol)
        if error:
            return jsonify({"error": error}), 400
        
        print(f"📊 Fetched {len(historical_data)} LIVE data points for {symbol}")
        
        # Train models
        training_results, training_error = predictor.train_advanced_models(historical_data)
        if training_error:
            return jsonify({"error": training_error}), 400
        
        print("🤖 Models trained successfully on LIVE data")
        
        # Make prediction for next trading day
        predictions, confidence_bands, scenarios, pred_error = predictor.predict_next_day_prices(historical_data)
        if pred_error:
            return jsonify({"error": pred_error}), 400
        
        # Calculate metrics
        predicted_close = predictions['Close']
        change_percent = ((predicted_close - current_price) / current_price) * 100
        
        # Calculate volatility from recent data
        volatility = historical_data['Close'].pct_change().std()
        
        # Generate risk assessment
        risk_level = get_risk_level(change_percent, volatility)
        confidence = confidence_bands['Close']['confidence']
        recommendation = get_trading_recommendation(change_percent, risk_level, volatility, confidence)
        
        # Generate risk alerts
        risk_alerts = predictor.get_risk_alerts(predictions, current_price, volatility, confidence)
        
        # Get early warning tier
        warning_tier = predictor.get_early_warning_tiers(change_percent, volatility, confidence)
        
        # Check model drift
        drift_analysis = predictor.check_model_drift()
        
        # Prepare comprehensive response
        response = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "prediction_date": get_next_trading_day(),
            "last_trading_day": get_last_market_date(),
            
            # Predicted prices
            "predicted_prices": {
                "open": round(predictions['Open'], 2),
                "high": round(predictions['High'], 2),
                "low": round(predictions['Low'], 2),
                "close": round(predictions['Close'], 2)
            },
            
            "change_percent": round(change_percent, 2),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "volatility": round(volatility, 4),
            "confidence": confidence,
            
            # Advanced features
            "confidence_bands": confidence_bands,
            "scenarios": scenarios,
            "risk_alerts": risk_alerts,
            "warning_tier": warning_tier,
            "model_drift": drift_analysis,
            "feature_importance": predictor.feature_importance,
            "model_health": predictor.model_health_metrics,
            
            "model_performance": training_results.get(predictor.best_model_name, {}),
            "best_model": predictor.best_model_name,
            "market_status": get_market_status()[1],
            "data_analysis": {
                "total_data_points": len(historical_data),
                "features_used": len(predictor.feature_columns),
                "training_period": f"{historical_data['Date'].iloc[0]} to {historical_data['Date'].iloc[-1]}"
            }
        }
        
        print(f"✅ LIVE Predictions generated for {symbol}")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# ========= DASH LAYOUT =========
app.layout = html.Div([
    html.Div([
        html.H1('🚀 Advanced AI Stock Prediction Platform', 
                style={'color': '#00e6ff', 'textAlign': 'center', 'marginBottom': '10px',
                      'fontFamily': 'Inter, sans-serif', 'fontWeight': '700', 'fontSize': '2.5rem'}),
        html.P("Live Predictions • Confidence Bands • Risk Assessment • Explainable AI • Early Warning System", 
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
                placeholder='e.g., AAPL, MSFT, TSLA, GOOGL, AMZN...', 
                type='text',
                value='AAPL',
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

        html.Button("🚀 Generate Advanced AI Prediction", 
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
            html.P("Enter a stock ticker symbol and click 'Generate Advanced AI Prediction' for comprehensive analysis.",
                  style={'color': '#94a3b8', 'fontSize': '16px', 'fontFamily': 'Inter, sans-serif'})
        ]), html.Div()
    
    if not ticker:
        return html.Div([
            html.P("❌ Please enter a valid stock ticker symbol.", 
                  style={'color': '#ff4d7c', 'fontSize': '16px', 'fontFamily': 'Inter, sans-serif'})
        ]), html.Div()
    
    try:
        print(f"🔄 Starting ADVANCED prediction process for {ticker}...")
        
        # Fetch LIVE data
        historical_data, current_price, error = get_live_stock_data(ticker)
        if error:
            return html.Div([
                html.P(f"❌ Error fetching LIVE data: {error}", 
                      style={'color': '#ff4d7c', 'fontSize': '16px', 'fontFamily': 'Inter, sans-serif'})
            ]), html.Div()
        
        # Train models
        training_results, training_error = predictor.train_advanced_models(historical_data)
        if training_error:
            return html.Div([
                html.P(f"❌ Training error: {training_error}", 
                      style={'color': '#ff4d7c', 'fontSize': '16px', 'fontFamily': 'Inter, sans-serif'})
            ]), html.Div()
        
        # Make prediction
        predictions, confidence_bands, scenarios, pred_error = predictor.predict_next_day_prices(historical_data)
        if pred_error:
            return html.Div([
                html.P(f"❌ Prediction error: {pred_error}", 
                      style={'color': '#ff4d7c', 'fontSize': '16px', 'fontFamily': 'Inter, sans-serif'})
            ]), html.Div()
        
        # Calculate metrics
        predicted_close = predictions['Close']
        change_percent = ((predicted_close - current_price) / current_price) * 100
        
        # Calculate volatility
        volatility = historical_data['Close'].pct_change().std()
        
        # Generate risk assessment
        risk_level = get_risk_level(change_percent, volatility)
        confidence = confidence_bands['Close']['confidence']
        recommendation = get_trading_recommendation(change_percent, risk_level, volatility, confidence)
        
        # Generate risk alerts
        risk_alerts = predictor.get_risk_alerts(predictions, current_price, volatility, confidence)
        
        # Get early warning tier
        warning_tier = predictor.get_early_warning_tiers(change_percent, volatility, confidence)
        
        # Check model drift
        drift_analysis = predictor.check_model_drift()
        
        # Create comprehensive results display
        results_content = create_advanced_prediction_display(
            ticker, current_price, predictions, change_percent, risk_level, 
            recommendation, volatility, confidence, confidence_bands, scenarios,
            risk_alerts, warning_tier, drift_analysis, predictor.feature_importance,
            predictor.model_health_metrics, training_results, predictor.best_model_name,
            len(historical_data), len(predictor.feature_columns)
        )
        
        status = html.Div([
            html.H4(f"✅ ADVANCED AI Analysis Complete for {ticker.upper()}", 
                   style={'color': '#00ff9d', 'marginBottom': '10px', 'fontSize': '24px', 'fontWeight': '700'}),
            html.P(f"📊 Live Data Points: {len(historical_data):,} | Features Used: {len(predictor.feature_columns)} | Best Model: {predictor.best_model_name}",
                  style={'color': '#94a3b8', 'fontSize': '14px'})
        ])
        
        return status, results_content
        
    except Exception as e:
        print(f"❌ ADVANCED Prediction failed: {str(e)}")
        return html.Div([
            html.P(f"❌ ADVANCED Prediction failed: {str(e)}", 
                  style={'color': '#ff4d7c', 'fontSize': '16px', 'fontFamily': 'Inter, sans-serif'})
        ]), html.Div()

def create_advanced_prediction_display(ticker, current_price, predictions, change_percent, risk_level, 
                                     recommendation, volatility, confidence, confidence_bands, scenarios,
                                     risk_alerts, warning_tier, drift_analysis, feature_importance,
                                     model_health, training_results, best_model, data_points, features_used):
    """Create comprehensive prediction results display with all advanced features"""
    
    change_color = '#00ff9d' if change_percent > 0 else '#ff4d7c'
    trend_icon = '📈' if change_percent > 0 else '📉'
    
    return html.Div([
        # Main Prediction Summary
        html.Div([
            html.H3(f"🔮 ADVANCED AI PREDICTION - {ticker.upper()}", 
                   style={'color': '#00e6ff', 'marginBottom': '25px', 'fontSize': '28px',
                         'fontFamily': 'Inter, sans-serif', 'fontWeight': '700', 'textAlign': 'center'}),
            
            # Early Warning Tier
            html.Div([
                html.H4("🚨 EARLY WARNING SYSTEM", style={'color': warning_tier['color'], 'marginBottom': '15px', 'fontSize': '20px', 'fontWeight': '700'}),
                html.P(warning_tier['tier'], style={'color': warning_tier['color'], 'fontSize': '18px', 'fontWeight': '600', 'marginBottom': '10px'}),
                html.P(warning_tier['description'], style={'color': '#ffffff', 'fontSize': '14px', 'marginBottom': '10px'}),
                html.Ul([html.Li(action, style={'color': '#94a3b8', 'fontSize': '13px'}) for action in warning_tier['actions']],
                       style={'marginBottom': '0'})
            ], style={'padding': '20px', 'backgroundColor': '#1a1a2e', 'borderRadius': '12px', 'border': f'2px solid {warning_tier["color"]}', 'marginBottom': '25px'}),
            
            # Price Prediction Cards
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
                        html.H2(f"${predictions['Close']:.2f}", style={'color': change_color, 'margin': '10px 0', 'fontSize': '36px', 'fontWeight': '700'}),
                        html.Div([
                            html.Span(f"{trend_icon} {change_percent:+.2f}%", 
                                     style={'color': change_color, 'fontWeight': '700', 'fontSize': '18px'}),
                            html.Span(f" • Confidence: {confidence}%", 
                                     style={'color': '#fbbf24', 'fontSize': '14px', 'marginLeft': '10px'})
                        ])
                    ], style={'textAlign': 'center', 'padding': '25px'})
                ], style={'flex': '1', 'backgroundColor': '#1e293b', 'borderRadius': '12px', 'margin': '0 10px', 'border': f'2px solid {change_color}'}),
                
                html.Div([
                    html.Div([
                        html.P("RISK LEVEL", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.H3(risk_level, style={'color': '#ffa500', 'margin': '15px 0', 'fontSize': '20px', 'fontWeight': '700'}),
                        html.P(f"Volatility: {volatility:.4f}", style={'color': '#64748b', 'margin': '0', 'fontSize': '12px'})
                    ], style={'textAlign': 'center', 'padding': '25px'})
                ], style={'flex': '1', 'backgroundColor': '#1e293b', 'borderRadius': '12px', 'margin': '0 10px', 'border': '1px solid #374151'})
            ], style={'display': 'flex', 'marginBottom': '30px', 'gap': '15px'}),
            
            # Trading Recommendation
            html.Div([
                html.H4("💡 AI TRADING RECOMMENDATION", style={'color': '#00e6ff', 'marginBottom': '15px', 'fontSize': '20px', 'fontWeight': '600'}),
                html.P(recommendation, style={'color': '#ffffff', 'fontSize': '16px', 'padding': '20px', 'backgroundColor': '#1a1a2e', 'borderRadius': '8px', 'borderLeft': '4px solid #00e6ff'})
            ], style={'marginBottom': '25px'}),
            
            # Confidence Bands
            html.Div([
                html.H4("🎯 CONFIDENCE BANDS & PRICE RANGES", style={'color': '#00e6ff', 'marginBottom': '20px', 'fontSize': '20px', 'fontWeight': '600'}),
                html.Div([
                    html.Div([
                        html.P("OPEN PRICE RANGE", style={'color': '#94a3b8', 'margin': '0 0 10px 0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.Div([
                            html.Span(f"${confidence_bands['Open']['lower_bound']:.2f}", 
                                     style={'color': '#ff4d7c', 'fontWeight': '700', 'fontSize': '16px'}),
                            html.Span(" → ", style={'color': '#94a3b8', 'margin': '0 10px'}),
                            html.Span(f"${confidence_bands['Open']['upper_bound']:.2f}", 
                                     style={'color': '#00ff9d', 'fontWeight': '700', 'fontSize': '16px'})
                        ]),
                        html.P(f"Confidence: {confidence_bands['Open']['confidence']}% | Range: ±{confidence_bands['Open']['range_pct']:.1f}%", 
                              style={'color': '#fbbf24', 'margin': '5px 0 0 0', 'fontSize': '12px', 'fontWeight': '600'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#1f2937', 'borderRadius': '10px', 'margin': '0 5px'}),
                    
                    html.Div([
                        html.P("CLOSE PRICE RANGE", style={'color': '#94a3b8', 'margin': '0 0 10px 0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.Div([
                            html.Span(f"${confidence_bands['Close']['lower_bound']:.2f}", 
                                     style={'color': '#ff4d7c', 'fontWeight': '700', 'fontSize': '16px'}),
                            html.Span(" → ", style={'color': '#94a3b8', 'margin': '0 10px'}),
                            html.Span(f"${confidence_bands['Close']['upper_bound']:.2f}", 
                                     style={'color': '#00ff9d', 'fontWeight': '700', 'fontSize': '16px'})
                        ]),
                        html.P(f"Confidence: {confidence_bands['Close']['confidence']}% | Range: ±{confidence_bands['Close']['range_pct']:.1f}%", 
                              style={'color': '#fbbf24', 'margin': '5px 0 0 0', 'fontSize': '12px', 'fontWeight': '600'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#1f2937', 'borderRadius': '10px', 'margin': '0 5px'})
                ], style={'display': 'flex', 'gap': '15px', 'marginBottom': '20px'}),
            ], style={'marginBottom': '25px'}),
            
            # Market Scenarios
            html.Div([
                html.H4("📊 MARKET SCENARIOS & PROBABILITIES", style={'color': '#00e6ff', 'marginBottom': '20px', 'fontSize': '20px', 'fontWeight': '600'}),
                html.Div([
                    html.Div([
                        html.P(f"BASE SCENARIO ({scenarios['base']['probability']}%)", style={'color': '#94a3b8', 'margin': '0 0 10px 0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.P(f"Change: {scenarios['base']['price_change']:+.2f}%", style={'color': '#ffffff', 'fontSize': '16px', 'fontWeight': '600', 'marginBottom': '5px'}),
                        html.P(scenarios['base']['description'], style={'color': '#94a3b8', 'fontSize': '12px', 'margin': '0'})
                    ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'}),
                    
                    html.Div([
                        html.P(f"BULLISH ({scenarios['bullish']['probability']}%)", style={'color': '#00ff9d', 'margin': '0 0 10px 0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.P(f"Change: {scenarios['bullish']['price_change']:+.2f}%", style={'color': '#00ff9d', 'fontSize': '16px', 'fontWeight': '600', 'marginBottom': '5px'}),
                        html.P(scenarios['bullish']['description'], style={'color': '#94a3b8', 'fontSize': '12px', 'margin': '0'})
                    ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'}),
                    
                    html.Div([
                        html.P(f"BEARISH ({scenarios['bearish']['probability']}%)", style={'color': '#ff4d7c', 'margin': '0 0 10px 0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.P(f"Change: {scenarios['bearish']['price_change']:+.2f}%", style={'color': '#ff4d7c', 'fontSize': '16px', 'fontWeight': '600', 'marginBottom': '5px'}),
                        html.P(scenarios['bearish']['description'], style={'color': '#94a3b8', 'fontSize': '12px', 'margin': '0'})
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
            
            # Model Health & Explainability
            html.Div([
                html.H4("🤖 MODEL HEALTH & EXPLAINABILITY", style={'color': '#00e6ff', 'marginBottom': '20px', 'fontSize': '20px', 'fontWeight': '600'}),
                
                html.Div([
                    html.Div([
                        html.P("MODEL PERFORMANCE", style={'color': '#94a3b8', 'margin': '0 0 8px 0', 'fontSize': '12px', 'fontWeight': '600'}),
                        html.P(f"Best Model: {best_model}", style={'color': '#ffffff', 'margin': '0', 'fontSize': '14px'}),
                        html.P(f"R² Score: {training_results[best_model]['R2']:.4f}", style={'color': '#00ff9d', 'margin': '0', 'fontSize': '13px'})
                    ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'}),
                    
                    html.Div([
                        html.P("DATA QUALITY", style={'color': '#94a3b8', 'margin': '0 0 8px 0', 'fontSize': '12px', 'fontWeight': '600'}),
                        html.P(f"Data Points: {data_points:,}", style={'color': '#ffffff', 'margin': '0', 'fontSize': '14px'}),
                        html.P(f"Features: {features_used}", style={'color': '#fbbf24', 'margin': '0', 'fontSize': '13px'})
                    ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'}),
                    
                    html.Div([
                        html.P("MODEL DRIFT", style={'color': '#94a3b8', 'margin': '0 0 8px 0', 'fontSize': '12px', 'fontWeight': '600'}),
                        html.P(drift_analysis['status'], style={'color': drift_analysis['status'][:2], 'margin': '0', 'fontSize': '14px'}),
                        html.P(drift_analysis['message'], style={'color': '#94a3b8', 'margin': '0', 'fontSize': '11px'})
                    ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'})
                ], style={'display': 'flex', 'gap': '10px', 'marginBottom': '15px'}),
                
                # Feature Importance
                html.Div([
                    html.H5("🔍 TOP FEATURES INFLUENCING PREDICTION", style={'color': '#00e6ff', 'marginBottom': '15px', 'fontSize': '16px', 'fontWeight': '600'}),
                    html.Div([
                        html.Div([
                            html.P(f"{i+1}. {feature}", style={'color': '#ffffff', 'margin': '0 0 5px 0', 'fontSize': '13px'}),
                            html.Div([
                                html.Div(style={'height': '6px', 'backgroundColor': '#00e6ff', 'borderRadius': '3px', 
                                              'width': f'{importance*100:.1f}%'})
                            ], style={'width': '100%', 'backgroundColor': '#374151', 'borderRadius': '3px', 'marginBottom': '8px'})
                        ]) for i, (feature, importance) in enumerate(list(feature_importance.items())[:5])
                    ])
                ]) if feature_importance else html.Div()
                
            ], style={'marginBottom': '25px'}),
            
            # Detailed Price Predictions
            html.Div([
                html.H4("💎 DETAILED PRICE PREDICTIONS", style={'color': '#00e6ff', 'marginBottom': '20px', 'fontSize': '20px', 'fontWeight': '600'}),
                html.Div([
                    html.Div([
                        html.P("OPEN", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '12px', 'fontWeight': '600'}),
                        html.H4(f"${predictions['Open']:.2f}", style={'color': '#fbbf24', 'margin': '8px 0', 'fontSize': '18px'}),
                        html.P(f"Range: ${confidence_bands['Open']['lower_bound']:.2f} - ${confidence_bands['Open']['upper_bound']:.2f}", 
                              style={'color': '#64748b', 'margin': '0', 'fontSize': '10px'})
                    ], style={'textAlign': 'center', 'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'}),
                    
                    html.Div([
                        html.P("HIGH", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '12px', 'fontWeight': '600'}),
                        html.H4(f"${predictions['High']:.2f}", style={'color': '#00ff9d', 'margin': '8px 0', 'fontSize': '18px'}),
                        html.P(f"Range: ${confidence_bands['High']['lower_bound']:.2f} - ${confidence_bands['High']['upper_bound']:.2f}", 
                              style={'color': '#64748b', 'margin': '0', 'fontSize': '10px'})
                    ], style={'textAlign': 'center', 'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'}),
                    
                    html.Div([
                        html.P("LOW", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '12px', 'fontWeight': '600'}),
                        html.H4(f"${predictions['Low']:.2f}", style={'color': '#ff4d7c', 'margin': '8px 0', 'fontSize': '18px'}),
                        html.P(f"Range: ${confidence_bands['Low']['lower_bound']:.2f} - ${confidence_bands['Low']['upper_bound']:.2f}", 
                              style={'color': '#64748b', 'margin': '0', 'fontSize': '10px'})
                    ], style={'textAlign': 'center', 'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'}),
                    
                    html.Div([
                        html.P("CLOSE", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '12px', 'fontWeight': '600'}),
                        html.H4(f"${predictions['Close']:.2f}", style={'color': change_color, 'margin': '8px 0', 'fontSize': '18px'}),
                        html.P(f"Range: ${confidence_bands['Close']['lower_bound']:.2f} - ${confidence_bands['Close']['upper_bound']:.2f}", 
                              style={'color': '#64748b', 'margin': '0', 'fontSize': '10px'})
                    ], style={'textAlign': 'center', 'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'})
                ], style={'display': 'flex', 'gap': '10px', 'marginBottom': '10px'})
            ], style={'marginBottom': '20px'})
            
        ], style={
            'backgroundColor': '#111827', 
            'padding': '30px', 
            'borderRadius': '16px',
            'border': '1px solid #2a2a4a',
            'marginBottom': '30px'
        })
    ])

# Serve static files
@server.route('/assets/<path:path>')
def send_assets(path):
    return send_from_directory('assets', path)

# Serve templates
@server.route('/templates/<path:path>')
def send_templates(path):
    return send_from_directory('templates', path)

# Main entry
if __name__ == '__main__':
    print("🚀 Starting Advanced AI Stock Prediction Dashboard...")
    print("📊 Access at: http://localhost:8080/")
    print("🤖 Prediction Page: http://localhost:8080/prediction")
    print("🔮 Features: Live Predictions • Confidence Bands • Risk Assessment • Explainable AI • Early Warning System")
    print("💡 Advanced Analytics: Market Scenarios • Risk Alerts • Model Health Monitoring • Feature Importance")
    print("🎯 Early Warning Tiers: Tier 0 (Low) to Tier 3 (Critical) Risk Assessment")
    app.run(debug=True, port=8080, host='0.0.0.0')