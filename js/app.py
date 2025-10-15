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

# ========= CRISIS DETECTION SYSTEM =========
class CrisisDetectionSystem:
    def __init__(self):
        self.crisis_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.crisis_features_used = []
        self.is_trained = False
        
    def detect_crises(self, data, threshold=0.05):
        """Detect crisis periods based on price movements"""
        data = data.copy()
        data['Crisis'] = (data['Return'].abs() > threshold).astype(int)
        return data
    
    def prepare_crisis_data(self, data_with_features):
        """Prepare data for crisis detection"""
        crisis_data = self.detect_crises(data_with_features)
        
        # Select crisis features (similar to notebook logic)
        base_features = ['Open', 'High', 'Low', 'Volume', 'Volatility']
        technical_features = ['Close_Ratio_5', 'Close_Ratio_20', 'Close_Ratio_50', 
                             'Volume_MA', 'Volume_Ratio', 'Price_Range', 'HL_Ratio', 
                             'OC_Ratio', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 
                             'Momentum_5', 'Momentum_10', 'Return']
        
        # Only use features that exist in our data
        self.crisis_features_used = [f for f in (base_features + technical_features) 
                                   if f in crisis_data.columns]
        
        # Remove duplicates while preserving order
        self.crisis_features_used = list(dict.fromkeys(self.crisis_features_used))
        
        crisis_data_clean = crisis_data[self.crisis_features_used + ['Crisis']].dropna()
        
        return crisis_data_clean
    
    def train_crisis_model(self, data_with_features):
        """Train crisis detection model"""
        try:
            print("🔄 Training crisis detection model...")
            
            crisis_data = self.prepare_crisis_data(data_with_features)
            
            if len(crisis_data) < 100:
                print("❌ Insufficient data for crisis detection training")
                return False
            
            X_crisis = crisis_data[self.crisis_features_used].values
            y_crisis = crisis_data['Crisis'].values
            
            # Split data chronologically
            split_idx = int(len(X_crisis) * 0.8)
            X_train = X_crisis[:split_idx]
            X_test = X_crisis[split_idx:]
            y_train = y_crisis[:split_idx]
            y_test = y_crisis[split_idx:]
            
            self.crisis_model.fit(X_train, y_train)
            
            # Calculate accuracy
            crisis_preds = self.crisis_model.predict(X_test)
            accuracy = accuracy_score(y_test, crisis_preds)
            
            self.is_trained = True
            
            print(f"✅ Crisis detection model trained successfully!")
            print(f"📊 Accuracy: {accuracy:.4f}")
            print(f"🔧 Features used: {len(self.crisis_features_used)}")
            print(f"📈 Crisis periods detected: {crisis_data['Crisis'].sum()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Crisis model training failed: {e}")
            return False
    
    def predict_crisis_probability(self, latest_features):
        """Predict crisis probability for latest data"""
        if not self.is_trained:
            return 0.0
        
        try:
            # Ensure we have the required features
            available_features = [f for f in self.crisis_features_used if f in latest_features.columns]
            
            if not available_features:
                return 0.0
            
            crisis_proba = self.crisis_model.predict_proba(latest_features[available_features])[0, 1]
            return crisis_proba
            
        except Exception as e:
            print(f"❌ Crisis prediction failed: {e}")
            return 0.0
    
    def get_alert_level(self, probability):
        """Convert probability to traffic light system"""
        if probability > 0.7:
            return '🔴 RED', 'HIGH RISK'
        elif probability > 0.4:
            return '🟡 AMBER', 'MEDIUM RISK'
        else:
            return '🟢 GREEN', 'LOW RISK'

# Initialize crisis detection system
crisis_detector = CrisisDetectionSystem()

# ========= ENHANCED FALLBACK STOCK DATA GENERATION =========
def generate_fallback_data(ticker, base_price=None, days=2520):  # 10 years of data
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
    start_date = end_date - timedelta(days=days*1.5)  # Account for weekends
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.dayofweek < 5]  # Keep only weekdays
    dates = dates[-days:]  # Take last 'days' trading days
    
    # Generate realistic price data with long-term trends and market cycles
    prices = [base_price * random.uniform(0.8, 1.2)]  # Start with some variation
    volumes = [random.randint(5000000, 50000000)]
    
    # Market cycles simulation (bull markets, corrections, etc.)
    cycle_period = len(dates) // 4  # 4 major cycles in 10 years
    
    for i in range(1, len(dates)):
        # Determine market regime based on cycle position
        cycle_pos = i % cycle_period
        cycle_phase = cycle_pos / cycle_period
        
        if cycle_phase < 0.3:  # Early bull market
            trend = 0.0003
            volatility = 0.012
        elif cycle_phase < 0.5:  # Late bull market
            trend = 0.0005
            volatility = 0.018
        elif cycle_phase < 0.7:  # Early bear market
            trend = -0.0004
            volatility = 0.022
        else:  # Late bear market / recovery
            trend = -0.0002
            volatility = 0.016
        
        # Seasonal effects (monthly)
        current_month = dates[i].month
        if current_month in [11, 12]:  # Year-end rally
            trend += 0.0002
        elif current_month in [9, 10]:  # September/October volatility
            volatility += 0.005
        
        # Random walk with trend and volatility
        change = random.gauss(trend, volatility)
        
        # Occasional large moves (market events)
        if random.random() < 0.015:  # 1.5% chance of large move
            change += random.gauss(0, 0.06)
        
        new_price = prices[-1] * (1 + change)
        
        # Ensure realistic price bounds (stocks don't go to zero or infinity)
        new_price = max(new_price, base_price * 0.1)
        new_price = min(new_price, base_price * 10.0)
        
        prices.append(new_price)
        
        # Volume with correlation to price movement and volatility
        volume_change = random.gauss(0, 0.2)
        if abs(change) > 0.03:  # High volume on large moves
            volume_change = random.gauss(0.6, 0.3)
        elif abs(change) < 0.005:  # Low volume on small moves
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
            # Today's open is based on yesterday's close with overnight gap
            overnight_gap = random.gauss(0, 0.008)
            open_price = prices[i-1] * (1 + overnight_gap)
        
        # Daily range based on volatility
        current_volatility = volatility * (1 + random.uniform(-0.3, 0.3))
        daily_range = current_volatility * close
        
        high_price = max(open_price, close) + daily_range * random.uniform(0.4, 0.8)
        low_price = min(open_price, close) - daily_range * random.uniform(0.4, 0.8)
        
        # Ensure logical relationships
        high_price = max(high_price, close, open_price)
        low_price = min(low_price, close, open_price)
        
        # Ensure high-low range is reasonable
        if (high_price - low_price) / open_price > 0.15:  # Max 15% daily range
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
    print(f"💰 Current price: ${prices[-1]:.2f}")
    print(f"📈 Price range: ${min(prices):.2f} - ${max(prices):.2f}")
    
    return df, prices[-1], None

# ========= ENHANCED REAL-TIME DATA FUNCTIONS =========
@rate_limit(0.3)
def get_live_stock_data(ticker):
    """Get 10 years of live stock data from yfinance with multiple fallback strategies"""
    try:
        print(f"📥 Fetching 10 YEARS of LIVE data for {ticker}...")
        
        # Add delay to avoid rate limiting
        time.sleep(random.uniform(1, 2))
        
        # STRATEGY 1: Try different period formats
        periods_to_try = [
            ("10y", "1d"),   # 10 years daily
            ("5y", "1d"),    # 5 years daily  
            ("2y", "1d"),    # 2 years daily
            ("1y", "1d"),    # 1 year daily
            ("6mo", "1d"),   # 6 months daily
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
                    # Get current price
                    current_price = hist_data['Close'].iloc[-1]
                    
                    # Reset index to get Date as column
                    hist_data = hist_data.reset_index()
                    hist_data['Date'] = hist_data['Date'].dt.strftime('%Y-%m-%d')
                    
                    print(f"✅ Fetched {len(hist_data)} LIVE data points for {ticker} ({period})")
                    print(f"💰 Current LIVE price: ${current_price:.2f}")
                    print(f"📅 Data range: {hist_data['Date'].iloc[0]} to {hist_data['Date'].iloc[-1]}")
                    
                    return hist_data, current_price, None
                    
            except Exception as e:
                print(f"❌ Failed with {period}/{interval}: {e}")
                continue
        
        # STRATEGY 2: Try Ticker object with different parameters
        try:
            print("🔄 Trying alternative method with yf.Ticker...")
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period="10y", interval="1d", auto_adjust=True)
            
            if not hist_data.empty and len(hist_data) > 100:
                current_price = hist_data['Close'].iloc[-1]
                hist_data = hist_data.reset_index()
                hist_data['Date'] = hist_data['Date'].dt.strftime('%Y-%m-%d')
                
                print(f"✅ Fetched {len(hist_data)} LIVE data points using Ticker method")
                print(f"💰 Current LIVE price: ${current_price:.2f}")
                return hist_data, current_price, None
                
        except Exception as e:
            print(f"❌ Ticker method failed: {e}")
        
        # STRATEGY 3: Try with start/end dates
        try:
            print("🔄 Trying with specific date range...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*10)  # 10 years back
            
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
                hist_data['Date'] = hist_data['Date'].dt.strftime('%Y-%m-%d')
                
                print(f"✅ Fetched {len(hist_data)} LIVE data points using date range")
                print(f"💰 Current LIVE price: ${current_price:.2f}")
                return hist_data, current_price, None
                
        except Exception as e:
            print(f"❌ Date range method failed: {e}")
        
        # If all methods fail, use enhanced fallback
        print("❌ All live data methods failed, using enhanced fallback...")
        return generate_fallback_data(ticker)
        
    except Exception as e:
        print(f"❌ Live data fetching error: {e}")
        print("🔄 Using enhanced fallback data due to error...")
        return generate_fallback_data(ticker)

# ========= TECHNICAL INDICATOR FUNCTIONS =========
def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ========= ENHANCED STOCK PREDICTOR =========
class AdvancedStockPredictor:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.is_fitted = False
        self.feature_columns = []
        self.best_model = None
        self.feature_importance = {}
        self.prediction_history = []
        self.model_health_metrics = {}
        self.last_crisis_probability = 0.0
        
    def create_advanced_features(self, df):
        """Create comprehensive technical indicators with enhanced features and NaN handling"""
        try:
            print("🔄 Creating enhanced technical indicators for 10-year data...")
            
            # Ensure numeric columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert Date to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Create a copy to avoid modifying original
            df_processed = df.copy()
            
            # Basic price features (less likely to create NaN)
            df_processed['Price_Range'] = (df_processed['High'] - df_processed['Low']) / df_processed['Open']
            df_processed['Body_Size'] = abs(df_processed['Close'] - df_processed['Open']) / df_processed['Open']
            df_processed['Close_Open_Ratio'] = df_processed['Close'] / df_processed['Open']
            df_processed['HL_Ratio'] = df_processed['High'] / df_processed['Low']
            df_processed['OC_Ratio'] = df_processed['Open'] / df_processed['Close']
            
            # Enhanced returns with NaN handling
            df_processed['Return'] = df_processed['Close'].pct_change()
            for lag in [1, 2, 5, 10]:
                df_processed[f'Return_{lag}'] = df_processed['Close'].pct_change(lag)
            
            # Volatility features with proper windowing
            df_processed['Volatility'] = df_processed['Return'].rolling(window=5).std()
            for window in [10, 20, 50]:
                df_processed[f'Volatility_{window}'] = df_processed['Return'].rolling(window, min_periods=5).std()
                df_processed[f'Rolling_Mean_{window}'] = df_processed['Close'].rolling(window, min_periods=5).mean()
                df_processed[f'Price_vs_MA_{window}'] = df_processed['Close'] / df_processed[f'Rolling_Mean_{window}'] - 1
            
            # RSI with error handling
            try:
                df_processed['RSI'] = calculate_rsi(df_processed['Close'])
            except:
                df_processed['RSI'] = 50  # Default value
            
            # MACD indicators (from notebook logic)
            try:
                df_processed['MACD'] = df_processed['Close'].ewm(span=12).mean() - df_processed['Close'].ewm(span=26).mean()
                df_processed['MACD_Signal'] = df_processed['MACD'].ewm(span=9).mean()
                df_processed['MACD_Histogram'] = df_processed['MACD'] - df_processed['MACD_Signal']
            except:
                df_processed['MACD'] = 0
                df_processed['MACD_Signal'] = 0
                df_processed['MACD_Histogram'] = 0
            
            # Momentum indicators (from notebook)
            df_processed['Momentum_5'] = df_processed['Close'] / df_processed['Close'].shift(5) - 1
            df_processed['Momentum_10'] = df_processed['Close'] / df_processed['Close'].shift(10) - 1
            
            # Close ratios (from notebook)
            df_processed['Close_Ratio_5'] = df_processed['Close'] / df_processed['Rolling_Mean_5']
            df_processed['Close_Ratio_20'] = df_processed['Close'] / df_processed['Rolling_Mean_20']
            df_processed['Close_Ratio_50'] = df_processed['Close'] / df_processed['Rolling_Mean_50']
            
            # Volume features (from notebook)
            df_processed['Volume_MA'] = df_processed['Volume'].rolling(window=5).mean()
            df_processed['Volume_Ratio'] = df_processed['Volume'] / df_processed['Volume_MA']
            
            # Simple moving averages
            try:
                df_processed['SMA_20'] = ta.trend.SMAIndicator(df_processed['Close'], window=20).sma_indicator()
                df_processed['SMA_50'] = ta.trend.SMAIndicator(df_processed['Close'], window=50).sma_indicator()
            except:
                df_processed['SMA_20'] = df_processed['Close'].rolling(20, min_periods=5).mean()
                df_processed['SMA_50'] = df_processed['Close'].rolling(50, min_periods=5).mean()
            
            # Price vs moving averages
            df_processed['Price_vs_SMA_20'] = df_processed['Close'] / df_processed['SMA_20'] - 1
            df_processed['Above_SMA_50'] = (df_processed['Close'] > df_processed['SMA_50']).astype(int)
            
            # Volume features
            df_processed['Volume_Change'] = df_processed['Volume'].pct_change()
            df_processed['Volume_SMA_10'] = df_processed['Volume'].rolling(10, min_periods=5).mean()
            df_processed['Volume_Ratio_10'] = df_processed['Volume'] / df_processed['Volume_SMA_10']
            
            # Time-based features
            df_processed['DayOfWeek'] = df_processed['Date'].dt.dayofweek
            df_processed['Month'] = df_processed['Date'].dt.month
            
            # Drop NaN values created by indicators
            initial_count = len(df_processed)
            df_clean = df_processed.dropna()
            final_count = len(df_clean)
            
            nan_removed = initial_count - final_count
            if nan_removed > 0:
                print(f"⚠️ Removed {nan_removed} rows with NaN values during feature creation")
            
            print(f"✅ Created {len(df_clean.columns)} robust features")
            print(f"📊 Data points: {initial_count} → {final_count} after cleaning")
            print(f"📅 Final date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
            
            return df_clean
            
        except Exception as e:
            print(f"❌ Error creating enhanced features: {e}")
            # Return original dataframe as fallback
            return df
    
    def prepare_advanced_data(self, df, target_days=1):
        """Prepare data for next-day prediction with enhanced targets"""
        try:
            # Create targets for next trading day (OHLC)
            df['Target_Open'] = df['Open'].shift(-target_days)
            df['Target_High'] = df['High'].shift(-target_days)
            df['Target_Low'] = df['Low'].shift(-target_days)
            df['Target_Close'] = df['Close'].shift(-target_days)
            
            # Feature columns (exclude date and targets)
            exclude_cols = ['Date', 'Target_Open', 'Target_High', 'Target_Low', 'Target_Close', 
                           'Open', 'High', 'Low', 'Close', 'Volume']
            self.feature_columns = [col for col in df.columns if col not in exclude_cols]
            
            # Remove rows where any target is NaN
            df_clean = df.dropna(subset=['Target_Open', 'Target_High', 'Target_Low', 'Target_Close'])
            
            X = df_clean[self.feature_columns]
            y_open = df_clean['Target_Open']
            y_high = df_clean['Target_High']
            y_low = df_clean['Target_Low']
            y_close = df_clean['Target_Close']
            
            print(f"📊 Prepared data: {len(X)} samples, {len(self.feature_columns)} features")
            print(f"🎯 Targets: Open, High, Low, Close prices")
            
            return X, y_open, y_high, y_low, y_close
            
        except Exception as e:
            print(f"❌ Error preparing data: {e}")
            return None, None, None, None, None
    
    def train_advanced_models(self, df, target_days=1):
        """Train models for next-day OHLC price prediction with crisis detection"""
        try:
            print("🔄 Creating enhanced features for 10-year data...")
            # Create features
            df_with_features = self.create_advanced_features(df)
            
            if len(df_with_features) < 300:  # Minimum data points
                return None, f"Insufficient data for training (minimum 300 data points required, got {len(df_with_features)})"
            
            # Train crisis detection model
            crisis_trained = crisis_detector.train_crisis_model(df_with_features)
            if crisis_trained:
                print("✅ Crisis detection integrated successfully")
            
            print(f"📈 Training models on {len(df_with_features)} data points...")
            # Prepare data
            X, y_open, y_high, y_low, y_close = self.prepare_advanced_data(df_with_features, target_days)
            
            if X is None:
                return None, "Error in data preparation"
            
            # ========= CRITICAL FIX: Handle NaN values =========
            print("🔍 Checking for NaN values in features...")
            
            # Check for NaN values before splitting
            nan_count = X.isna().sum().sum()
            if nan_count > 0:
                print(f"⚠️ Found {nan_count} NaN values in features. Cleaning data...")
                
                # Remove rows with NaN values
                nan_mask = X.isna().any(axis=1)
                X_clean = X[~nan_mask].copy()
                y_open_clean = y_open[~nan_mask].copy()
                y_high_clean = y_high[~nan_mask].copy()
                y_low_clean = y_low[~nan_mask].copy()
                y_close_clean = y_close[~nan_mask].copy()
                
                print(f"📊 Removed {nan_mask.sum()} rows with NaN values")
                print(f"📊 Clean data shape: {X_clean.shape}")
                
                if len(X_clean) < 100:  # Check if we have enough data after cleaning
                    return None, f"Not enough clean data after NaN removal (minimum 100 data points required, got {len(X_clean)})"
                
                X, y_open, y_high, y_low, y_close = X_clean, y_open_clean, y_high_clean, y_low_clean, y_close_clean
            
            # Also check target variables for NaN
            target_nan_mask = y_close.isna()
            if target_nan_mask.any():
                print(f"⚠️ Found {target_nan_mask.sum()} NaN values in target. Cleaning...")
                X = X[~target_nan_mask]
                y_open = y_open[~target_nan_mask]
                y_high = y_high[~target_nan_mask]
                y_low = y_low[~target_nan_mask]
                y_close = y_close[~target_nan_mask]
                print(f"📊 Final clean data shape: {X.shape}")
            
            if len(X) < 100:
                return None, f"Not enough data after cleaning (minimum 100 data points required, got {len(X)})"
            # ========= END FIX =========
            
            # Split data chronologically (important for time series)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_open_train, y_open_test = y_open.iloc[:split_idx], y_open.iloc[split_idx:]
            y_high_train, y_high_test = y_high.iloc[:split_idx], y_high.iloc[split_idx:]
            y_low_train, y_low_test = y_low.iloc[:split_idx], y_low.iloc[split_idx:]
            y_close_train, y_close_test = y_close.iloc[:split_idx], y_close.iloc[split_idx:]
            
            print(f"📊 Training set: {len(X_train)} samples")
            print(f"📊 Test set: {len(X_test)} samples")
            
            # Final NaN check before training
            if X_train.isna().any().any() or X_test.isna().any().any():
                print("❌ CRITICAL: NaN values still present after cleaning. Using fallback training...")
                return self.fallback_training(X_train, X_test, y_close_train, y_close_test)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            results = {}
            models_trained = {}
            
            # Train models for CLOSE price (primary target)
            for name, model in self.models.items():
                print(f"🤖 Training {name} for CLOSE price...")
                
                try:
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
                    
                except Exception as e:
                    print(f"⚠️ {name} training failed: {e}")
                    # Continue with other models even if one fails
                    continue
            
            if not results:
                return None, "All models failed to train"
            
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
                'feature_count': len(self.feature_columns),
                'data_range': f"{df_with_features['Date'].min()} to {df_with_features['Date'].max()}",
                'total_years': len(df_with_features) / 252,  # Approximate years
                'nan_values_removed': nan_count,
                'crisis_detection': crisis_trained,
                'crisis_features': len(crisis_detector.crisis_features_used) if crisis_trained else 0
            }
            
            self.is_fitted = True
            self.trained_models = models_trained
            
            print(f"✅ All models trained successfully! Best model: {best_model_name}")
            print(f"📊 Model R²: {results[best_model_name]['R2']:.4f}")
            return results, None
            
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            return None, f"Training error: {str(e)}"

    def fallback_training(self, X_train, X_test, y_train, y_test):
        """Fallback training method with simplified features"""
        print("🔄 Using fallback training with simplified features...")
        
        # Use only basic features that are less likely to have NaN values
        basic_features = ['Price_Range', 'Body_Size', 'Close_Open_Ratio', 'Return_1', 'Return_5', 'Volatility_20', 'Price_vs_MA_20']
        available_features = [f for f in basic_features if f in X_train.columns]
        
        if not available_features:
            available_features = X_train.columns[:6]  # Use first 6 features
        
        X_train_simple = X_train[available_features].copy()
        X_test_simple = X_test[available_features].copy()
        
        # Fill any remaining NaN with mean values
        X_train_simple = X_train_simple.fillna(X_train_simple.mean())
        X_test_simple = X_test_simple.fillna(X_train_simple.mean())
        
        results = {}
        models_trained = {}
        
        # Try only robust models
        robust_models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        for name, model in robust_models.items():
            try:
                print(f"🤖 Training {name} (fallback mode)...")
                model.fit(X_train_simple, y_train)
                y_pred = model.predict(X_test_simple)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2,
                    'Predictions': y_pred.tolist()
                }
                
                models_trained[name] = model
                self.feature_columns = available_features
                
            except Exception as e:
                print(f"⚠️ Fallback {name} training failed: {e}")
        
        if not results:
            return None, "Fallback training also failed"
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['R2'])
        self.best_model = models_trained[best_model_name]
        self.best_model_name = best_model_name
        
        self.is_fitted = True
        self.trained_models = models_trained
        
        print(f"✅ Fallback training successful! Best model: {best_model_name}")
        return results, None
    
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
        """Predict next day OHLC prices with crisis detection"""
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
            current_date = latest_data['Date'].iloc[0]
            
            print(f"📅 Predicting for date: {current_date}")
            print(f"💰 Current price: ${current_price:.2f}")
            
            # Get crisis probability
            crisis_probability = crisis_detector.predict_crisis_probability(latest_data)
            alert_level, alert_description = crisis_detector.get_alert_level(crisis_probability)
            self.last_crisis_probability = crisis_probability
            
            print(f"🚨 Crisis Probability: {crisis_probability:.3f} ({alert_level})")
            
            # Ensure we have the required features
            missing_features = set(self.feature_columns) - set(latest_data.columns)
            if missing_features:
                print(f"⚠️ Missing features: {missing_features}. Using available features...")
                available_features = [f for f in self.feature_columns if f in latest_data.columns]
                if not available_features:
                    return None, None, None, "No common features available for prediction"
            else:
                available_features = self.feature_columns
            
            # Check for NaN in prediction data
            if latest_data[available_features].isna().any().any():
                print("⚠️ NaN values in prediction data. Using fallback prediction...")
                # Use simple average of recent data for prediction
                recent_avg_change = df_with_features['Close'].pct_change(5).mean()
                predicted_close = current_price * (1 + recent_avg_change)
            else:
                # Predict close price using best model
                if self.best_model_name == 'SVR':
                    X_scaled = self.scaler.transform(latest_data[available_features])
                    predicted_close = self.best_model.predict(X_scaled)[0]
                else:
                    predicted_close = self.best_model.predict(latest_data[available_features])[0]
            
            # Use historical patterns for other prices with enhanced logic
            recent_data = df_with_features.tail(252)  # Use 1 year of data for patterns
            
            # Calculate typical price relationships from historical data
            open_pattern = (recent_data['Open'] / recent_data['Close'].shift(1)).mean()
            high_pattern = (recent_data['High'] / recent_data['Close']).mean()
            low_pattern = (recent_data['Low'] / recent_data['Close']).mean()
            
            # Calculate current volatility for confidence bands
            volatility = recent_data['Return_1'].std()
            current_volatility = recent_data['Volatility_20'].iloc[-1] if 'Volatility_20' in recent_data.columns else volatility
            
            # Enhanced prediction with volatility adjustment
            predicted_open = predicted_close * open_pattern
            predicted_high = predicted_close * high_pattern * (1 + volatility * 0.8)
            predicted_low = predicted_close * low_pattern * (1 - volatility * 0.8)
            
            # Ensure logical price relationships
            predicted_high = max(predicted_high, predicted_close, predicted_open)
            predicted_low = min(predicted_low, predicted_close, predicted_open)
            
            # Ensure reasonable daily range
            daily_range = predicted_high - predicted_low
            max_reasonable_range = current_price * 0.1  # Max 10% daily range
            if daily_range > max_reasonable_range:
                range_ratio = max_reasonable_range / daily_range
                predicted_high = predicted_close + (predicted_high - predicted_close) * range_ratio
                predicted_low = predicted_close - (predicted_close - predicted_low) * range_ratio
            
            predictions = {
                'Open': predicted_open,
                'High': predicted_high,
                'Low': predicted_low,
                'Close': predicted_close
            }
            
            # Enhanced confidence calculation
            model_confidence = min(0.95, max(0.60, self.model_health_metrics.get('best_r2', 0.7)))
            data_quality = min(1.0, len(df_with_features) / 2000)  # More data = higher confidence
            volatility_penalty = current_volatility * 1.5
            
            confidence_score = max(50, (model_confidence * 0.6 + data_quality * 0.4 - volatility_penalty) * 100)
            
            # Create enhanced confidence bands
            confidence_bands = {}
            range_multiplier = 1.2 + (current_volatility * 3)  # Dynamic range based on volatility
            
            for price_type in ['Open', 'High', 'Low', 'Close']:
                price = predictions[price_type]
                range_pct = volatility * range_multiplier
                
                confidence_bands[price_type] = {
                    'confidence': round(confidence_score, 1),
                    'lower_bound': price * (1 - range_pct),
                    'upper_bound': price * (1 + range_pct),
                    'range_pct': range_pct * 100
                }
            
            # Generate enhanced scenarios with crisis information
            scenarios = self.generate_scenarios(predictions, current_price, volatility, recent_data)
            
            # Add crisis information to scenarios
            for scenario in scenarios.values():
                scenario['crisis_probability'] = crisis_probability
                scenario['alert_level'] = alert_level
            
            # Store prediction history with crisis info
            prediction_record = {
                'timestamp': datetime.now(),
                'symbol': 'N/A',
                'current_price': current_price,
                'predicted_close': predicted_close,
                'confidence': confidence_score,
                'volatility': volatility,
                'crisis_probability': crisis_probability,
                'alert_level': alert_level,
                'data_points': len(df_with_features)
            }
            self.prediction_history.append(prediction_record)
            
            print(f"✅ Enhanced prediction complete with crisis detection")
            print(f"📊 Using {len(df_with_features)} data points ({len(df_with_features)/252:.1f} years)")
            print(f"🎯 Confidence: {confidence_score:.1f}%")
            print(f"🚨 Crisis Alert: {alert_level} ({crisis_probability:.3f})")
            
            return predictions, confidence_bands, scenarios, None
            
        except Exception as e:
            return None, None, None, f"Prediction error: {str(e)}"
    
    def generate_scenarios(self, predictions, current_price, volatility, recent_data):
        """Generate multiple market scenarios with crisis awareness"""
        base_change = ((predictions['Close'] - current_price) / current_price) * 100
        
        # Get crisis probability for scenario adjustments
        crisis_prob = self.last_crisis_probability
        
        # Adjust scenarios based on crisis probability
        crisis_multiplier = 1.0 + crisis_prob * 2  # Higher crisis prob = more extreme scenarios
        
        # Bullish scenario
        bull_volatility = volatility * (0.7 / crisis_multiplier)
        bull_change = base_change * (1.3 * crisis_multiplier)
        
        # Bearish scenario
        bear_volatility = volatility * (1.5 * crisis_multiplier)
        bear_change = base_change * (0.7 / crisis_multiplier)
        
        # Sideways scenario
        side_volatility = volatility * (0.8 / crisis_multiplier)
        side_change = base_change * (0.3 / crisis_multiplier)
        
        scenarios = {
            'base': {
                'probability': 50,
                'price_change': base_change,
                'volatility': volatility,
                'description': self.get_scenario_description(base_change, volatility),
                'crisis_aware': crisis_prob > 0.4
            },
            'bullish': {
                'probability': 25,
                'price_change': bull_change,
                'volatility': bull_volatility,
                'description': self.get_scenario_description(bull_change, bull_volatility),
                'crisis_aware': crisis_prob > 0.4
            },
            'bearish': {
                'probability': 15,
                'price_change': bear_change,
                'volatility': bear_volatility,
                'description': self.get_scenario_description(bear_change, bear_volatility),
                'crisis_aware': crisis_prob > 0.4
            },
            'sideways': {
                'probability': 10,
                'price_change': side_change,
                'volatility': side_volatility,
                'description': self.get_scenario_description(side_change, side_volatility),
                'crisis_aware': crisis_prob > 0.4
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
        """Generate risk alerts based on multiple factors including crisis detection"""
        alerts = []
        change_pct = ((predictions['Close'] - current_price) / current_price) * 100
        
        # Get crisis probability from latest prediction
        crisis_prob = self.last_crisis_probability
        
        # Crisis alerts
        if crisis_prob > 0.7:
            alerts.append({
                'level': '🔴 CRITICAL',
                'type': 'CRISIS DETECTED',
                'message': f'High crisis probability detected ({crisis_prob:.3f}) - Extreme market conditions',
                'action': 'AVOID new positions, use maximum risk management'
            })
        elif crisis_prob > 0.5:
            alerts.append({
                'level': '🟡 HIGH',
                'type': 'ELEVATED CRISIS RISK',
                'message': f'Moderate crisis probability ({crisis_prob:.3f}) - High volatility expected',
                'action': 'Reduce position size significantly, use tight stop-losses'
            })
        elif crisis_prob > 0.3:
            alerts.append({
                'level': '🟠 MEDIUM',
                'type': 'INCREASED CRISIS RISK',
                'message': f'Elevated crisis watch ({crisis_prob:.3f})',
                'action': 'Exercise caution with new positions'
            })
        
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

# ========= UTILITY FUNCTIONS =========
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
    """API endpoint for stock prediction with crisis detection"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'SPY').upper()
        
        print(f"🔮 Generating LIVE predictions with crisis detection for {symbol}...")
        
        # Fetch LIVE historical data with fallback
        historical_data, current_price, error = get_live_stock_data(symbol)
        if error:
            return jsonify({"error": error}), 400
        
        print(f"📊 Using {len(historical_data)} data points for {symbol}")
        
        # Train models
        training_results, training_error = predictor.train_advanced_models(historical_data)
        if training_error:
            return jsonify({"error": training_error}), 400
        
        print("🤖 Models trained successfully")
        
        # Make prediction for next trading day
        predictions, confidence_bands, scenarios, pred_error = predictor.predict_next_day_prices(historical_data)
        if pred_error:
            return jsonify({"error": pred_error}), 400
        
        # Get crisis probability from latest prediction
        crisis_prob = predictor.last_crisis_probability
        alert_level, alert_description = crisis_detector.get_alert_level(crisis_prob)
        
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
        
        # Prepare comprehensive response with crisis info
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
            
            # Crisis detection
            "crisis_detection": {
                "probability": round(crisis_prob, 3),
                "alert_level": alert_level,
                "alert_description": alert_description
            },
            
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
                "crisis_features": len(crisis_detector.crisis_features_used) if crisis_detector.is_trained else 0,
                "training_period": f"{historical_data['Date'].iloc[0]} to {historical_data['Date'].iloc[-1]}"
            }
        }
        
        print(f"✅ Predictions generated for {symbol} with crisis detection")
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
        html.P("10-Year Data • Live Predictions • Crisis Detection • Confidence Bands • Risk Assessment • Explainable AI", 
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
                value='SPY',  # Default to SPY for reliability
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
        
        # Fetch data with fallback
        historical_data, current_price, error = get_live_stock_data(ticker)
        if error:
            return html.Div([
                html.P(f"❌ Error: {error}", 
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
        
        # Get crisis probability
        crisis_prob = predictor.last_crisis_probability
        alert_level, alert_description = crisis_detector.get_alert_level(crisis_prob)
        
        # Create comprehensive results display
        results_content = create_advanced_prediction_display(
            ticker, current_price, predictions, change_percent, risk_level, 
            recommendation, volatility, confidence, confidence_bands, scenarios,
            risk_alerts, warning_tier, drift_analysis, predictor.feature_importance,
            predictor.model_health_metrics, training_results, predictor.best_model_name,
            len(historical_data), len(predictor.feature_columns), crisis_prob, alert_level, alert_description
        )
        
        status = html.Div([
            html.H4(f"✅ ADVANCED AI Analysis Complete for {ticker.upper()}", 
                   style={'color': '#00ff9d', 'marginBottom': '10px', 'fontSize': '24px', 'fontWeight': '700'}),
            html.P(f"📊 Data Points: {len(historical_data):,} | Features Used: {len(predictor.feature_columns)} | Best Model: {predictor.best_model_name} | Crisis Detection: {'✅ ACTIVE' if crisis_detector.is_trained else '❌ INACTIVE'}",
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
                                     model_health, training_results, best_model, data_points, features_used,
                                     crisis_prob, alert_level, alert_description):
    """Create comprehensive prediction results display with crisis detection"""
    
    change_color = '#00ff9d' if change_percent > 0 else '#ff4d7c'
    trend_icon = '📈' if change_percent > 0 else '📉'
    
    return html.Div([
        # Main Prediction Summary
        html.Div([
            html.H3(f"🔮 ADVANCED AI PREDICTION - {ticker.upper()}", 
                   style={'color': '#00e6ff', 'marginBottom': '25px', 'fontSize': '28px',
                         'fontFamily': 'Inter, sans-serif', 'fontWeight': '700', 'textAlign': 'center'}),
            
            # Crisis Detection Banner
            html.Div([
                html.H4("🚨 CRISIS DETECTION SYSTEM", style={'color': alert_level[:2], 'marginBottom': '15px', 'fontSize': '20px', 'fontWeight': '700'}),
                html.P(f"{alert_level} - {alert_description}", style={'color': alert_level[:2], 'fontSize': '18px', 'fontWeight': '600', 'marginBottom': '10px'}),
                html.P(f"Crisis Probability: {crisis_prob:.3f}", style={'color': '#ffffff', 'fontSize': '16px', 'marginBottom': '10px'}),
                html.P("Advanced machine learning model detecting potential market crisis conditions", 
                      style={'color': '#94a3b8', 'fontSize': '14px', 'marginBottom': '0'})
            ], style={'padding': '20px', 'backgroundColor': '#1a1a2e', 'borderRadius': '12px', 
                     'border': f'2px solid {alert_level[:2]}', 'marginBottom': '25px'}),
            
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
                        html.P("CRISIS DETECTION", style={'color': '#94a3b8', 'margin': '0 0 8px 0', 'fontSize': '12px', 'fontWeight': '600'}),
                        html.P(f"Status: {'✅ ACTIVE' if crisis_detector.is_trained else '❌ INACTIVE'}", 
                              style={'color': '#00ff9d' if crisis_detector.is_trained else '#ff4d7c', 'margin': '0', 'fontSize': '14px'}),
                        html.P(f"Features: {len(crisis_detector.crisis_features_used)}", 
                              style={'color': '#fbbf24', 'margin': '0', 'fontSize': '13px'})
                    ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#1f2937', 'borderRadius': '8px', 'margin': '0 5px'}),
                    
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
    print("🚀 Starting Enhanced AI Stock Prediction Platform...")
    print("📊 Features: 10-Year Data • Live Predictions • Crisis Detection • Confidence Bands • Risk Assessment • Explainable AI")
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