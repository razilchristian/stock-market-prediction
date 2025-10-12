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
from functools import wraps
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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

# ========= ENHANCED MACHINE LEARNING MODELS =========
class AdvancedStockPredictor:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100),
            'Linear Regression': LinearRegression(),
            'SVR': SVR()
        }
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns = []
        self.best_model = None
        
    def create_advanced_features(self, df):
        """Create comprehensive technical indicators and features"""
        try:
            print("🔄 Creating advanced technical indicators...")
            
            # Ensure numeric columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Basic price features
            df['Price_Range'] = (df['High'] - df['Low']) / df['Open']
            df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Open']
            df['High_Low_Ratio'] = df['High'] / df['Low']
            
            # Returns and momentum
            for lag in [1, 2, 3, 5]:
                df[f'Return_{lag}'] = df['Close'].pct_change(lag)
                df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            
            # Volatility features
            for window in [5, 10, 20]:
                df[f'Volatility_{window}'] = df['Return_1'].rolling(window).std()
                df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window).mean()
            
            # RSI
            df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                df[f'SMA_{window}'] = ta.trend.SMAIndicator(df['Close'], window=window).sma_indicator()
                df[f'EMA_{window}'] = ta.trend.EMAIndicator(df['Close'], window=window).ema_indicator()
                df[f'Price_vs_SMA_{window}'] = df['Close'] / df[f'SMA_{window}']
            
            # Volume indicators
            df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'], window=20)
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Middle'] = bb.bollinger_mavg()
            
            # Drop NaN values created by indicators
            initial_count = len(df)
            df = df.dropna()
            final_count = len(df)
            print(f"✅ Created features. Data points: {initial_count} → {final_count} after cleaning")
            
            return df
            
        except Exception as e:
            print(f"❌ Error creating features: {e}")
            return df
    
    def prepare_advanced_data(self, df, target_days=1):
        """Prepare data for next-day prediction"""
        try:
            # Create targets for next trading day
            df['Target_Close'] = df['Close'].shift(-target_days)
            
            # Feature columns (exclude date and targets)
            exclude_cols = ['Date', 'Target_Close', 'Open', 'High', 'Low', 'Close', 'Volume']
            self.feature_columns = [col for col in df.columns if col not in exclude_cols]
            
            # Remove rows where target is NaN
            df_clean = df.dropna(subset=['Target_Close'])
            
            X = df_clean[self.feature_columns]
            y = df_clean['Target_Close']
            
            print(f"📊 Prepared data: {len(X)} samples, {len(self.feature_columns)} features")
            return X, y
            
        except Exception as e:
            print(f"❌ Error preparing data: {e}")
            return None, None
    
    def train_advanced_models(self, df, target_days=1):
        """Train models for next-day price prediction"""
        try:
            print("🔄 Creating advanced features...")
            # Create features
            df_with_features = self.create_advanced_features(df)
            
            if len(df_with_features) < 50:
                return None, "Insufficient data for training (minimum 50 data points required)"
            
            print(f"📈 Training models on {len(df_with_features)} data points...")
            # Prepare data
            X, y = self.prepare_advanced_data(df_with_features, target_days)
            
            if X is None:
                return None, "Error in data preparation"
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            results = {}
            models_trained = {}
            
            # Train models
            for name, model in self.models.items():
                print(f"🤖 Training {name}...")
                
                if name == 'SVR':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2
                }
                
                models_trained[name] = model
            
            # Select best model based on R2 score
            best_model_name = max(results.keys(), key=lambda x: results[x]['R2'])
            self.best_model = models_trained[best_model_name]
            self.best_model_name = best_model_name
            
            self.is_fitted = True
            self.trained_models = models_trained
            
            print(f"✅ All models trained successfully! Best model: {best_model_name}")
            return results, None
            
        except Exception as e:
            return None, f"Training error: {str(e)}"
    
    def predict_next_day_prices(self, df):
        """Predict next day prices"""
        if not self.is_fitted:
            return None, None, "Models not trained"
        
        try:
            # Create features for the latest data
            df_with_features = self.create_advanced_features(df)
            
            if len(df_with_features) == 0:
                return None, None, "No data available for prediction"
            
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
            
            # Predict other prices using typical ratios
            open_ratio = (recent_data['Open'] / recent_data['Close'].shift(1)).mean()
            high_ratio = (recent_data['High'] / recent_data['Close']).mean()
            low_ratio = (recent_data['Low'] / recent_data['Close']).mean()
            
            predicted_open = predicted_close * open_ratio
            predicted_high = predicted_close * high_ratio * (1 + volatility)
            predicted_low = predicted_close * low_ratio * (1 - volatility)
            
            predictions = {
                'Open': predicted_open,
                'High': predicted_high,
                'Low': predicted_low,
                'Close': predicted_close
            }
            
            # Calculate confidence intervals
            confidence_score = max(60, min(95, 80 - (volatility * 1000)))
            
            confidence_intervals = {}
            for price_type in ['Open', 'High', 'Low', 'Close']:
                price = predictions[price_type]
                range_pct = volatility * 1.5
                
                confidence_intervals[price_type] = {
                    'confidence': confidence_score,
                    'lower_bound': price * (1 - range_pct),
                    'upper_bound': price * (1 + range_pct)
                }
            
            return predictions, confidence_intervals, None
            
        except Exception as e:
            return None, None, f"Prediction error: {str(e)}"

# Initialize predictor
predictor = AdvancedStockPredictor()

# ========= REAL-TIME DATA FUNCTIONS =========
def get_live_stock_data(ticker):
    """Get live stock data from yfinance with proper error handling"""
    try:
        print(f"📥 Fetching LIVE data for {ticker}...")
        
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Try to get info first to check if stock exists
        info = stock.info
        if not info:
            return None, None, f"Stock {ticker} not found or may be delisted"
        
        # Get historical data - try different periods
        periods_to_try = ["1y", "6mo", "3mo", "1mo"]
        
        for period in periods_to_try:
            try:
                print(f"   Trying period: {period}")
                hist_data = stock.history(period=period, interval="1d")
                
                if not hist_data.empty and len(hist_data) > 30:
                    print(f"✅ Successfully fetched data with period: {period}")
                    break
            except Exception as e:
                print(f"   Period {period} failed: {e}")
                continue
        else:
            return None, None, f"Could not fetch historical data for {ticker}"
        
        # Get current price from the latest data
        if len(hist_data) > 0:
            current_price = hist_data['Close'].iloc[-1]
        else:
            # Fallback to info if historical data is empty
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price:
                return None, None, "Could not determine current price"
        
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
    if abs(change_percent) > 10 or volatility > 0.04:
        return "🔴 HIGH RISK"
    elif abs(change_percent) > 5 or volatility > 0.02:
        return "🟡 MEDIUM RISK"
    elif abs(change_percent) > 2:
        return "🟠 MODERATE RISK"
    else:
        return "🟢 LOW RISK"

def get_trading_recommendation(change_percent, risk_level, volatility):
    """Generate trading recommendation"""
    if risk_level == "🔴 HIGH RISK":
        if change_percent > 0:
            return "🚨 CAUTION: High volatility - Consider small positions with stop-loss"
        else:
            return "⛔ AVOID: High risk - Wait for better market conditions"
    elif risk_level == "🟡 MEDIUM RISK":
        if change_percent > 7:
            return "📈 BULLISH: Good upside potential with managed risk"
        elif change_percent > 3:
            return "📈 CONSIDER BUY: Positive momentum"
        elif change_percent < -3:
            return "📉 CONSIDER SELL: Downside risk present"
        else:
            return "⚖️ HOLD: Wait for clearer market direction"
    elif risk_level == "🟠 MODERATE RISK":
        if change_percent > 2:
            return "✅ BUY: Positive outlook with acceptable risk"
        elif change_percent < -2:
            return "💼 SELL: Consider reducing exposure"
        else:
            return "🔄 HOLD: Stable with minimal expected movement"
    else:  # LOW RISK
        if change_percent > 1:
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
@rate_limit(0.5)
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
        predictions, confidence_intervals, pred_error = predictor.predict_next_day_prices(historical_data)
        if pred_error:
            return jsonify({"error": pred_error}), 400
        
        # Calculate metrics
        predicted_close = predictions['Close']
        change_percent = ((predicted_close - current_price) / current_price) * 100
        
        # Calculate volatility from recent data
        volatility = historical_data['Close'].pct_change().std()
        
        # Generate risk assessment
        risk_level = get_risk_level(change_percent, volatility)
        recommendation = get_trading_recommendation(change_percent, risk_level, volatility)
        
        # Prepare response
        response = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "prediction_date": get_next_trading_day(),
            "last_trading_day": get_last_market_date(),
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
            "confidence_intervals": confidence_intervals,
            "model_performance": training_results.get(predictor.best_model_name, {}),
            "best_model": predictor.best_model_name,
            "market_status": get_market_status()[1],
            "data_analysis": {
                "total_data_points": len(historical_data),
                "features_used": len(predictor.feature_columns) if hasattr(predictor, 'feature_columns') else 0
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
        html.H1('🚀 Live AI Stock Prediction Platform', 
                style={'color': '#00e6ff', 'textAlign': 'center', 'marginBottom': '10px',
                      'fontFamily': 'Inter, sans-serif', 'fontWeight': '700', 'fontSize': '2.5rem'}),
        html.P("Real-Time Data • Live Market Prices • Next Trading Day Predictions • AI Risk Assessment", 
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

        html.Button("🚀 Generate Live AI Stock Prediction", 
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
            html.P("Enter a stock ticker symbol and click 'Generate Live AI Stock Prediction' to start real-time analysis.",
                  style={'color': '#94a3b8', 'fontSize': '16px', 'fontFamily': 'Inter, sans-serif'})
        ]), html.Div()
    
    if not ticker:
        return html.Div([
            html.P("❌ Please enter a valid stock ticker symbol.", 
                  style={'color': '#ff4d7c', 'fontSize': '16px', 'fontFamily': 'Inter, sans-serif'})
        ]), html.Div()
    
    try:
        print(f"🔄 Starting LIVE prediction process for {ticker}...")
        
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
        predictions, confidence_intervals, pred_error = predictor.predict_next_day_prices(historical_data)
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
        recommendation = get_trading_recommendation(change_percent, risk_level, volatility)
        
        # Create results display
        change_color = '#00ff9d' if change_percent > 0 else '#ff4d7c'
        trend_icon = '📈' if change_percent > 0 else '📉'
        
        results_content = html.Div([
            html.Div([
                html.H3(f"🔮 LIVE AI PREDICTION - {ticker.upper()}", 
                       style={'color': '#00e6ff', 'marginBottom': '25px', 'fontSize': '28px',
                             'fontFamily': 'Inter, sans-serif', 'fontWeight': '700', 'textAlign': 'center'}),
                
                html.Div([
                    html.Div([
                        html.P("CURRENT PRICE", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.H2(f"${current_price:.2f}", style={'color': '#ffffff', 'margin': '10px 0', 'fontSize': '36px', 'fontWeight': '700'}),
                    ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#1e293b', 'borderRadius': '12px', 'margin': '0 10px', 'flex': '1'}),
                    
                    html.Div([
                        html.P("PREDICTED CLOSE", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.H2(f"${predicted_close:.2f}", style={'color': change_color, 'margin': '10px 0', 'fontSize': '36px', 'fontWeight': '700'}),
                        html.P(f"{trend_icon} {change_percent:+.2f}%", style={'color': change_color, 'margin': '0', 'fontSize': '16px', 'fontWeight': '600'})
                    ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#1e293b', 'borderRadius': '12px', 'margin': '0 10px', 'flex': '1'}),
                    
                    html.Div([
                        html.P("RISK LEVEL", style={'color': '#94a3b8', 'margin': '0', 'fontSize': '14px', 'fontWeight': '600'}),
                        html.H3(risk_level, style={'color': '#ffa500', 'margin': '10px 0', 'fontSize': '20px', 'fontWeight': '700'}),
                        html.P(f"Volatility: {volatility:.4f}", style={'color': '#64748b', 'margin': '0', 'fontSize': '12px'})
                    ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#1e293b', 'borderRadius': '12px', 'margin': '0 10px', 'flex': '1'})
                ], style={'display': 'flex', 'marginBottom': '30px', 'gap': '15px'}),
                
                html.Div([
                    html.H4("💡 Trading Recommendation", style={'color': '#00e6ff', 'marginBottom': '15px', 'fontSize': '20px', 'fontWeight': '600'}),
                    html.P(recommendation, style={'color': '#ffffff', 'fontSize': '16px', 'padding': '15px', 'backgroundColor': '#1a1a2e', 'borderRadius': '8px'})
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.H4("🤖 Model Performance", style={'color': '#00e6ff', 'marginBottom': '15px', 'fontSize': '20px', 'fontWeight': '600'}),
                    html.P(f"Best Model: {predictor.best_model_name} | R² Score: {training_results[predictor.best_model_name]['R2']:.4f}", 
                          style={'color': '#94a3b8', 'fontSize': '14px'})
                ])
                
            ], style={
                'backgroundColor': '#111827', 
                'padding': '30px', 
                'borderRadius': '16px',
                'border': '1px solid #2a2a4a'
            })
        ])
        
        status = html.Div([
            html.H4(f"✅ LIVE AI Analysis Complete for {ticker.upper()}", 
                   style={'color': '#00ff9d', 'marginBottom': '10px', 'fontSize': '24px', 'fontWeight': '700'}),
            html.P(f"📊 Live Data Points: {len(historical_data):,} | Features Used: {len(predictor.feature_columns)}",
                  style={'color': '#94a3b8', 'fontSize': '14px'})
        ])
        
        return status, results_content
        
    except Exception as e:
        print(f"❌ LIVE Prediction failed: {str(e)}")
        return html.Div([
            html.P(f"❌ LIVE Prediction failed: {str(e)}", 
                  style={'color': '#ff4d7c', 'fontSize': '16px', 'fontFamily': 'Inter, sans-serif'})
        ]), html.Div()

# Main entry
if __name__ == '__main__':
    print("🚀 Starting Live AI Stock Prediction Dashboard...")
    print("📊 Access at: http://localhost:8080/")
    print("🤖 Prediction Page: http://localhost:8080/prediction")
    print("🔮 Features: Real-Time Data • Live Market Prices • Next Trading Day Predictions")
    print("💡 Using yfinance for live market data")
    app.run(debug=True, port=8080, host='0.0.0.0')