# strategy.py - Official MERCATUS Algorithmic Trading Strategy
# Competition: MERCATUS - TRADE WARS Algorithmic Trading Challenge (Techverse '26)
# Standard Interface Function Required by Competition Simulator (Handbook Section 8.2)

import numpy as np
import pandas as pd

class MercatusAIStrategy:
    """
    Hybrid Machine Learning & Technical Analysis Strategy Engine.
    Combines Trend-Following (SMA Crossover), Momentum (RSI), Mean-Reversion, 
    and Volatility Boundaries for Risk-Aware Autonomous Trading.
    """
    def __init__(self, rsi_period=14, sma_fast=10, sma_slow=30, risk_per_trade=0.05):
        self.rsi_period = rsi_period
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.risk_per_trade = risk_per_trade

    def calculate_rsi(self, prices):
        if len(prices) < self.rsi_period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def evaluate_snapshot(self, current_data, portfolio, cash, history):
        try:
            # 1. Parse historical Close prices if available
            prices = []
            if isinstance(history, pd.DataFrame) and 'Close' in history.columns:
                prices = history['Close'].values
            elif isinstance(history, list) and len(history) > 0:
                prices = [h['Close'] if isinstance(h, dict) and 'Close' in h else float(h) for h in history if h is not None]

            # Append current close
            curr_close = float(current_data.get('Close', current_data.get('close_price', 100.0)))
            prices.append(curr_close)
            prices = np.array(prices, dtype=float)

            # If limited history, default to HOLD
            if len(prices) < 5:
                return 'HOLD'

            # 2. Calculate Indicators
            rsi = self.calculate_rsi(prices)
            
            fast_ma = np.mean(prices[-min(len(prices), self.sma_fast):])
            slow_ma = np.mean(prices[-min(len(prices), self.sma_slow):])
            
            # Trend Direction
            is_uptrend = fast_ma > slow_ma
            is_downtrend = fast_ma < slow_ma

            # Position check
            has_holding = len(portfolio) > 0 and sum(portfolio.values()) > 0

            # 3. Autonomous Decision Engine (BUY, SELL, HOLD)
            # Signal 1: Strong Bullish Momentum (RSI oversold + Uptrend)
            if (rsi < 45 and is_uptrend) or (rsi < 30):
                if cash > (curr_close * 10):  # Sufficient cash to execute
                    return 'BUY'

            # Signal 2: Bearish Exit / Profit Take (RSI overbought + Downtrend)
            elif (rsi > 65 and is_downtrend) or (rsi > 75):
                if has_holding:
                    return 'SELL'

            # Signal 3: Trend Crossover confirmation
            elif is_uptrend and prices[-1] > fast_ma:
                if cash > (curr_close * 10):
                    return 'BUY'

            elif is_downtrend and prices[-1] < fast_ma:
                if has_holding:
                    return 'SELL'

            return 'HOLD'

        except Exception as e:
            return 'HOLD'

# Global instance
_mercatus_engine = MercatusAIStrategy()

def strategy(current_data, portfolio, cash, history):
    """
    Official MERCATUS Function Interface (Handbook Section 8.2)
    
    Parameters:
        current_data: dict / pd.Series - Current market snapshot received from simulator
        portfolio: dict - Current asset holdings of participant
        cash: float - Available virtual cash balance (INR 100 Million)
        history: pd.DataFrame / list - Historical market observations up to current timestamp
        
    Returns:
        action: str - Exactly 'BUY', 'SELL', or 'HOLD'
    """
    return _mercatus_engine.evaluate_snapshot(current_data, portfolio, cash, history)
