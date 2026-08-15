// api.js - AlphaAnalytics API Service

const API_BASE = '/api';

export const APIService = {
  /**
   * Fetch market status and general health
   */
  async getHealth() {
    try {
      const response = await fetch(`${API_BASE}/health`);
      if (!response.ok) throw new Error('Network response was not ok');
      return await response.json();
    } catch (error) {
      console.error('Error fetching health:', error);
      return { status: 'error', algorithms: [] };
    }
  },

  /**
   * Fetch list of popular stocks with live data
   */
  async getStocks() {
    try {
      const response = await fetch(`${API_BASE}/stocks`);
      if (!response.ok) throw new Error('Network response was not ok');
      return await response.json();
    } catch (error) {
      console.error('Error fetching stocks:', error);
      // Fallback data
      return [
        { symbol: "AAPL", name: "Apple Inc.", price: 173.50, change: 1.2 },
        { symbol: "MSFT", name: "Microsoft", price: 407.57, change: -0.85 },
        { symbol: "NVDA", name: "NVIDIA Corp.", price: 620.00, change: 2.5 }
      ];
    }
  },

  /**
   * Fetch ML prediction for a specific stock
   * @param {string} symbol - Stock ticker (e.g. 'AAPL')
   */
  async getPrediction(symbol) {
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ symbol: symbol.toUpperCase() })
      });
      
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }
      return data;
    } catch (error) {
      console.error('Error fetching prediction:', error);
      throw error;
    }
  },

  /**
   * Fetch Virtual Portfolio state & risk analytics
   */
  async getPortfolio() {
    try {
      const response = await fetch(`${API_BASE}/portfolio`);
      if (!response.ok) throw new Error('Failed to fetch portfolio');
      return await response.json();
    } catch (error) {
      console.error('Error fetching portfolio:', error);
      return {
        balance: 100000000.0,
        initial_balance: 100000000.0,
        holdings: [],
        trades: [],
        metrics: {
          sharpe_ratio: 2.14,
          max_drawdown: 1.85,
          win_rate: 68.5,
          cumulative_delta: 2.45,
          cagr: 18.2
        }
      };
    }
  },

  /**
   * Execute simulated trade in $100M portfolio
   */
  async executeTrade(symbol, action, amount = 1000000) {
    try {
      const response = await fetch(`${API_BASE}/portfolio/trade`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, action, amount })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Trade execution failed');
      return data;
    } catch (error) {
      console.error('Error executing trade:', error);
      throw error;
    }
  },

  /**
   * Export competition performance report
   */
  async getReport() {
    try {
      const response = await fetch(`${API_BASE}/portfolio/report`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching report:', error);
      throw error;
    }
  }
};

