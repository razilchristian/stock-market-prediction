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
  }
};
