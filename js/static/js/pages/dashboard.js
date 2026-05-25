// dashboard.js
import { APIService } from '../api.js';
import { showToast } from '../components/toast.js';

export async function renderDashboard(mountNode) {
  // Initial loading state
  mountNode.innerHTML = `
    <div class="bento-grid stagger-1">
      <div class="bento-card" style="grid-column: span 3; min-height: 120px;">
        <div class="shimmer-loading" style="height: 100%; border-radius: var(--radius-sm);"></div>
      </div>
      <div class="bento-card" style="min-height: 200px;">
        <div class="shimmer-loading" style="height: 100%; border-radius: var(--radius-sm);"></div>
      </div>
      <div class="bento-card" style="min-height: 200px; grid-column: span 2;">
        <div class="shimmer-loading" style="height: 100%; border-radius: var(--radius-sm);"></div>
      </div>
    </div>
  `;

  try {
    const [health, stocks] = await Promise.all([
      APIService.getHealth(),
      APIService.getStocks()
    ]);

    const topStock = stocks[0] || { symbol: 'AAPL', change: 1.2 };
    const marketStatus = health.status === 'healthy' ? '<span class="text-green">●</span> Market Open' : '<span class="text-down">●</span> Market Closed';

    // Render Actual Dashboard
    mountNode.innerHTML = `
      <!-- Hero Stats -->
      <div class="bento-grid" style="margin-bottom: var(--space-4);">
        <!-- Portfolio Value -->
        <div class="bento-card animate-fade-in stagger-1">
          <p class="text-muted">Total Portfolio Value</p>
          <h2 style="font-size: 32px; margin: 8px 0;">$124,592.00</h2>
          <p class="text-up"><i class="fas fa-arrow-up"></i> 2.4% ($2,941) Today</p>
        </div>
        
        <!-- Market Status -->
        <div class="bento-card animate-fade-in stagger-2">
          <p class="text-muted">Market Status</p>
          <h2 style="font-size: 24px; margin: 8px 0;">${marketStatus}</h2>
          <p class="text-muted">Algorithms: ${health.algorithms ? health.algorithms.length : 7} Active</p>
        </div>

        <!-- Top Mover -->
        <div class="bento-card animate-fade-in stagger-3">
          <p class="text-muted">Top AI Pick</p>
          <div class="flex-between" style="margin-top: 8px;">
            <h2 class="text-cyan">${topStock.symbol}</h2>
            <span class="${topStock.change >= 0 ? 'text-up' : 'text-down'} font-mono" style="font-size: 20px;">
              ${topStock.change >= 0 ? '+' : ''}${topStock.change}%
            </span>
          </div>
          <p class="text-muted" style="margin-top: 8px;">High confidence signal detected</p>
        </div>
      </div>

      <!-- Main Grid -->
      <div class="bento-grid">
        <!-- Chart Section -->
        <div class="bento-card animate-fade-in stagger-4" style="grid-column: span 2; min-height: 400px;">
          <div class="flex-between" style="margin-bottom: var(--space-3);">
            <h3>Market Overview</h3>
            <div style="display: flex; gap: 8px;">
              <button class="btn btn-secondary" style="padding: 4px 12px; font-size: 12px;">1D</button>
              <button class="btn btn-primary" style="padding: 4px 12px; font-size: 12px;">1W</button>
              <button class="btn btn-secondary" style="padding: 4px 12px; font-size: 12px;">1M</button>
            </div>
          </div>
          <div id="main-chart" style="height: 300px; width: 100%;"></div>
        </div>

        <!-- AI Insights -->
        <div class="bento-card animate-fade-in stagger-5">
          <div class="flex-between" style="margin-bottom: var(--space-3);">
            <h3 class="text-gradient">AI Brain</h3>
            <i class="fas fa-brain text-purple animate-pulse" style="font-size: 24px;"></i>
          </div>
          
          <div style="display: flex; flex-direction: column; gap: var(--space-3);">
            <div style="padding: 12px; background: rgba(255,255,255,0.03); border-radius: 8px; border-left: 3px solid var(--accent-green);">
              <p style="font-size: 14px; font-weight: 600;">Bullish Sentiment Detected</p>
              <p class="text-muted" style="font-size: 12px; margin-top: 4px;">Tech sector shows 85% probability of continued rally over next 48h.</p>
            </div>
            
            <div style="padding: 12px; background: rgba(255,255,255,0.03); border-radius: 8px; border-left: 3px solid var(--warning-color);">
              <p style="font-size: 14px; font-weight: 600;">Volatility Warning</p>
              <p class="text-muted" style="font-size: 12px; margin-top: 4px;">Energy stocks experiencing unusually high pre-market volume.</p>
            </div>

            <button class="btn btn-primary" style="width: 100%; justify-content: center; margin-top: 8px;" onclick="window.location.href='/prediction'">
              View Full Predictions <i class="fas fa-arrow-right"></i>
            </button>
          </div>
        </div>
      </div>
      
      <!-- Watchlist -->
      <h3 style="margin: var(--space-5) 0 var(--space-3) 0;">Live Watchlist</h3>
      <div class="bento-grid" id="watchlist-grid">
        ${stocks.map((stock, index) => `
          <div class="bento-card animate-fade-in" style="animation-delay: ${0.1 * index}s; display: flex; align-items: center; justify-content: space-between;">
            <div>
              <h4 style="font-size: 18px;">${stock.symbol}</h4>
              <p class="text-muted" style="font-size: 12px;">${stock.name || 'Stock'}</p>
            </div>
            <div style="text-align: right;">
              <p class="font-mono" style="font-size: 16px;">$${stock.price.toFixed(2)}</p>
              <p class="${stock.change >= 0 ? 'text-up' : 'text-down'} font-mono" style="font-size: 14px;">
                ${stock.change >= 0 ? '<i class="fas fa-caret-up"></i>' : '<i class="fas fa-caret-down"></i>'} ${Math.abs(stock.change).toFixed(2)}%
              </p>
            </div>
          </div>
        `).join('')}
      </div>
    `;

    // Render Chart using TradingView Advanced Widget
    const chartContainer = document.getElementById('main-chart');
    if (chartContainer) {
      chartContainer.innerHTML = '';
      
      const script = document.createElement('script');
      script.type = 'text/javascript';
      script.src = 'https://s3.tradingview.com/tv.js';
      script.onload = () => {
        new TradingView.widget({
          "autosize": true,
          "symbol": topStock.symbol,
          "interval": "D",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "enable_publishing": false,
          "backgroundColor": "transparent",
          "gridColor": "rgba(255, 255, 255, 0.05)",
          "hide_top_toolbar": false,
          "hide_legend": true,
          "save_image": false,
          "container_id": "main-chart"
        });
      };
      document.body.appendChild(script);
    }

  } catch (err) {
    showToast('Failed to load dashboard data', 'error');
    console.error(err);
  }
}
