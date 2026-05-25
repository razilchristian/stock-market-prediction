// watchlist.js
import { APIService } from '../api.js';
import { showToast } from '../components/toast.js';

export async function renderWatchlist(mountNode) {
  mountNode.innerHTML = `
    <div class="flex-between" style="margin-bottom: var(--space-4);">
      <h2 class="text-gradient" style="font-size: 32px;"><i class="fas fa-star"></i> My Watchlist</h2>
      <div class="search-box">
        <i class="fas fa-search text-muted"></i>
        <input type="text" placeholder="Add symbol..." />
      </div>
    </div>
    
    <div id="watchlist-container">
      <div class="bento-card flex-center"><div class="animate-spin text-cyan"><i class="fas fa-circle-notch"></i></div></div>
    </div>
  `;

  try {
    const stocks = await APIService.getStocks();
    
    const html = `
      <div class="bento-grid">
        ${stocks.map((stock, i) => {
          const isUp = stock.change >= 0;
          return `
            <div class="bento-card animate-fade-in" style="animation-delay: ${i * 0.1}s">
              <div class="flex-between" style="margin-bottom: 16px;">
                <div>
                  <h3>${stock.symbol}</h3>
                  <p class="text-muted" style="font-size: 12px;">${stock.name}</p>
                </div>
                <button class="icon-btn text-cyan" style="border-color: var(--accent-cyan)"><i class="fas fa-star"></i></button>
              </div>
              <h2 style="font-size: 28px;">$${stock.price.toFixed(2)}</h2>
              <p class="${isUp ? 'text-up' : 'text-down'}" style="margin-top: 8px;">
                <i class="fas fa-arrow-${isUp ? 'up' : 'down'}"></i> ${Math.abs(stock.change).toFixed(2)}%
              </p>
              
              <!-- TradingView Mini Widget -->
              <div class="tradingview-widget-container" style="height: 100px; margin-top: 16px; margin-left: -16px; margin-right: -16px; margin-bottom: -16px;">
                <div id="tv-mini-${stock.symbol}" style="height: 100%;"></div>
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;
    
    document.getElementById('watchlist-container').innerHTML = html;

    // Inject TradingView Mini Charts
    const script = document.createElement('script');
    script.type = 'text/javascript';
    script.src = 'https://s3.tradingview.com/tv.js';
    script.onload = () => {
      stocks.forEach(stock => {
        new TradingView.widget({
          "autosize": true,
          "symbol": stock.symbol,
          "interval": "D",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "3", // Area chart
          "locale": "en",
          "enable_publishing": false,
          "backgroundColor": "transparent",
          "gridColor": "transparent",
          "hide_top_toolbar": true,
          "hide_legend": true,
          "save_image": false,
          "container_id": `tv-mini-${stock.symbol}`,
          "hide_volume": true
        });
      });
    };
    document.body.appendChild(script);

  } catch (err) {
    showToast('Failed to load watchlist', 'error');
  }
}
