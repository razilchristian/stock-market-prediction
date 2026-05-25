// portfolio.js
import { APIService } from '../api.js';
import { showToast } from '../components/toast.js';

export async function renderPortfolio(mountNode) {
  mountNode.innerHTML = `
    <div class="flex-between" style="margin-bottom: var(--space-4);">
      <h2 class="text-gradient" style="font-size: 32px;">Your Portfolio</h2>
      <button class="btn btn-primary"><i class="fas fa-plus"></i> Add Asset</button>
    </div>
    
    <div class="bento-grid" style="margin-bottom: var(--space-4);">
      <div class="bento-card animate-fade-in stagger-1 flex-center" style="flex-direction: column;">
        <p class="text-muted">Total Balance</p>
        <h1 style="font-size: 40px; margin: 8px 0;">$124,592.00</h1>
        <p class="text-up"><i class="fas fa-arrow-up"></i> 2.4% Today</p>
      </div>
      
      <div class="bento-card animate-fade-in stagger-2" style="grid-column: span 2;">
        <h3>Asset Allocation</h3>
        <div id="allocation-chart" style="height: 150px; margin-top: 16px;"></div>
      </div>
    </div>
    
    <div class="bento-card animate-fade-in stagger-3">
      <h3 style="margin-bottom: 16px;">Holdings</h3>
      <div id="holdings-list">
        <div class="shimmer-loading" style="height: 200px; border-radius: var(--radius-sm);"></div>
      </div>
    </div>
  `;

  try {
    const stocks = await APIService.getStocks();
    
    const holdingsHTML = stocks.slice(0, 5).map(stock => {
      const shares = Math.floor(Math.random() * 50) + 10;
      const total = (stock.price * shares).toFixed(2);
      const isUp = stock.change >= 0;
      return `
        <div class="flex-between" style="padding: 16px 0; border-bottom: 1px solid var(--border-light);">
          <div style="display: flex; align-items: center; gap: 16px;">
            <div style="width: 40px; height: 40px; background: rgba(255,255,255,0.05); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold;">
              ${stock.symbol.charAt(0)}
            </div>
            <div>
              <h4 style="font-size: 16px;">${stock.symbol}</h4>
              <p class="text-muted" style="font-size: 12px;">${shares} Shares</p>
            </div>
          </div>
          <div style="text-align: right;">
            <h4 style="font-size: 16px;">$${total}</h4>
            <p class="${isUp ? 'text-up' : 'text-down'}" style="font-size: 12px;">
              ${isUp ? '+' : ''}${stock.change}%
            </p>
          </div>
        </div>
      `;
    }).join('');
    
    document.getElementById('holdings-list').innerHTML = holdingsHTML;

    // Render Allocation Chart
    if (window.ApexCharts) {
      const options = {
        series: [44, 55, 13, 33],
        labels: ['AAPL', 'MSFT', 'NVDA', 'Cash'],
        chart: { type: 'donut', height: 150, background: 'transparent' },
        theme: { mode: 'dark' },
        colors: ['#00E6FF', '#8A2BE2', '#00FF9D', '#64748B'],
        stroke: { show: false },
        dataLabels: { enabled: false },
        legend: { position: 'right' }
      };
      new ApexCharts(document.querySelector("#allocation-chart"), options).render();
    }
  } catch (err) {
    showToast('Failed to load portfolio', 'error');
  }
}
