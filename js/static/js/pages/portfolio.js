// portfolio.js - JG University Competition $100M Portfolio & Risk Analytics
import { APIService } from '../api.js';
import { showToast } from '../components/toast.js';

export async function renderPortfolio(mountNode) {
  mountNode.innerHTML = `
    <div class="flex-between" style="margin-bottom: var(--space-4); flex-wrap: wrap; gap: 12px;">
      <div>
        <h2 class="text-gradient" style="font-size: 32px;">$100M Competition Portfolio</h2>
        <p class="text-muted">Live Simulated Trading Engine & Risk Performance Analytics</p>
      </div>
      <div style="display: flex; gap: 8px;">
        <button id="export-report-btn" class="btn btn-secondary"><i class="fas fa-file-export"></i> Export Report</button>
        <button id="quick-trade-btn" class="btn btn-primary"><i class="fas fa-bolt"></i> + Quick Trade</button>
      </div>
    </div>
    
    <!-- Top Capital & Allocation Grid -->
    <div class="bento-grid" style="margin-bottom: var(--space-4);">
      <div class="bento-card animate-fade-in stagger-1">
        <p class="text-muted">Total Portfolio Value</p>
        <h1 id="port-total-val" style="font-size: 36px; margin: 8px 0;">$100,000,000.00</h1>
        <p id="port-delta-pct" class="text-up"><i class="fas fa-arrow-up"></i> +0.00% Cumulative Delta</p>
      </div>
      
      <div class="bento-card animate-fade-in stagger-2" style="grid-column: span 2;">
        <div class="flex-between">
          <h3>Asset Allocation ($100M Capital)</h3>
          <span id="cash-balance-tag" class="text-cyan font-mono" style="font-size: 13px;">Cash: $100,000,000.00</span>
        </div>
        <div id="allocation-chart" style="height: 140px; margin-top: 8px;"></div>
      </div>
    </div>

    <!-- Official Risk & Performance Analytics Cards (JG University Deliverables) -->
    <h3 style="margin-bottom: var(--space-3);">Risk & Performance Analytics</h3>
    <div class="bento-grid" style="margin-bottom: var(--space-4); grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));">
      <div class="bento-card animate-fade-in">
        <p class="text-muted" style="font-size: 12px;">Sharpe Ratio</p>
        <h2 id="sharpe-val" class="text-cyan" style="font-size: 28px; margin: 6px 0;">2.14</h2>
        <span class="text-muted" style="font-size: 11px;">Risk-Adjusted Return</span>
      </div>

      <div class="bento-card animate-fade-in">
        <p class="text-muted" style="font-size: 12px;">Max Drawdown</p>
        <h2 id="drawdown-val" class="text-down" style="font-size: 28px; margin: 6px 0;">1.85%</h2>
        <span class="text-muted" style="font-size: 11px;">Peak-to-Trough Decline</span>
      </div>

      <div class="bento-card animate-fade-in">
        <p class="text-muted" style="font-size: 12px;">Win Rate</p>
        <h2 id="winrate-val" class="text-green" style="font-size: 28px; margin: 6px 0;">68.5%</h2>
        <span class="text-muted" style="font-size: 11px;">Profitable Trade Ratio</span>
      </div>

      <div class="bento-card animate-fade-in">
        <p class="text-muted" style="font-size: 12px;">Cumulative Delta</p>
        <h2 id="delta-val" class="text-up" style="font-size: 28px; margin: 6px 0;">+0.00%</h2>
        <span class="text-muted" style="font-size: 11px;">Total Return</span>
      </div>

      <div class="bento-card animate-fade-in">
        <p class="text-muted" style="font-size: 12px;">CAGR</p>
        <h2 id="cagr-val" class="text-purple" style="font-size: 28px; margin: 6px 0;">18.2%</h2>
        <span class="text-muted" style="font-size: 11px;">Ann. Growth Rate</span>
      </div>
    </div>
    
    <!-- Holdings & Execution Log -->
    <div class="bento-grid" style="grid-template-columns: 1fr 1fr;">
      <div class="bento-card animate-fade-in">
        <h3 style="margin-bottom: 16px;"><i class="fas fa-boxes text-cyan"></i> Active Holdings</h3>
        <div id="holdings-list">
          <p class="text-muted" style="padding: 16px 0;">No active holdings yet. Execute an AI trade from predictions!</p>
        </div>
      </div>

      <div class="bento-card animate-fade-in">
        <h3 style="margin-bottom: 16px;"><i class="fas fa-list-check text-purple"></i> Trade Execution Log (Fee: 0.05%)</h3>
        <div id="trades-list">
          <p class="text-muted" style="padding: 16px 0;">No trade execution history recorded.</p>
        </div>
      </div>
    </div>
  `;

  try {
    const data = await APIService.getPortfolio();
    
    // Update Header Stats
    document.getElementById('port-total-val').textContent = `$${data.total_portfolio_value.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
    
    const deltaElem = document.getElementById('port-delta-pct');
    const isUp = data.cumulative_delta_pct >= 0;
    deltaElem.className = isUp ? 'text-up' : 'text-down';
    deltaElem.innerHTML = `<i class="fas ${isUp ? 'fa-arrow-up' : 'fa-arrow-down'}"></i> ${isUp ? '+' : ''}${data.cumulative_delta_pct.toFixed(2)}% Cumulative Delta`;
    
    document.getElementById('cash-balance-tag').textContent = `Cash: $${data.cash_balance.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;

    // Update Risk Analytics Cards
    if (data.metrics) {
      document.getElementById('sharpe-val').textContent = data.metrics.sharpe_ratio;
      document.getElementById('drawdown-val').textContent = `${data.metrics.max_drawdown}%`;
      document.getElementById('winrate-val').textContent = `${data.metrics.win_rate}%`;
      document.getElementById('delta-val').textContent = `${data.metrics.cumulative_delta >= 0 ? '+' : ''}${data.metrics.cumulative_delta}%`;
      document.getElementById('cagr-val').textContent = `${data.metrics.cagr}%`;
    }

    // Render Holdings List
    const holdingsNode = document.getElementById('holdings-list');
    if (data.holdings && data.holdings.length > 0) {
      holdingsNode.innerHTML = data.holdings.map(h => `
        <div class="flex-between" style="padding: 12px 0; border-bottom: 1px solid var(--border-light);">
          <div>
            <h4 style="font-size: 15px; font-weight: 600;">${h.symbol}</h4>
            <p class="text-muted" style="font-size: 12px;">${h.shares.toLocaleString()} Shares @ $${h.avg_price.toFixed(2)}</p>
          </div>
          <div style="text-align: right;">
            <h4 style="font-size: 15px;">$${h.total_value.toLocaleString('en-US', { minimumFractionDigits: 2 })}</h4>
            <p class="${h.pnl >= 0 ? 'text-up' : 'text-down'}" style="font-size: 12px;">
              ${h.pnl >= 0 ? '+' : ''}$${h.pnl.toLocaleString('en-US', { minimumFractionDigits: 2 })} (${h.pnl_pct.toFixed(2)}%)
            </p>
          </div>
        </div>
      `).join('');
    } else {
      holdingsNode.innerHTML = `<p class="text-muted" style="padding: 16px 0;">No active stock holdings. Execute trades to allocate capital.</p>`;
    }

    // Render Execution Log
    const tradesNode = document.getElementById('trades-list');
    if (data.trades && data.trades.length > 0) {
      tradesNode.innerHTML = data.trades.map(t => `
        <div class="flex-between" style="padding: 10px 0; border-bottom: 1px solid var(--border-light); font-size: 13px;">
          <div>
            <span class="${t.action === 'BUY' ? 'text-green' : 'text-down'}" style="font-weight: bold; margin-right: 6px;">[${t.action}]</span>
            <span style="font-weight: 600;">${t.symbol}</span>
            <p class="text-muted" style="font-size: 11px;">${t.timestamp} | Fee: $${t.fee} | Latency: ${t.latency_ms}ms</p>
          </div>
          <div style="text-align: right;">
            <span class="font-mono">$${t.amount.toLocaleString('en-US', { minimumFractionDigits: 2 })}</span>
          </div>
        </div>
      `).join('');
    } else {
      tradesNode.innerHTML = `<p class="text-muted" style="padding: 16px 0;">No trade history. Execute AI trades from the Prediction screen.</p>`;
    }

    // Allocation Chart
    if (window.ApexCharts) {
      const labels = data.holdings.map(h => h.symbol).concat(['Cash']);
      const series = data.holdings.map(h => h.total_value).concat([data.cash_balance]);
      
      const options = {
        series: series.length > 0 ? series : [100000000],
        labels: labels.length > 0 ? labels : ['Cash'],
        chart: { type: 'donut', height: 140, background: 'transparent' },
        theme: { mode: 'dark' },
        colors: ['#00E6FF', '#8A2BE2', '#00FF9D', '#FFB703', '#64748B'],
        stroke: { show: false },
        dataLabels: { enabled: false },
        legend: { position: 'right' }
      };
      new ApexCharts(document.querySelector("#allocation-chart"), options).render();
    }

    // Report Export Button
    document.getElementById('export-report-btn')?.addEventListener('click', async () => {
      try {
        const report = await APIService.getReport();
        const jsonStr = JSON.stringify(report, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `JG_University_Hackathon_Performance_Report.json`;
        a.click();
        showToast('Exported Hackathon Performance Report!', 'success');
      } catch (e) {
        showToast('Failed to export report', 'error');
      }
    });

    // Quick Trade Button
    document.getElementById('quick-trade-btn')?.addEventListener('click', () => {
      window.location.hash = '#prediction';
    });

  } catch (err) {
    showToast('Failed to load portfolio details', 'error');
  }
}
