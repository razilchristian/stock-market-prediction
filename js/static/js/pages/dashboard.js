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
            <div style="display: flex; align-items: center; gap: 16px;">
              <h3>Market Overview</h3>
              <div class="search-box" style="width: 200px; padding: 4px 8px; min-height: 30px;">
                <i class="fas fa-search text-cyan" style="font-size: 12px;"></i>
                <input type="text" id="chart-symbol-input" placeholder="Search symbol..." autocomplete="off" style="font-size: 12px; height: 20px;" />
                <button id="chart-search-btn" class="btn btn-primary" style="padding: 2px 8px; font-size: 10px; height: 20px; min-height: 20px;">GO</button>
              </div>
            </div>
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

    let currentChartInstance = null;

    // Render interactive Candlestick chart via ApexCharts (Instant, reliable, dark neon cyber styled)
    const renderChart = async (symbol, range = '1W') => {
      const chartContainer = document.getElementById('main-chart');
      if (!chartContainer) return;

      chartContainer.innerHTML = `
        <div class="flex-center" style="height: 320px; flex-direction: column; gap: 12px;">
          <div class="animate-spin text-cyan" style="font-size: 28px;"><i class="fas fa-circle-notch"></i></div>
          <p class="text-muted" style="font-size: 13px;">Loading market data for ${symbol}...</p>
        </div>
      `;

      try {
        const chartData = await APIService.getChart(symbol);
        let seriesData = chartData.candles || [];

        // Apply Timeframe slicing
        if (range === '1D') {
          seriesData = seriesData.slice(-5);
        } else if (range === '1W') {
          seriesData = seriesData.slice(-25);
        } else if (range === '1M') {
          seriesData = seriesData.slice(-60);
        }

        chartContainer.innerHTML = '';

        if (!seriesData || seriesData.length === 0) {
          chartContainer.innerHTML = `
            <div class="flex-center" style="height: 320px; flex-direction: column; gap: 8px;">
              <i class="fas fa-chart-line text-muted" style="font-size: 36px;"></i>
              <p class="text-muted">No price candle history found for ${symbol}</p>
            </div>
          `;
          return;
        }

        if (currentChartInstance) {
          try { currentChartInstance.destroy(); } catch (e) {}
        }

        const options = {
          series: [{
            name: `${symbol} Price`,
            data: seriesData
          }],
          chart: {
            type: 'candlestick',
            height: 320,
            background: 'transparent',
            toolbar: {
              show: true,
              tools: {
                download: false,
                selection: true,
                zoom: true,
                zoomin: true,
                zoomout: true,
                pan: true,
                reset: true
              }
            },
            animations: {
              enabled: true,
              easing: 'easeinout',
              speed: 400
            }
          },
          theme: { mode: 'dark' },
          plotOptions: {
            candlestick: {
              colors: {
                upward: '#00FF9D',
                downward: '#FF3366'
              },
              wick: {
                useFillColor: true
              }
            }
          },
          grid: {
            borderColor: 'rgba(255, 255, 255, 0.05)',
            strokeDashArray: 4
          },
          xaxis: {
            type: 'datetime',
            labels: {
              style: { colors: '#94a3b8', fontSize: '11px', fontFamily: 'JetBrains Mono' },
              datetimeUTC: false
            },
            axisBorder: { color: 'rgba(255, 255, 255, 0.1)' }
          },
          yaxis: {
            tooltip: { enabled: true },
            labels: {
              style: { colors: '#94a3b8', fontSize: '11px', fontFamily: 'JetBrains Mono' },
              formatter: (val) => '$' + val.toFixed(2)
            }
          },
          tooltip: {
            theme: 'dark',
            x: { format: 'dd MMM yyyy' },
            style: { fontSize: '12px', fontFamily: 'JetBrains Mono' }
          }
        };

        currentChartInstance = new ApexCharts(chartContainer, options);
        await currentChartInstance.render();

      } catch (err) {
        console.error('Failed to render chart:', err);
        chartContainer.innerHTML = `
          <div class="flex-center" style="height: 320px; flex-direction: column; gap: 8px;">
            <i class="fas fa-exclamation-triangle text-down" style="font-size: 32px;"></i>
            <p class="text-down">Failed to load chart for ${symbol}</p>
          </div>
        `;
      }
    };

    let activeSymbol = topStock.symbol || 'AAPL';
    let activeRange = '1W';
    renderChart(activeSymbol, activeRange);

    // Timeframe Buttons (1D, 1W, 1M)
    const tfButtons = document.querySelectorAll('.bento-card [data-tf]');
    const headerBtnBox = document.querySelector('.bento-card .flex-between div:last-child');
    if (headerBtnBox) {
      headerBtnBox.innerHTML = `
        <button class="btn btn-secondary tf-btn" data-tf="1D" style="padding: 4px 12px; font-size: 12px;">1D</button>
        <button class="btn btn-primary tf-btn" data-tf="1W" style="padding: 4px 12px; font-size: 12px;">1W</button>
        <button class="btn btn-secondary tf-btn" data-tf="1M" style="padding: 4px 12px; font-size: 12px;">1M</button>
      `;
      headerBtnBox.querySelectorAll('.tf-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          headerBtnBox.querySelectorAll('.tf-btn').forEach(b => {
            b.classList.remove('btn-primary');
            b.classList.add('btn-secondary');
          });
          btn.classList.remove('btn-secondary');
          btn.classList.add('btn-primary');
          activeRange = btn.getAttribute('data-tf');
          renderChart(activeSymbol, activeRange);
        });
      });
    }

    const chartBtn = document.getElementById('chart-search-btn');
    const chartInput = document.getElementById('chart-symbol-input');
    const globalSearch = document.getElementById('global-search');

    const updateChartSymbol = (sym) => {
      if (!sym) return;
      const cleanSym = sym.trim().toUpperCase();
      activeSymbol = cleanSym;
      renderChart(cleanSym, activeRange);
      if (chartInput) chartInput.value = cleanSym;
      showToast('Loaded live chart for ' + cleanSym, 'success');
    };

    if (chartBtn && chartInput) {
      chartBtn.addEventListener('click', () => updateChartSymbol(chartInput.value));
      chartInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') updateChartSymbol(chartInput.value);
      });
    }

    if (globalSearch) {
      globalSearch.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') updateChartSymbol(globalSearch.value);
      });
    }

    // Also clicking any watchlist item updates the chart
    const watchlistCards = document.querySelectorAll('#watchlist-grid .bento-card');
    watchlistCards.forEach(card => {
      card.style.cursor = 'pointer';
      card.addEventListener('click', () => {
        const symbolElem = card.querySelector('h4');
        if (symbolElem) updateChartSymbol(symbolElem.textContent);
      });
    });



  } catch (err) {
    showToast('Failed to load dashboard data', 'error');
    console.error(err);
  }
}
