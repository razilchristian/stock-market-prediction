// prediction.js
import { APIService } from '../api.js';
import { showToast } from '../components/toast.js';

export async function renderPrediction(mountNode) {
  mountNode.innerHTML = `
    <div class="flex-between" style="margin-bottom: var(--space-4);">
      <div>
        <h2 class="text-gradient" style="font-size: 32px;">AI Prediction Engine</h2>
        <p class="text-muted">Powered by 8 advanced machine learning models</p>
      </div>
      
      <div class="search-box" style="width: 300px;">
        <i class="fas fa-search text-cyan"></i>
        <input type="text" id="symbol-input" placeholder="Enter Stock Symbol (e.g., AAPL)..." autocomplete="off" />
        <button id="predict-btn" class="btn btn-primary" style="padding: 4px 16px; margin-left: 8px;">Analyze</button>
      </div>
    </div>

    <!-- Results Container -->
    <div id="prediction-results">
      <div class="flex-center" style="height: 300px; flex-direction: column; gap: var(--space-3); border: 1px dashed var(--border-light); border-radius: var(--radius-lg);">
        <i class="fas fa-robot text-muted" style="font-size: 48px;"></i>
        <p class="text-muted">Enter a symbol above to run the multi-model AI analysis.</p>
      </div>
    </div>
  `;

  const predictBtn = document.getElementById('predict-btn');
  const symbolInput = document.getElementById('symbol-input');
  
  const runPrediction = async () => {
    const symbol = symbolInput.value.trim().toUpperCase();
    if (!symbol) return showToast('Please enter a stock symbol', 'error');

    const resultsContainer = document.getElementById('prediction-results');
    
    // Loading State
    resultsContainer.innerHTML = `
      <div class="bento-card flex-center" style="height: 400px; flex-direction: column; gap: var(--space-4);">
        <div class="animate-spin text-cyan" style="font-size: 48px;">
          <i class="fas fa-circle-notch"></i>
        </div>
        <h3 class="animate-pulse">Running 8 Machine Learning Models...</h3>
        <p class="text-muted">Fetching live data, handling splits, and generating forecasts for ${symbol}. This may take a few seconds.</p>
        
        <div style="width: 300px; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden; margin-top: 16px;">
          <div class="shimmer-loading" style="width: 100%; height: 100%;"></div>
        </div>
      </div>
    `;

    try {
      const data = await APIService.getPrediction(symbol);
      renderResults(resultsContainer, data);
      showToast(`Successfully analyzed ${symbol}`, 'success');
    } catch (err) {
      showToast(err.message, 'error');
      resultsContainer.innerHTML = `
        <div class="bento-card" style="border-color: var(--down-color);">
          <h3 class="text-down"><i class="fas fa-exclamation-triangle"></i> Analysis Failed</h3>
          <p style="margin-top: 8px;">${err.message}</p>
          <button class="btn btn-secondary" style="margin-top: 16px;" onclick="document.getElementById('predict-btn').click()">Try Again</button>
        </div>
      `;
    }
  };

  predictBtn.addEventListener('click', runPrediction);
  symbolInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') runPrediction();
  });
}

function renderResults(container, data) {
  const { symbol, prediction, current_prices, trading_recommendation, insight, model_info, market_status } = data;
  
  const currentPrice = current_prices.close || 0;
  const predictedClose = prediction.predicted ? prediction.predicted.Close : currentPrice;
  const changePercent = ((predictedClose - currentPrice) / currentPrice) * 100;
  
  const isPositive = changePercent >= 0;
  const trendColor = isPositive ? 'var(--up-color)' : 'var(--down-color)';
  const trendIcon = isPositive ? 'fa-arrow-up' : 'fa-arrow-down';

  // Recommendation Color
  let recColor = 'var(--text-primary)';
  if (trading_recommendation.includes('BUY')) recColor = 'var(--accent-green)';
  if (trading_recommendation.includes('SELL')) recColor = 'var(--down-color)';
  if (trading_recommendation.includes('HOLD')) recColor = 'var(--warning-color)';

  container.innerHTML = `
    <!-- Top Stats -->
    <div class="bento-grid" style="margin-bottom: var(--space-4);">
      <!-- Main Forecast -->
      <div class="bento-card animate-fade-in stagger-1" style="border-top: 3px solid ${trendColor};">
        <div class="flex-between">
          <p class="text-muted">Target Price (Next Close)</p>
          <span style="padding: 4px 12px; background: rgba(255,255,255,0.05); border-radius: 12px; font-size: 12px;">${market_status}</span>
        </div>
        <h1 style="font-size: 48px; margin: 12px 0;">$${predictedClose.toFixed(2)}</h1>
        <div class="flex-between">
          <p style="color: ${trendColor}; font-weight: 600; font-size: 18px;">
            <i class="fas ${trendIcon}"></i> ${Math.abs(changePercent).toFixed(2)}%
          </p>
          <p class="text-muted">Current: $${currentPrice.toFixed(2)}</p>
        </div>
      </div>

      <!-- Action & Confidence -->
      <div class="bento-card animate-fade-in stagger-2 flex-center" style="flex-direction: column; text-align: center;">
        <p class="text-muted" style="margin-bottom: 12px;">AI Trading Recommendation</p>
        <h2 style="font-size: 28px; color: ${recColor}; margin-bottom: 24px;">${trading_recommendation}</h2>
        
        <div style="width: 100%;">
          <div class="flex-between" style="margin-bottom: 8px;">
            <span class="text-muted">Model Confidence</span>
            <span class="text-cyan font-mono">${prediction.overall_confidence}%</span>
          </div>
          <div style="width: 100%; height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden;">
            <div style="width: ${prediction.overall_confidence}%; height: 100%; background: var(--grad-cyan-purple); border-radius: 4px;"></div>
          </div>
        </div>
      </div>
      
      <!-- AI Insight -->
      <div class="bento-card animate-fade-in stagger-3">
        <h3 class="text-gradient" style="margin-bottom: 16px;"><i class="fas fa-brain"></i> AI Insight</h3>
        <p style="font-size: 16px; line-height: 1.6;">${insight}</p>
        
        <div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--border-light);">
          <div class="flex-between" style="margin-bottom: 8px;">
            <span class="text-muted">Models Employed</span>
            <span class="font-mono">${model_info.version} (${model_info.feature_count} features)</span>
          </div>
          <div class="flex-between">
            <span class="text-muted">Fallback Mode</span>
            <span class="${model_info.fallback_mode ? 'text-down' : 'text-green'} font-mono">${model_info.fallback_mode ? 'ACTIVE' : 'OFF'}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Range Prediction Grid -->
    <h3 style="margin: var(--space-5) 0 var(--space-3) 0;">Predicted Range</h3>
    <div class="bento-grid">
      ${['Open', 'High', 'Low', 'Close'].map((metric, i) => {
        const val = prediction.predicted ? prediction.predicted[metric] : currentPrice;
        const conf = prediction.confidence ? prediction.confidence[metric] : 50;
        return `
          <div class="bento-card animate-fade-in" style="animation-delay: ${0.3 + (i*0.1)}s">
            <p class="text-muted">${metric}</p>
            <h3 style="font-size: 24px; margin: 8px 0;">$${val.toFixed(2)}</h3>
            <div class="flex-between text-muted" style="font-size: 12px;">
              <span>Confidence</span>
              <span class="text-cyan">${conf}%</span>
            </div>
          </div>
        `;
      }).join('')}
    </div>

    <!-- Algorithm Breakdown -->
    <h3 style="margin: var(--space-5) 0 var(--space-3) 0;">Algorithm Breakdown (Close Price)</h3>
    <div class="bento-grid" style="grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));">
      ${Object.entries(model_info.detailed_predictions?.Close || {}).map(([algo, price], i) => {
        const diff = price - currentPrice;
        const diffColor = diff >= 0 ? 'var(--up-color)' : 'var(--down-color)';
        const diffIcon = diff >= 0 ? 'fa-caret-up' : 'fa-caret-down';
        const pPct = (Math.abs(diff) / currentPrice) * 100;
        return `
          <div class="bento-card animate-fade-in" style="animation-delay: ${0.5 + (i*0.05)}s; padding: var(--space-3);">
            <div class="flex-between">
              <span style="font-weight: 600; font-size: 14px;">${algo}</span>
              <i class="fas fa-microchip text-muted" style="font-size: 12px;"></i>
            </div>
            <h3 style="margin: 12px 0 4px 0; font-size: 20px;">$${price.toFixed(2)}</h3>
            <p style="color: ${diffColor}; font-size: 12px;">
              <i class="fas ${diffIcon}"></i> ${pPct.toFixed(2)}%
            </p>
          </div>
        `;
      }).join('') || '<p class="text-muted" style="grid-column: 1/-1;">Detailed algorithm predictions not available.</p>'}
    </div>
  `;
}
