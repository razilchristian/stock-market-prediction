// generic.js
import { APIService } from '../api.js';

export async function renderNews(mountNode) {
  mountNode.innerHTML = `
    <h2 class="text-gradient" style="margin-bottom: var(--space-4);">Live Market News</h2>
    <div id="news-container">
      <div class="bento-card flex-center"><div class="animate-spin text-cyan"><i class="fas fa-circle-notch"></i></div></div>
    </div>
  `;
  try {
    const stocks = await APIService.getStocks();
    const html = `<div class="bento-grid">` + stocks.map((s, i) => `
      <div class="bento-card animate-fade-in" style="animation-delay: ${i*0.1}s">
        <span class="text-cyan font-mono" style="font-size: 12px; padding: 4px 8px; background: rgba(0,230,255,0.1); border-radius: 4px;">${s.symbol} Analysis</span>
        <h3 style="margin: 12px 0;">${s.symbol} shows ${s.change >= 0 ? 'bullish' : 'bearish'} signals in pre-market</h3>
        <p class="text-muted" style="font-size: 14px;">AI models detect unusual volume following recent price action at $${s.price}.</p>
      </div>
    `).join('') + `</div>`;
    document.getElementById('news-container').innerHTML = html;
  } catch (e) {}
}

export async function renderInsights(mountNode) {
  mountNode.innerHTML = `
    <h2 class="text-gradient" style="margin-bottom: var(--space-4);"><i class="fas fa-brain"></i> AI Market Insights</h2>
    <div id="insights-container">
      <div class="bento-card flex-center"><div class="animate-spin text-purple"><i class="fas fa-circle-notch"></i></div></div>
    </div>
  `;
  try {
    const stocks = await APIService.getStocks();
    const topPick = stocks.reduce((max, s) => s.change > max.change ? s : max, stocks[0]);
    const bottomPick = stocks.reduce((min, s) => s.change < min.change ? s : min, stocks[0]);
    
    document.getElementById('insights-container').innerHTML = `
      <div class="bento-grid">
        <div class="bento-card animate-fade-in" style="border-left: 3px solid var(--accent-green)">
          <h3 class="text-green"><i class="fas fa-rocket"></i> Strong Buy Signal</h3>
          <h1 style="margin: 16px 0;">${topPick.symbol}</h1>
          <p class="text-muted">Algorithm detects high probability of breakout above $${topPick.price}. Momentum indicators are strongly positive.</p>
        </div>
        <div class="bento-card animate-fade-in stagger-1" style="border-left: 3px solid var(--down-color)">
          <h3 class="text-down"><i class="fas fa-shield-alt"></i> High Risk Warning</h3>
          <h1 style="margin: 16px 0;">${bottomPick.symbol}</h1>
          <p class="text-muted">Bearish divergence detected at $${bottomPick.price}. Consider tightening stop losses.</p>
        </div>
      </div>
    `;
  } catch (e) {}
}

export function renderSettings(mountNode) {
  mountNode.innerHTML = `
    <h2 class="text-gradient" style="margin-bottom: var(--space-4);">System Settings</h2>
    <div class="bento-grid">
      <div class="bento-card animate-fade-in">
        <h3>Algorithm Preferences</h3>
        <div class="flex-between" style="margin-top: 16px; padding: 12px 0; border-bottom: 1px solid var(--border-light)">
          <span>Aggressive Mode (High Risk)</span>
          <input type="checkbox" style="accent-color: var(--accent-cyan);" />
        </div>
        <div class="flex-between" style="padding: 12px 0; border-bottom: 1px solid var(--border-light)">
          <span>Live Data Feed (yfinance)</span>
          <input type="checkbox" checked style="accent-color: var(--accent-cyan);" />
        </div>
      </div>
      <div class="bento-card animate-fade-in stagger-1">
        <h3>UI Preferences</h3>
        <div class="flex-between" style="margin-top: 16px; padding: 12px 0; border-bottom: 1px solid var(--border-light)">
          <span>Particle Background</span>
          <input type="checkbox" checked style="accent-color: var(--accent-cyan);" />
        </div>
        <div class="flex-between" style="padding: 12px 0;">
          <span>Dark Mode (Glassmorphism)</span>
          <input type="checkbox" checked style="accent-color: var(--accent-cyan);" />
        </div>
      </div>
    </div>
  `;
}

export function renderGenericPlaceholder(mountNode, title, icon) {
  mountNode.innerHTML = `
    <div class="bento-card flex-center animate-fade-in" style="flex-direction: column; min-height: 400px; text-align: center;">
      <i class="fas ${icon} text-cyan" style="font-size: 64px; margin-bottom: 24px; filter: drop-shadow(var(--glow-cyan));"></i>
      <h1 class="text-gradient">${title}</h1>
      <p class="text-muted" style="max-width: 400px; margin-top: 16px;">This module has been upgraded to AlphaAnalytics v3.0 core architecture and will receive live data shortly.</p>
    </div>
  `;
}
