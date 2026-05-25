// main.js - AlphaAnalytics v3.0 Entry Point

import { renderSidebar } from './components/sidebar.js';
import { renderNavbar } from './components/navbar.js';
import { APIService } from './api.js';

import { renderDashboard } from './pages/dashboard.js';
import { renderPrediction } from './pages/prediction.js';

// 1. Initialize Layout Elements
function initLayout(title) {
  renderSidebar();
  renderNavbar(title);
}

// 2. Animated Background Canvas
function initCanvas() {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  
  const particles = [];
  const particleCount = window.innerWidth < 768 ? 30 : 70;
  
  for (let i = 0; i < particleCount; i++) {
    particles.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      radius: Math.random() * 2,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5
    });
  }
  
  function animate() {
    requestAnimationFrame(animate);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    ctx.fillStyle = 'rgba(0, 230, 255, 0.5)';
    
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      
      if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
      
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fill();
    });
  }
  
  animate();
  
  window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  });
}

// 3. Simple Router
async function route() {
  const path = window.location.pathname;
  const contentMount = document.getElementById('page-content');
  
  if (!contentMount) return;
  
  try {
    if (path === '/' || path === '/jeet' || path === '/jeet.html') {
      initLayout('Dashboard');
      await renderDashboard(contentMount);
    } 
    else if (path.includes('prediction')) {
      initLayout('AI Predictions');
      await renderPrediction(contentMount);
    }
    else {
      initLayout('AlphaAnalytics');
      contentMount.innerHTML = `<div class="bento-card"><h2 class="text-gradient">Under Construction</h2><p class="text-muted">The ${path} page is being built for v3.0.</p></div>`;
    }
  } catch (err) {
    console.error("Routing error:", err);
    contentMount.innerHTML = `<div class="bento-card" style="border-color: var(--down-color)"><h2 class="text-down">Error Loading Page</h2><p>${err.message}</p></div>`;
  }
}

// App Initialization
document.addEventListener('DOMContentLoaded', () => {
  initCanvas();
  route();
});
