// sidebar.js

export function renderSidebar(currentPath) {
  const mount = document.getElementById('sidebar-mount');
  if (!mount) return;

  const links = [
    { path: '/jeet', icon: 'fa-chart-pie', label: 'Dashboard' },
    { path: '/prediction', icon: 'fa-robot', label: 'Predictions' },
    { path: '/portfolio', icon: 'fa-briefcase', label: 'Portfolio' },
    { path: '/deposit', icon: 'fa-wallet', label: 'Deposit' },
    { path: '/mystock', icon: 'fa-star', label: 'Watchlist' },
    { path: '/insight', icon: 'fa-brain', label: 'AI Insights' },
    { path: '/news', icon: 'fa-newspaper', label: 'News' },
    { path: '/alerts', icon: 'fa-bell', label: 'Alerts' },
    { path: '/videos', icon: 'fa-play-circle', label: 'Videos' },
    { path: '/Superstars', icon: 'fa-trophy', label: 'Superstars' },
    { path: '/profile', icon: 'fa-user', label: 'Profile' },
    { path: '/setting', icon: 'fa-cog', label: 'Settings' }
  ];

  const currentExactPath = window.location.pathname;

  mount.innerHTML = `
    <aside class="sidebar" id="main-sidebar">
      <div class="sidebar-header">
        <a href="/" class="brand">
          <i class="fas fa-chart-line brand-icon"></i>
          <span class="brand-text">AlphaAnalytics</span>
        </a>
      </div>
      <nav class="nav-links">
        ${links.map(link => `
          <a href="${link.path}" class="nav-item ${currentExactPath === link.path || currentExactPath === link.path + '.html' ? 'active' : ''}">
            <i class="fas ${link.icon}"></i>
            <span class="nav-text">${link.label}</span>
          </a>
        `).join('')}
      </nav>
    </aside>
  `;
}

export function toggleSidebar() {
  const sidebar = document.getElementById('main-sidebar');
  if (sidebar) {
    sidebar.classList.toggle('open');
  }
}
