// navbar.js
import { toggleSidebar } from './sidebar.js';

export function renderNavbar(title = 'AlphaAnalytics') {
  const mount = document.getElementById('navbar-mount');
  if (!mount) return;

  mount.innerHTML = `
    <header class="navbar">
      <div class="nav-left">
        <button class="mobile-toggle" id="mobile-menu-btn">
          <i class="fas fa-bars"></i>
        </button>
        <h1 class="page-title text-gradient">${title}</h1>
      </div>
      
      <div class="nav-right">
        <div class="search-box">
          <i class="fas fa-search text-muted"></i>
          <input type="text" placeholder="Search markets (Cmd+K)..." id="global-search" />
        </div>
        
        <button class="icon-btn" title="Notifications">
          <i class="fas fa-bell"></i>
        </button>
        
        <a href="/profile" class="icon-btn" title="Profile">
          <i class="fas fa-user-astronaut"></i>
        </a>
      </div>
    </header>
  `;

  // Attach event listener to mobile toggle
  const mobileToggle = document.getElementById('mobile-menu-btn');
  if (mobileToggle) {
    mobileToggle.addEventListener('click', () => {
      toggleSidebar();
    });
  }
}
