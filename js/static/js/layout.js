/* ═══════════════════════════════════════════════════════════════════
   AlphaAnalytics v2.0 — Layout Controller
   Sidebar, particles, theme, and global interactivity
   ═══════════════════════════════════════════════════════════════════ */

// ─── Sidebar Manager ─────────────────────────────────────────────
class SidebarManager {
  static init() {
    const sidebar = document.getElementById('sidebar');
    const collapseBtn = document.getElementById('sidebarCollapseBtn');
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');

    if (!sidebar) return;

    // Desktop collapse toggle
    if (collapseBtn) {
      collapseBtn.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
        const isCollapsed = sidebar.classList.contains('collapsed');
        localStorage.setItem('sidebarCollapsed', isCollapsed ? 'true' : 'false');

        const icon = collapseBtn.querySelector('i');
        if (icon) {
          icon.className = isCollapsed ? 'fas fa-chevron-right' : 'fas fa-chevron-left';
        }
      });

      // Restore saved state
      const savedState = localStorage.getItem('sidebarCollapsed');
      if (savedState === 'true') {
        sidebar.classList.add('collapsed');
        const icon = collapseBtn.querySelector('i');
        if (icon) icon.className = 'fas fa-chevron-right';
      }
    }

    // Mobile toggle
    if (mobileMenuBtn) {
      mobileMenuBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        sidebar.classList.toggle('mobile-show');
      });
    }

    // Click-away to close mobile sidebar
    document.addEventListener('click', (e) => {
      if (sidebar.classList.contains('mobile-show') &&
          !sidebar.contains(e.target) &&
          mobileMenuBtn && !mobileMenuBtn.contains(e.target)) {
        sidebar.classList.remove('mobile-show');
      }
    });

    // ESC key to close mobile sidebar
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && sidebar.classList.contains('mobile-show')) {
        sidebar.classList.remove('mobile-show');
      }
    });
  }
}

// ─── Theme Manager ───────────────────────────────────────────────
class ThemeManager {
  static init() {
    // Restore saved theme
    const savedTheme = localStorage.getItem('alpha-theme');
    if (savedTheme === 'light') {
      document.body.classList.add('light');
    }

    // Theme toggle buttons
    document.querySelectorAll('[data-theme-toggle]').forEach(btn => {
      btn.addEventListener('click', () => ThemeManager.toggle());
    });
  }

  static toggle() {
    document.body.classList.toggle('light');
    const isLight = document.body.classList.contains('light');
    localStorage.setItem('alpha-theme', isLight ? 'light' : 'dark');

    // Update toggle button icons
    document.querySelectorAll('[data-theme-toggle] i').forEach(icon => {
      icon.className = isLight ? 'fas fa-moon' : 'fas fa-sun';
    });
  }
}

// ─── Particle Background ─────────────────────────────────────────
class CanvasParticles {
  static init() {
    const canvas = document.getElementById('particlesCanvas');
    if (!canvas) return;

    // Check user preference
    const particlesDisabled = localStorage.getItem('alpha-particles') === 'false';
    if (particlesDisabled) {
      canvas.style.display = 'none';
      return;
    }

    const ctx = canvas.getContext('2d');
    let particles = [];
    let animationId;

    // Fewer particles on mobile for performance
    const isMobile = window.innerWidth < 768;
    const particleCount = isMobile ? 18 : 35;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    let resizeTimer;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(resize, 150);
    });
    resize();

    class Particle {
      constructor() {
        this.reset();
      }
      reset() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 1.5 + 0.5;
        this.speedX = (Math.random() - 0.5) * 0.25;
        this.speedY = (Math.random() - 0.5) * 0.25;
        this.opacity = Math.random() * 0.15 + 0.05;
        this.hue = Math.random() > 0.5 ? 186 : 270; // cyan or purple
      }
      update() {
        this.x += this.speedX;
        this.y += this.speedY;
        if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
        if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;
      }
      draw() {
        ctx.fillStyle = `hsla(${this.hue}, 100%, 60%, ${this.opacity})`;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    for (let i = 0; i < particleCount; i++) {
      particles.push(new Particle());
    }

    // Draw connections between nearby particles (desktop only)
    const drawConnections = () => {
      if (isMobile) return;
      const maxDist = 120;
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < maxDist) {
            const opacity = (1 - dist / maxDist) * 0.06;
            ctx.strokeStyle = `rgba(0, 230, 255, ${opacity})`;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach(p => {
        p.update();
        p.draw();
      });
      drawConnections();
      animationId = requestAnimationFrame(animate);
    };
    animate();

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
      cancelAnimationFrame(animationId);
    });
  }
}

// ─── Global Keyboard Shortcuts ───────────────────────────────────
class KeyboardShortcuts {
  static init() {
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + K → Focus search
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('.search-bar, #searchInput');
        if (searchInput) searchInput.focus();
      }
    });
  }
}

// ─── Auto Init ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  SidebarManager.init();
  ThemeManager.init();
  CanvasParticles.init();
  KeyboardShortcuts.init();
});
