// Sidebar collapse controller
class SidebarManager {
  static init() {
    const sidebar = document.getElementById('sidebar');
    const collapseBtn = document.getElementById('sidebarCollapseBtn');
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    
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
      
      const savedState = localStorage.getItem('sidebarCollapsed');
      if (savedState === 'true') {
        sidebar.classList.add('collapsed');
        const icon = collapseBtn.querySelector('i');
        if (icon) icon.className = 'fas fa-chevron-right';
      }
    }
    
    if (mobileMenuBtn) {
      mobileMenuBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        sidebar.classList.toggle('mobile-show');
      });
    }
    
    document.addEventListener('click', (e) => {
      if (sidebar.classList.contains('mobile-show') && 
          !sidebar.contains(e.target) && 
          !mobileMenuBtn.contains(e.target)) {
        sidebar.classList.remove('mobile-show');
      }
    });
  }
}

// Particle background script
class CanvasParticles {
  static init() {
    const canvas = document.getElementById('particlesCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let particles = [];
    
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', resize);
    resize();
    
    class Particle {
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 2 + 1;
        this.speedX = Math.random() * 0.3 - 0.15;
        this.speedY = Math.random() * 0.3 - 0.15;
        this.color = Math.random() > 0.5 ? 'rgba(0, 230, 255, 0.15)' : 'rgba(138, 43, 226, 0.15)';
      }
      update() {
        this.x += this.speedX;
        this.y += this.speedY;
        if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
        if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;
      }
      draw() {
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    
    for (let i = 0; i < 40; i++) {
      particles.push(new Particle());
    }
    
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach(p => {
        p.update();
        p.draw();
      });
      requestAnimationFrame(animate);
    };
    animate();
  }
}

// Auto init on page load
document.addEventListener('DOMContentLoaded', () => {
  SidebarManager.init();
  CanvasParticles.init();
});
