/* ═══════════════════════════════════════════════════════════════════
   AlphaAnalytics v2.0 — Animation Utilities
   Lightweight, pure JS + CSS animation helpers (no dependencies)
   ═══════════════════════════════════════════════════════════════════ */

class AlphaAnimations {

  // ─── Fade In on Scroll (IntersectionObserver) ──────────────────
  static initScrollAnimations() {
    const elements = document.querySelectorAll('.animate-on-scroll, .bento-card, .stat-card, .market-card, .news-card, .video-card, .investor-card, .alert-card, .settings-card, .portfolio-card');
    
    if (!elements.length) return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry, index) => {
        if (entry.isIntersecting) {
          const delay = Math.min(index * 50, 400);
          entry.target.style.transitionDelay = `${delay}ms`;
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.05, rootMargin: '0px 0px -30px 0px' });

    elements.forEach(el => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(16px)';
      el.style.transition = 'opacity 0.5s cubic-bezier(0.4, 0, 0.2, 1), transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
      observer.observe(el);
    });

    // CSS for visible state
    const style = document.createElement('style');
    style.textContent = `.is-visible { opacity: 1 !important; transform: translateY(0) !important; }`;
    document.head.appendChild(style);
  }

  // ─── Animated Counters ─────────────────────────────────────────
  static animateCounter(element, target, duration = 1200) {
    if (!element) return;
    
    const start = 0;
    const startTime = performance.now();
    const isDecimal = String(target).includes('.');
    const prefix = element.dataset.prefix || '';
    const suffix = element.dataset.suffix || '';
    
    const animate = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = start + (target - start) * eased;
      
      if (isDecimal) {
        element.textContent = prefix + current.toFixed(2) + suffix;
      } else {
        element.textContent = prefix + Math.round(current).toLocaleString() + suffix;
      }
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    
    requestAnimationFrame(animate);
  }

  // ─── Init All Counters on Page ─────────────────────────────────
  static initCounters() {
    const counters = document.querySelectorAll('[data-count-to]');
    
    if (!counters.length) return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const target = parseFloat(entry.target.dataset.countTo);
          const duration = parseInt(entry.target.dataset.countDuration) || 1200;
          AlphaAnimations.animateCounter(entry.target, target, duration);
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.3 });

    counters.forEach(el => observer.observe(el));
  }

  // ─── Shimmer Loading ───────────────────────────────────────────
  static showShimmer(element) {
    if (!element) return;
    element.classList.add('shimmer');
    element.dataset.originalContent = element.innerHTML;
    element.innerHTML = '&nbsp;';
    element.style.minHeight = '20px';
  }

  static hideShimmer(element) {
    if (!element) return;
    element.classList.remove('shimmer');
    if (element.dataset.originalContent) {
      element.innerHTML = element.dataset.originalContent;
      delete element.dataset.originalContent;
    }
    element.style.minHeight = '';
  }

  // ─── Hover Tilt (3D) ──────────────────────────────────────────
  static initHoverTilt(selector = '.tilt-card') {
    document.querySelectorAll(selector).forEach(card => {
      card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        const rotateX = (y - centerY) / centerY * -3;
        const rotateY = (x - centerX) / centerX * 3;
        
        card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-4px)`;
      });
      
      card.addEventListener('mouseleave', () => {
        card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateY(0)';
        card.style.transition = 'transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
      });
      
      card.addEventListener('mouseenter', () => {
        card.style.transition = 'transform 0.1s ease';
      });
    });
  }

  // ─── Smooth Page Transition ────────────────────────────────────
  static initPageTransition() {
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.3s ease';
    
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        document.body.style.opacity = '1';
      });
    });
  }

  // ─── Typewriter Effect ─────────────────────────────────────────
  static typewriter(element, texts, speed = 50, pause = 2000) {
    if (!element || !texts.length) return;
    
    let textIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    
    const type = () => {
      const currentText = texts[textIndex];
      
      if (isDeleting) {
        element.textContent = currentText.substring(0, charIndex - 1);
        charIndex--;
      } else {
        element.textContent = currentText.substring(0, charIndex + 1);
        charIndex++;
      }
      
      let delay = isDeleting ? speed / 2 : speed;
      
      if (!isDeleting && charIndex === currentText.length) {
        delay = pause;
        isDeleting = true;
      } else if (isDeleting && charIndex === 0) {
        isDeleting = false;
        textIndex = (textIndex + 1) % texts.length;
        delay = 300;
      }
      
      setTimeout(type, delay);
    };
    
    type();
  }

  // ─── Notification Toast ────────────────────────────────────────
  static showNotification(message, type = 'info', duration = 4000) {
    const icons = {
      success: 'fas fa-check-circle',
      error: 'fas fa-exclamation-circle',
      warning: 'fas fa-exclamation-triangle',
      info: 'fas fa-info-circle'
    };

    let container = document.querySelector('.notification');
    if (!container) {
      container = document.createElement('div');
      container.className = 'notification';
      container.innerHTML = `
        <div class="notification-content ${type}">
          <i class="notification-icon ${icons[type]}"></i>
          <span class="notification-message"></span>
          <button class="notification-close" onclick="this.closest('.notification').classList.remove('show')">
            <i class="fas fa-times"></i>
          </button>
        </div>`;
      document.body.appendChild(container);
    } else {
      const content = container.querySelector('.notification-content');
      content.className = `notification-content ${type}`;
      const icon = container.querySelector('.notification-icon');
      icon.className = `notification-icon ${icons[type]}`;
    }

    container.querySelector('.notification-message').textContent = message;
    
    requestAnimationFrame(() => {
      container.classList.add('show');
    });
    
    if (duration > 0) {
      setTimeout(() => container.classList.remove('show'), duration);
    }
  }

  // ─── Confidence Ring Updater ───────────────────────────────────
  static updateConfidenceRing(element, value) {
    if (!element) return;
    element.style.setProperty('--confidence', value);
    const valueEl = element.querySelector('.ring-value');
    if (valueEl) {
      AlphaAnimations.animateCounter(valueEl, value, 800);
      valueEl.dataset.suffix = '%';
    }
  }

  // ─── Initialize All ───────────────────────────────────────────
  static init() {
    AlphaAnimations.initPageTransition();
    
    // Defer non-critical animations
    requestAnimationFrame(() => {
      AlphaAnimations.initScrollAnimations();
      AlphaAnimations.initCounters();
    });
  }
}

// Auto-init
document.addEventListener('DOMContentLoaded', () => {
  AlphaAnimations.init();
});
