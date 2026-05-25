// toast.js

export function showToast(message, type = 'info') {
  const mount = document.getElementById('toast-mount');
  if (!mount) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type} animate-fade-in`;
  
  // Base styling for toast
  toast.style.position = 'fixed';
  toast.style.bottom = '24px';
  toast.style.right = '24px';
  toast.style.padding = '16px 24px';
  toast.style.background = 'var(--bg-glass-heavy)';
  toast.style.backdropFilter = 'blur(10px)';
  toast.style.border = '1px solid var(--border-light)';
  toast.style.borderRadius = 'var(--radius-md)';
  toast.style.color = 'var(--text-primary)';
  toast.style.zIndex = 'var(--z-toast)';
  toast.style.display = 'flex';
  toast.style.alignItems = 'center';
  toast.style.gap = '12px';
  toast.style.boxShadow = 'var(--shadow-md)';

  let icon = 'fa-info-circle';
  let color = 'var(--accent-cyan)';
  
  if (type === 'success') {
    icon = 'fa-check-circle';
    color = 'var(--accent-green)';
  } else if (type === 'error') {
    icon = 'fa-exclamation-circle';
    color = 'var(--down-color)';
  }

  toast.innerHTML = `
    <i class="fas ${icon}" style="color: ${color}; font-size: 20px;"></i>
    <span style="font-weight: 500;">${message}</span>
  `;

  mount.appendChild(toast);

  // Auto remove
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(20px)';
    toast.style.transition = 'all var(--transition-normal)';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
