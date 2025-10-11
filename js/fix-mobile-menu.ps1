$files = Get-ChildItem "templates\*.html"

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    
    # Add mobile CSS after <style> tag
    $mobileCSS = @"
/* Mobile Menu Animation */
.menu-toggle span {
  transition: all 0.3s ease;
}

.menu-toggle span.active:nth-child(1) {
  transform: rotate(45deg) translate(6px, 6px);
}

.menu-toggle span.active:nth-child(2) {
  opacity: 0;
}

.menu-toggle span.active:nth-child(3) {
  transform: rotate(-45deg) translate(6px, -6px);
}

/* Ensure mobile menu displays properly */
@media (max-width: 768px) {
  .nav-items {
    display: none;
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: var(--bg-dark);
    border-top: 1px solid var(--border-color);
    padding: 16px;
    flex-direction: column;
    gap: 8px;
    z-index: 1000;
  }
  
  .nav-items.show {
    display: flex;
  }
  
  .nav-item {
    padding: 12px 16px;
    border-radius: 8px;
    justify-content: flex-start;
  }
}
"@

    $mobileJS = @"
// Mobile menu functionality
document.addEventListener('DOMContentLoaded', function() {
  const menuToggle = document.querySelector('.menu-toggle');
  const navItems = document.querySelector('.nav-items');
  
  if (menuToggle && navItems) {
    menuToggle.addEventListener('click', function() {
      navItems.classList.toggle('show');
      // Animate hamburger to X
      const spans = menuToggle.querySelectorAll('span');
      spans.forEach(span => span.classList.toggle('active'));
    });
    
    // Close menu when clicking outside
    document.addEventListener('click', function(event) {
      if (!event.target.closest('.navbar') && navItems.classList.contains('show')) {
        navItems.classList.remove('show');
        const spans = menuToggle.querySelectorAll('span');
        spans.forEach(span => span.classList.remove('active'));
      }
    });
  }
});
"@

    # Insert CSS after <style> tag
    $content = $content -replace '(?<=<style>[^>]*>)', "`n$mobileCSS`n"
    
    # Insert JS after <script> tag
    $content = $content -replace '(?<=<script>[^>]*>)', "`n$mobileJS`n"
    
    Set-Content -Path $file.FullName -Value $content
    Write-Host "Updated: $($file.Name)"
}