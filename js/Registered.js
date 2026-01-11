// Animation Utilities
class Animations {
    static createParticles() {
        const bg = document.getElementById('animatedBg');
        if (!bg) return;
        
        // Clear existing particles
        const existingParticles = bg.querySelectorAll('.floating-particle');
        existingParticles.forEach(particle => particle.remove());
        
        // Create new particles
        for (let i = 0; i < 15; i++) {
            const particle = document.createElement('div');
            particle.className = 'floating-particle';
            
            const size = Math.random() * 40 + 10;
            const x = Math.random() * 100;
            const y = Math.random() * 100;
            const duration = Math.random() * 20 + 10;
            const delay = Math.random() * 5;
            
            particle.style.width = `${size}px`;
            particle.style.height = `${size}px`;
            particle.style.left = `${x}%`;
            particle.style.top = `${y}%`;
            particle.style.animationDuration = `${duration}s`;
            particle.style.animationDelay = `${delay}s`;
            
            // Random gradient background
            const colors = ['#00e6ff', '#00ff9d', '#764ba2', '#667eea', '#f093fb', '#f5576c'];
            const color1 = colors[Math.floor(Math.random() * colors.length)];
            let color2 = colors[Math.floor(Math.random() * colors.length)];
            
            // Ensure colors are different
            while (color2 === color1) {
                color2 = colors[Math.floor(Math.random() * colors.length)];
            }
            
            particle.style.background = `linear-gradient(45deg, ${color1}, ${color2})`;
            particle.style.opacity = Math.random() * 0.6 + 0.2;
            particle.style.zIndex = '-1';
            
            bg.appendChild(particle);
        }
    }
    
    static shakeElement(element) {
        if (!element) return;
        element.style.animation = 'shake 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
        setTimeout(() => {
            element.style.animation = '';
        }, 500);
    }
    
    static pulseElement(element) {
        if (!element) return;
        const originalTransform = element.style.transform;
        const originalBoxShadow = element.style.boxShadow;
        
        element.style.transform = 'scale(1.05)';
        element.style.boxShadow = '0 0 30px rgba(0, 230, 255, 0.5)';
        
        setTimeout(() => {
            element.style.transform = originalTransform;
            element.style.boxShadow = originalBoxShadow;
        }, 300);
    }
    
    static highlightElement(element) {
        if (!element) return;
        const originalBackground = element.style.background;
        element.style.background = 'linear-gradient(135deg, #00ff9d, #00cc7a)';
        
        setTimeout(() => {
            element.style.background = originalBackground;
        }, 1500);
    }
}

// Notification System
class NotificationSystem {
    static show(message, type = 'info', duration = 5000) {
        // Create container if it doesn't exist
        let container = document.querySelector('.notification-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'notification-container';
            document.body.appendChild(container);
        }
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };
        
        notification.innerHTML = `
            <div class="notification-content-wrapper">
                <i class="fas ${icons[type]} notification-icon"></i>
                <div class="notification-content">
                    <div class="notification-title">${type.charAt(0).toUpperCase() + type.slice(1)}</div>
                    <div class="notification-message">${message}</div>
                </div>
                <button class="notification-close">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="notification-progress">
                <div class="notification-progress-bar"></div>
            </div>
        `;
        
        container.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.classList.add('show');
            
            // Start progress bar
            const progressBar = notification.querySelector('.notification-progress-bar');
            if (progressBar) {
                progressBar.style.animation = `progress ${duration}ms linear forwards`;
            }
        }, 10);
        
        // Close button handler
        const closeBtn = notification.querySelector('.notification-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                this.closeNotification(notification);
            });
        }
        
        // Auto remove
        const autoRemove = setTimeout(() => {
            this.closeNotification(notification);
        }, duration);
        
        // Hover behavior
        notification.addEventListener('mouseenter', () => {
            const progressBar = notification.querySelector('.notification-progress-bar');
            if (progressBar) {
                progressBar.style.animationPlayState = 'paused';
            }
        });
        
        notification.addEventListener('mouseleave', () => {
            const progressBar = notification.querySelector('.notification-progress-bar');
            if (progressBar) {
                progressBar.style.animationPlayState = 'running';
            }
        });
        
        return {
            notification,
            close: () => this.closeNotification(notification),
            updateMessage: (newMessage) => {
                const messageEl = notification.querySelector('.notification-message');
                if (messageEl) {
                    messageEl.textContent = newMessage;
                }
            }
        };
    }
    
    static closeNotification(notification) {
        if (!notification || !notification.parentNode) return;
        
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }
}

// Loading System
class LoadingSystem {
    static show(message = 'Loading...') {
        let overlay = document.getElementById('loadingOverlay');
        
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'loadingOverlay';
            overlay.className = 'loading-overlay';
            overlay.innerHTML = `
                <div class="loading-content">
                    <div class="loading-spinner"></div>
                    <div class="loading-message">${message}</div>
                </div>
            `;
            document.body.appendChild(overlay);
        }
        
        overlay.classList.add('active');
        
        return {
            updateMessage: (newMessage) => {
                const messageEl = overlay.querySelector('.loading-message');
                if (messageEl) {
                    messageEl.textContent = newMessage;
                }
            }
        };
    }
    
    static hide() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.remove('active');
            setTimeout(() => {
                if (overlay.parentNode && !overlay.classList.contains('active')) {
                    overlay.parentNode.removeChild(overlay);
                }
            }, 500);
        }
    }
}

// Form Validation
class FormValidator {
    static validateEmail(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    }
    
    static validatePassword(password) {
        return password.length >= 6;
    }
    
    static validateUsername(username) {
        return username.length >= 3 && /^[a-zA-Z0-9_]+$/.test(username);
    }
    
    static validateName(name) {
        return name.length >= 2 && /^[a-zA-Z\s]+$/.test(name);
    }
    
    static validatePhone(phone) {
        const re = /^[\+]?[1-9][\d]{0,15}$/;
        return re.test(phone.replace(/[\s\-\(\)]/g, ''));
    }
    
    static showError(inputId, message) {
        const errorElement = document.getElementById(inputId + 'Error');
        const inputField = document.getElementById(inputId)?.parentElement;
        
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = 'block';
        }
        
        if (inputField) {
            inputField.classList.add('error');
            Animations.shakeElement(inputField);
            
            // Auto-hide error after 5 seconds
            setTimeout(() => {
                if (errorElement) errorElement.style.display = 'none';
                inputField.classList.remove('error');
            }, 5000);
        }
    }
    
    static showSuccess(inputId) {
        const inputField = document.getElementById(inputId)?.parentElement;
        if (inputField) {
            inputField.classList.add('success');
            setTimeout(() => {
                inputField.classList.remove('success');
            }, 3000);
        }
    }
    
    static clearErrors() {
        document.querySelectorAll('.error-message').forEach(el => {
            el.style.display = 'none';
        });
        document.querySelectorAll('.input-field.error, .input-field.success').forEach(el => {
            el.classList.remove('error', 'success');
        });
    }
    
    static validateForm(formId) {
        const form = document.getElementById(formId);
        if (!form) return false;
        
        let isValid = true;
        const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
        
        inputs.forEach(input => {
            if (!input.value.trim()) {
                this.showError(input.id, 'This field is required');
                isValid = false;
            }
        });
        
        return isValid;
    }
}

// API Service
class APIService {
    static async request(endpoint, options = {}) {
        const baseUrl = 'http://localhost:3000';
        const url = endpoint.startsWith('http') ? endpoint : `${baseUrl}${endpoint}`;
        
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            credentials: 'include',
            ...options
        };
        
        try {
            const response = await fetch(url, defaultOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            }
            
            return await response.text();
        } catch (error) {
            console.error(`API Request Error (${endpoint}):`, error);
            throw error;
        }
    }
    
    static async signup(userData) {
        return await this.request('/signup', {
            method: 'POST',
            body: JSON.stringify(userData)
        });
    }
    
    static async login(username, password) {
        return await this.request('/login', {
            method: 'POST',
            body: JSON.stringify({ username, password })
        });
    }
    
    static async checkAuth() {
        try {
            const data = await this.request('/api/auth/status');
            return data.authenticated ? data.user : null;
        } catch (error) {
            console.log('Auth check failed:', error);
            return null;
        }
    }
    
    static async logout() {
        return await this.request('/logout', {
            method: 'POST'
        });
    }
    
    static async forgotPassword(email) {
        return await this.request('/forgot-password', {
            method: 'POST',
            body: JSON.stringify({ email })
        });
    }
    
    static async resetPassword(token, newPassword) {
        return await this.request('/reset-password', {
            method: 'POST',
            body: JSON.stringify({ token, newPassword })
        });
    }
    
    static async updateProfile(userData) {
        return await this.request('/api/user/profile', {
            method: 'PUT',
            body: JSON.stringify(userData)
        });
    }
}

// Session Manager
class SessionManager {
    static setUser(user) {
        try {
            localStorage.setItem('currentUser', JSON.stringify(user));
            sessionStorage.setItem('sessionActive', 'true');
        } catch (error) {
            console.error('Error saving user data:', error);
        }
    }
    
    static getUser() {
        try {
            const user = localStorage.getItem('currentUser');
            return user ? JSON.parse(user) : null;
        } catch (error) {
            console.error('Error retrieving user data:', error);
            return null;
        }
    }
    
    static clearUser() {
        localStorage.removeItem('currentUser');
        sessionStorage.removeItem('sessionActive');
    }
    
    static isSessionActive() {
        return sessionStorage.getItem('sessionActive') === 'true';
    }
    
    static setRememberMe(username) {
        localStorage.setItem('rememberedUsername', username);
    }
    
    static getRememberedUsername() {
        return localStorage.getItem('rememberedUsername');
    }
    
    static clearRememberMe() {
        localStorage.removeItem('rememberedUsername');
    }
}

// Main Application
class RegistrationApp {
    constructor() {
        this.currentUser = null;
        this.init();
    }
    
    async init() {
        // Create particles
        Animations.createParticles();
        
        // Initialize theme
        this.initTheme();
        
        // Check if user is already logged in
        await this.checkExistingSession();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Setup keyboard shortcuts
        this.setupKeyboardShortcuts();
        
        // Add input animations
        this.setupInputAnimations();
        
        // Setup form auto-save
        this.setupFormAutoSave();
        
        // Refresh particles periodically
        this.setupParticleRefresh();
        
        console.log('AlphaAnalytics Registration App Initialized');
    }
    
    initTheme() {
        const themeToggle = document.getElementById('themeToggle');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const savedTheme = localStorage.getItem('theme') || (prefersDark ? 'dark' : 'light');
        
        if (savedTheme === 'light') {
            document.body.classList.add('light-mode');
        }
        
        if (themeToggle) {
            const icon = themeToggle.querySelector('i');
            if (icon) {
                icon.className = document.body.classList.contains('light-mode') 
                    ? 'fas fa-sun' 
                    : 'fas fa-moon';
            }
            
            themeToggle.addEventListener('click', () => {
                const isLightMode = document.body.classList.toggle('light-mode');
                const icon = themeToggle.querySelector('i');
                
                if (icon) {
                    icon.className = isLightMode ? 'fas fa-sun' : 'fas fa-moon';
                }
                
                localStorage.setItem('theme', isLightMode ? 'light' : 'dark');
                
                NotificationSystem.show(
                    isLightMode ? 'Light mode activated' : 'Dark mode activated',
                    'info',
                    2000
                );
                
                // Recreate particles with theme-appropriate colors
                setTimeout(() => Animations.createParticles(), 100);
            });
        }
    }
    
    async checkExistingSession() {
        try {
            const user = await APIService.checkAuth();
            if (user) {
                this.currentUser = user;
                SessionManager.setUser(user);
                
                // If already authenticated and trying to access registration page, redirect
                if (window.location.pathname.includes('index.html') || 
                    window.location.pathname === '/' ||
                    window.location.pathname.includes('registration.html')) {
                    
                    const notification = NotificationSystem.show(
                        `Welcome back, ${user.username}! Redirecting to dashboard...`,
                        'success'
                    );
                    
                    setTimeout(() => {
                        window.location.href = 'dashboard.html';
                    }, 1500);
                }
            }
        } catch (error) {
            console.log('No existing session found or error checking session:', error);
        }
    }
    
    setupEventListeners() {
        // Form switching
        const sign_in_btn = document.querySelector("#sign-in-btn");
        const sign_up_btn = document.querySelector("#sign-up-btn");
        const switchToSignIn = document.getElementById('switchToSignIn');
        const switchToSignUp = document.getElementById('switchToSignUp');
        const container = document.querySelector(".container");
        
        const switchToSignUpHandler = () => {
            if (container) {
                container.classList.add("sign-up-mode");
                Animations.pulseElement(sign_up_btn);
                NotificationSystem.show('Switched to Sign Up form', 'info', 1500);
                this.loadSavedFormData('signUpForm');
            }
        };
        
        const switchToSignInHandler = () => {
            if (container) {
                container.classList.remove("sign-up-mode");
                Animations.pulseElement(sign_in_btn);
                NotificationSystem.show('Switched to Sign In form', 'info', 1500);
                this.loadSavedFormData('signInForm');
            }
        };
        
        if (sign_up_btn) sign_up_btn.addEventListener("click", switchToSignUpHandler);
        if (sign_in_btn) sign_in_btn.addEventListener("click", switchToSignInHandler);
        if (switchToSignUp) switchToSignUp.addEventListener("click", switchToSignUpHandler);
        if (switchToSignIn) switchToSignIn.addEventListener("click", switchToSignInHandler);
        
        // Sign Up Form
        const signUpForm = document.getElementById('signUpForm');
        if (signUpForm) {
            signUpForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                await this.handleSignUp();
            });
        }
        
        // Sign In Form
        const signInForm = document.getElementById('signInForm');
        if (signInForm) {
            signInForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                await this.handleSignIn();
            });
        }
        
        // Forgot password
        const forgotPassword = document.getElementById('forgotPassword');
        if (forgotPassword) {
            forgotPassword.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleForgotPassword();
            });
        }
        
        // Terms & Privacy
        const showTerms = document.getElementById('showTerms');
        const showPrivacy = document.getElementById('showPrivacy');
        
        if (showTerms) {
            showTerms.addEventListener('click', (e) => {
                e.preventDefault();
                this.showModal('Terms & Conditions', `
                    <h3>Terms of Service</h3>
                    <p>Welcome to AlphaAnalytics! By using our service, you agree to:</p>
                    <ul>
                        <li>Provide accurate registration information</li>
                        <li>Maintain the confidentiality of your account</li>
                        <li>Use the service in compliance with applicable laws</li>
                        <li>Not engage in unauthorized access or use</li>
                    </ul>
                    <p>We reserve the right to suspend accounts violating these terms.</p>
                `);
            });
        }
        
        if (showPrivacy) {
            showPrivacy.addEventListener('click', (e) => {
                e.preventDefault();
                this.showModal('Privacy Policy', `
                    <h3>Your Privacy Matters</h3>
                    <p>We collect and use your information to:</p>
                    <ul>
                        <li>Provide and improve our services</li>
                        <li>Personalize your experience</li>
                        <li>Communicate important updates</li>
                        <li>Ensure account security</li>
                    </ul>
                    <p>We never sell your personal data to third parties.</p>
                `);
            });
        }
        
        // Social login buttons
        document.querySelectorAll('.social-icon').forEach(icon => {
            icon.addEventListener('click', (e) => {
                e.preventDefault();
                Animations.pulseElement(icon);
                this.handleSocialLogin(icon.dataset.provider || 'unknown');
            });
        });
        
        // View password toggle
        document.querySelectorAll('.view-password').forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const input = button.parentElement.querySelector('input');
                if (input) {
                    const type = input.type === 'password' ? 'text' : 'password';
                    input.type = type;
                    const icon = button.querySelector('i');
                    if (icon) {
                        icon.className = type === 'password' ? 'fas fa-eye' : 'fas fa-eye-slash';
                    }
                }
            });
        });
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Don't trigger if user is typing in an input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }
            
            switch(e.key) {
                case 'l':
                    if (e.ctrlKey) {
                        e.preventDefault();
                        this.switchToSignIn();
                    }
                    break;
                    
                case 'r':
                    if (e.ctrlKey) {
                        e.preventDefault();
                        this.switchToSignUp();
                    }
                    break;
                    
                case 'Escape':
                    FormValidator.clearErrors();
                    this.closeModal();
                    break;
                    
                case 'F5':
                    if (e.ctrlKey) {
                        e.preventDefault();
                        Animations.createParticles();
                        NotificationSystem.show('Refreshed animations', 'info');
                    }
                    break;
            }
        });
    }
    
    setupInputAnimations() {
        document.querySelectorAll('.input-field input, .input-field textarea').forEach(input => {
            // Focus effects
            input.addEventListener('focus', function() {
                const parent = this.parentElement;
                if (parent) {
                    parent.style.transform = 'translateY(-2px) scale(1.01)';
                    parent.style.boxShadow = '0 10px 30px rgba(0, 230, 255, 0.15)';
                    parent.classList.add('focused');
                }
            });
            
            input.addEventListener('blur', function() {
                const parent = this.parentElement;
                if (parent) {
                    parent.style.transform = '';
                    parent.style.boxShadow = '';
                    parent.classList.remove('focused');
                    
                    // Validate on blur
                    if (this.value.trim()) {
                        this.validateField();
                    }
                }
            });
            
            // Add validation method to input
            input.validateField = function() {
                const value = this.value.trim();
                const id = this.id;
                
                if (!value) return false;
                
                switch(this.type) {
                    case 'email':
                        if (FormValidator.validateEmail(value)) {
                            FormValidator.showSuccess(id);
                            return true;
                        } else {
                            FormValidator.showError(id, 'Please enter a valid email');
                            return false;
                        }
                        
                    case 'password':
                        if (FormValidator.validatePassword(value)) {
                            FormValidator.showSuccess(id);
                            return true;
                        } else {
                            FormValidator.showError(id, 'Password must be at least 6 characters');
                            return false;
                        }
                        
                    default:
                        if (id.includes('username')) {
                            if (FormValidator.validateUsername(value)) {
                                FormValidator.showSuccess(id);
                                return true;
                            } else {
                                FormValidator.showError(id, 'Username must be 3+ alphanumeric characters');
                                return false;
                            }
                        }
                        return true;
                }
            };
        });
    }
    
    setupFormAutoSave() {
        const forms = ['signUpForm', 'signInForm'];
        forms.forEach(formId => {
            const form = document.getElementById(formId);
            if (!form) return;
            
            const inputs = form.querySelectorAll('input, textarea');
            const storageKey = `form_${formId}_data`;
            
            // Load saved data
            const savedData = localStorage.getItem(storageKey);
            if (savedData) {
                try {
                    const data = JSON.parse(savedData);
                    inputs.forEach(input => {
                        if (data[input.id]) {
                            input.value = data[input.id];
                        }
                    });
                } catch (error) {
                    console.error('Error loading saved form data:', error);
                }
            }
            
            // Save on input
            inputs.forEach(input => {
                input.addEventListener('input', () => {
                    const data = {};
                    inputs.forEach(i => {
                        if (i.id) {
                            data[i.id] = i.value;
                        }
                    });
                    localStorage.setItem(storageKey, JSON.stringify(data));
                });
            });
            
            // Clear on submit
            form.addEventListener('submit', () => {
                localStorage.removeItem(storageKey);
            });
        });
    }
    
    setupParticleRefresh() {
        // Refresh particles every 60 seconds
        setInterval(() => {
            Animations.createParticles();
        }, 60000);
        
        // Also refresh on window resize
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                Animations.createParticles();
            }, 250);
        });
    }
    
    async handleSignUp() {
        FormValidator.clearErrors();
        
        const username = document.getElementById('signupUsername')?.value.trim();
        const email = document.getElementById('signupEmail')?.value.trim();
        const password = document.getElementById('signupPassword')?.value;
        const confirmPassword = document.getElementById('confirmPassword')?.value;
        const termsAgree = document.getElementById('termsAgree')?.checked;
        
        // Validation
        let isValid = true;
        
        if (!username) {
            FormValidator.showError('signupUsername', 'Username is required');
            isValid = false;
        } else if (!FormValidator.validateUsername(username)) {
            FormValidator.showError('signupUsername', 'Username must be at least 3 characters and can only contain letters, numbers, and underscores');
            isValid = false;
        }
        
        if (!email) {
            FormValidator.showError('signupEmail', 'Email is required');
            isValid = false;
        } else if (!FormValidator.validateEmail(email)) {
            FormValidator.showError('signupEmail', 'Please enter a valid email address');
            isValid = false;
        }
        
        if (!password) {
            FormValidator.showError('signupPassword', 'Password is required');
            isValid = false;
        } else if (!FormValidator.validatePassword(password)) {
            FormValidator.showError('signupPassword', 'Password must be at least 6 characters');
            isValid = false;
        }
        
        if (!confirmPassword) {
            FormValidator.showError('confirmPassword', 'Please confirm your password');
            isValid = false;
        } else if (password !== confirmPassword) {
            FormValidator.showError('confirmPassword', 'Passwords do not match');
            isValid = false;
        }
        
        if (!termsAgree) {
            NotificationSystem.show('Please agree to the terms and conditions', 'warning');
            const termsCheckbox = document.getElementById('termsAgree');
            if (termsCheckbox) {
                Animations.shakeElement(termsCheckbox.parentElement);
            }
            isValid = false;
        }
        
        if (!isValid) return;
        
        const loading = LoadingSystem.show('Creating your account...');
        
        try {
            const userData = {
                username,
                email,
                password,
                createdAt: new Date().toISOString()
            };
            
            const response = await APIService.signup(userData);
            
            if (response.success || response.message) {
                NotificationSystem.show(
                    response.message || 'Registration successful! Please sign in.',
                    'success',
                    3000
                );
                
                // Switch to sign in form with animation
                this.switchToSignIn();
                
                // Clear form
                const signUpForm = document.getElementById('signUpForm');
                if (signUpForm) signUpForm.reset();
                
                // Auto-fill username in login form
                const loginUsernameInput = document.getElementById('loginUsername');
                if (loginUsernameInput) {
                    loginUsernameInput.value = username;
                    loginUsernameInput.focus();
                }
                
                // Animate success
                const signInForm = document.querySelector('.sign-in-form');
                if (signInForm) {
                    Animations.highlightElement(signInForm);
                }
                
            } else if (response.error) {
                NotificationSystem.show(response.error, 'error');
                
                if (response.error.toLowerCase().includes('username')) {
                    FormValidator.showError('signupUsername', 'Username already exists');
                } else if (response.error.toLowerCase().includes('email')) {
                    FormValidator.showError('signupEmail', 'Email already in use');
                }
            }
        } catch (error) {
            console.error('Signup error:', error);
            NotificationSystem.show(
                error.message || 'Error connecting to server. Please try again.',
                'error'
            );
        } finally {
            loading.hide();
        }
    }
    
    async handleSignIn() {
        FormValidator.clearErrors();
        
        const username = document.getElementById('loginUsername')?.value.trim();
        const password = document.getElementById('loginPassword')?.value;
        const rememberMe = document.getElementById('rememberMe')?.checked;
        
        // Validation
        if (!username) {
            FormValidator.showError('loginUsername', 'Username is required');
            return;
        }
        
        if (!password) {
            FormValidator.showError('loginPassword', 'Password is required');
            return;
        }
        
        const loading = LoadingSystem.show('Signing you in...');
        
        try {
            const response = await APIService.login(username, password);
            
            if (response.success || response.message) {
                // Store user preference
                if (rememberMe) {
                    SessionManager.setRememberMe(username);
                } else {
                    SessionManager.clearRememberMe();
                }
                
                // Store user info
                if (response.user) {
                    this.currentUser = response.user;
                    SessionManager.setUser(response.user);
                }
                
                NotificationSystem.show(
                    response.message || 'Login successful! Redirecting...',
                    'success',
                    2000
                );
                
                // Animate success
                const loginBtn = document.getElementById('loginBtn');
                if (loginBtn) {
                    const originalHTML = loginBtn.innerHTML;
                    const originalBackground = loginBtn.style.background;
                    
                    loginBtn.innerHTML = '<i class="fas fa-check"></i> Success!';
                    loginBtn.style.background = 'linear-gradient(135deg, #00ff9d, #00cc7a)';
                    loginBtn.disabled = true;
                    
                    // Redirect after animation
                    setTimeout(() => {
                        window.location.href = 'dashboard.html';
                    }, 1500);
                } else {
                    // Fallback redirect
                    setTimeout(() => {
                        window.location.href = 'dashboard.html';
                    }, 1500);
                }
                
            } else if (response.error) {
                NotificationSystem.show(response.error, 'error');
                
                if (response.error.toLowerCase().includes('username') || 
                    response.error.toLowerCase().includes('invalid') ||
                    response.error.toLowerCase().includes('password')) {
                    
                    FormValidator.showError('loginUsername', 'Invalid username or password');
                    FormValidator.showError('loginPassword', 'Invalid username or password');
                    
                    // Shake the form
                    const signInForm = document.querySelector('.sign-in-form');
                    if (signInForm) {
                        Animations.shakeElement(signInForm);
                    }
                }
            }
        } catch (error) {
            console.error('Login error:', error);
            NotificationSystem.show(
                error.message || 'Error connecting to server. Please try again.',
                'error'
            );
        } finally {
            loading.hide();
        }
    }
    
    async handleForgotPassword() {
        const email = prompt('Please enter your email address to reset your password:');
        if (!email) return;
        
        if (!FormValidator.validateEmail(email)) {
            NotificationSystem.show('Please enter a valid email address', 'error');
            return;
        }
        
        const loading = LoadingSystem.show('Sending reset instructions...');
        
        try {
            const response = await APIService.forgotPassword(email);
            
            if (response.success || response.message) {
                NotificationSystem.show(
                    response.message || 'Password reset instructions sent to your email.',
                    'success'
                );
            } else if (response.error) {
                NotificationSystem.show(response.error, 'error');
            }
        } catch (error) {
            console.error('Forgot password error:', error);
            NotificationSystem.show('Error sending reset instructions', 'error');
        } finally {
            loading.hide();
        }
    }
    
    async handleSocialLogin(provider) {
        const providers = {
            google: 'Google',
            facebook: 'Facebook',
            github: 'GitHub',
            twitter: 'Twitter'
        };
        
        const providerName = providers[provider] || provider;
        NotificationSystem.show(`${providerName} login coming soon!`, 'info');
    }
    
    switchToSignIn() {
        const container = document.querySelector(".container");
        const sign_in_btn = document.querySelector("#sign-in-btn");
        
        if (container) container.classList.remove("sign-up-mode");
        if (sign_in_btn) Animations.pulseElement(sign_in_btn);
        
        NotificationSystem.show('Switched to Sign In form', 'info', 1500);
    }
    
    switchToSignUp() {
        const container = document.querySelector(".container");
        const sign_up_btn = document.querySelector("#sign-up-btn");
        
        if (container) container.classList.add("sign-up-mode");
        if (sign_up_btn) Animations.pulseElement(sign_up_btn);
        
        NotificationSystem.show('Switched to Sign Up form', 'info', 1500);
    }
    
    showModal(title, content) {
        // Remove existing modal
        this.closeModal();
        
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>${title}</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
                <div class="modal-footer">
                    <button class="btn btn-primary modal-ok">OK</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Add animations
        setTimeout(() => {
            modal.classList.add('active');
            document.querySelector('.modal-content')?.classList.add('active');
        }, 10);
        
        // Close handlers
        const closeModal = () => this.closeModal();
        
        modal.querySelector('.modal-close')?.addEventListener('click', closeModal);
        modal.querySelector('.modal-ok')?.addEventListener('click', closeModal);
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeModal();
        });
        
        // Escape key
        const escapeHandler = (e) => {
            if (e.key === 'Escape') closeModal();
        };
        document.addEventListener('keydown', escapeHandler);
        
        // Store handler for cleanup
        this.currentModal = { modal, escapeHandler };
    }
    
    closeModal() {
        if (this.currentModal) {
            const { modal, escapeHandler } = this.currentModal;
            document.removeEventListener('keydown', escapeHandler);
            
            modal.classList.remove('active');
            setTimeout(() => {
                if (modal.parentNode) {
                    modal.parentNode.removeChild(modal);
                }
            }, 300);
            
            this.currentModal = null;
        }
    }
    
    loadSavedFormData(formId) {
        const storageKey = `form_${formId}_data`;
        const savedData = localStorage.getItem(storageKey);
        
        if (savedData) {
            try {
                const data = JSON.parse(savedData);
                Object.keys(data).forEach(id => {
                    const input = document.getElementById(id);
                    if (input) {
                        input.value = data[id];
                    }
                });
            } catch (error) {
                console.error('Error loading form data:', error);
            }
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create global app instance
    window.registrationApp = new RegistrationApp();
    
    // Auto-fill saved username
    const savedUsername = SessionManager.getRememberedUsername();
    if (savedUsername) {
        const loginUsernameInput = document.getElementById('loginUsername');
        const rememberMeCheckbox = document.getElementById('rememberMe');
        
        if (loginUsernameInput) {
            loginUsernameInput.value = savedUsername;
        }
        if (rememberMeCheckbox) {
            rememberMeCheckbox.checked = true;
        }
    }
    
    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) {
            // Refresh animations when page becomes visible
            Animations.createParticles();
        }
    });
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    
    // Don't show notification for network errors to avoid spam
    if (event.error instanceof TypeError && event.error.message.includes('fetch')) {
        return;
    }
    
    NotificationSystem.show(
        'An unexpected error occurred. Please refresh the page.',
        'error'
    );
});

// Unhandled promise rejection handler
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    
    if (event.reason instanceof Error) {
        NotificationSystem.show(
            `Error: ${event.reason.message}`,
            'error'
        );
    }
});

// Export for global access
window.Animations = Animations;
window.NotificationSystem = NotificationSystem;
window.LoadingSystem = LoadingSystem;
window.FormValidator = FormValidator;
window.APIService = APIService;
window.SessionManager = SessionManager;
window.RegistrationApp = RegistrationApp;