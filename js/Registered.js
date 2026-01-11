
// Animation Utilities
class Animations {
    static createParticles() {
        const bg = document.getElementById('animatedBg');
        if (!bg) return;
        
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
            const colors = ['#00e6ff', '#00ff9d', '#764ba2', '#667eea'];
            const color1 = colors[Math.floor(Math.random() * colors.length)];
            const color2 = colors[Math.floor(Math.random() * colors.length)];
            particle.style.background = `linear-gradient(45deg, ${color1}, ${color2})`;
            
            bg.appendChild(particle);
        }
    }
    
    static shakeElement(element) {
        element.style.animation = 'shake 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
        setTimeout(() => {
            element.style.animation = '';
        }, 500);
    }
    
    static pulseElement(element) {
        element.style.transform = 'scale(1.05)';
        element.style.boxShadow = '0 0 30px rgba(0, 230, 255, 0.5)';
        setTimeout(() => {
            element.style.transform = '';
            element.style.boxShadow = '';
        }, 300);
    }
}

// Notification System
class NotificationSystem {
    static show(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };
        
        notification.innerHTML = `
            <i class="fas ${icons[type]} notification-icon"></i>
            <div class="notification-content">
                <div class="notification-title">${type.charAt(0).toUpperCase() + type.slice(1)}</div>
                <div class="notification-message">${message}</div>
            </div>
            <button class="notification-close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        // Close button handler
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 500);
        });
        
        // Auto remove
        setTimeout(() => {
            if (notification.parentNode) {
                notification.classList.remove('show');
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.parentNode.removeChild(notification);
                    }
                }, 500);
            }
        }, duration);
        
        return notification;
    }
}

// Loading System
class LoadingSystem {
    static show() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.add('active');
        }
    }
    
    static hide() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.remove('active');
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
    
    static showError(inputId, message) {
        const errorElement = document.getElementById(inputId + 'Error');
        const inputField = document.getElementById(inputId).parentElement;
        
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = 'block';
        }
        
        if (inputField) {
            inputField.classList.add('error');
            Animations.shakeElement(inputField);
            
            setTimeout(() => {
                if (errorElement) errorElement.style.display = 'none';
                inputField.classList.remove('error');
            }, 5000);
        }
    }
    
    static clearErrors() {
        document.querySelectorAll('.error-message').forEach(el => {
            el.style.display = 'none';
        });
        document.querySelectorAll('.input-field.error').forEach(el => {
            el.classList.remove('error');
        });
    }
}

// API Service
class APIService {
    static async signup(username, email, password) {
        try {
            const response = await fetch('http://localhost:3000/signup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, email, password })
            });
            
            return await response.json();
        } catch (error) {
            console.error('Signup API Error:', error);
            throw error;
        }
    }
    
    static async login(username, password) {
        try {
            const response = await fetch('http://localhost:3000/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify({ username, password })
            });
            
            return await response.json();
        } catch (error) {
            console.error('Login API Error:', error);
            throw error;
        }
    }
    
    static async checkAuth() {
        try {
            const response = await fetch('http://localhost:3000/me', {
                method: 'GET',
                credentials: 'include'
            });
            
            return response.ok ? await response.json() : null;
        } catch (error) {
            return null;
        }
    }
}

// Main Application
class RegistrationApp {
    constructor() {
        this.init();
    }
    
    init() {
        // Create particles
        Animations.createParticles();
        
        // Initialize theme
        this.initTheme();
        
        // Check if user is already logged in
        this.checkExistingSession();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Setup keyboard shortcuts
        this.setupKeyboardShortcuts();
        
        // Add input animations
        this.setupInputAnimations();
        
        console.log('AlphaAnalytics Registration App Initialized');
    }
    
    initTheme() {
        const themeToggle = document.getElementById('themeToggle');
        const savedTheme = localStorage.getItem('theme');
        
        if (savedTheme === 'light') {
            document.body.classList.add('light-mode');
            themeToggle.querySelector('i').className = 'fas fa-sun';
        }
        
        themeToggle.addEventListener('click', () => {
            document.body.classList.toggle('light-mode');
            const icon = themeToggle.querySelector('i');
            
            if (document.body.classList.contains('light-mode')) {
                icon.className = 'fas fa-sun';
                localStorage.setItem('theme', 'light');
                NotificationSystem.show('Light mode activated', 'info');
            } else {
                icon.className = 'fas fa-moon';
                localStorage.setItem('theme', 'dark');
                NotificationSystem.show('Dark mode activated', 'info');
            }
        });
    }
    
    async checkExistingSession() {
        try {
            const user = await APIService.checkAuth();
            if (user) {
                NotificationSystem.show(`Welcome back, ${user.username}! Redirecting...`, 'success');
                setTimeout(() => {
                    window.location.href = 'jeet.html';
                }, 1500);
            }
        } catch (error) {
            console.log('No existing session found');
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
            container.classList.add("sign-up-mode");
            Animations.pulseElement(sign_up_btn);
            NotificationSystem.show('Switched to Sign Up form', 'info');
        };
        
        const switchToSignInHandler = () => {
            container.classList.remove("sign-up-mode");
            Animations.pulseElement(sign_in_btn);
            NotificationSystem.show('Switched to Sign In form', 'info');
        };
        
        sign_up_btn.addEventListener("click", switchToSignUpHandler);
        sign_in_btn.addEventListener("click", switchToSignInHandler);
        switchToSignUp?.addEventListener("click", switchToSignUpHandler);
        switchToSignIn?.addEventListener("click", switchToSignInHandler);
        
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
                NotificationSystem.show('Password reset feature coming soon!', 'info');
            });
        }
        
        // Terms & Privacy
        const showTerms = document.getElementById('showTerms');
        const showPrivacy = document.getElementById('showPrivacy');
        
        if (showTerms) {
            showTerms.addEventListener('click', (e) => {
                e.preventDefault();
                NotificationSystem.show('Terms & Conditions will be displayed here', 'info');
            });
        }
        
        if (showPrivacy) {
            showPrivacy.addEventListener('click', (e) => {
                e.preventDefault();
                NotificationSystem.show('Privacy Policy will be displayed here', 'info');
            });
        }
        
        // Social login buttons
        document.querySelectorAll('.social-icon').forEach(icon => {
            icon.addEventListener('click', (e) => {
                e.preventDefault();
                Animations.pulseElement(icon);
                NotificationSystem.show('Social login coming soon!', 'info');
            });
        });
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'l') {
                e.preventDefault();
                const container = document.querySelector(".container");
                container.classList.remove("sign-up-mode");
                NotificationSystem.show('Switched to Login form (Ctrl+L)', 'info');
            }
            
            if (e.ctrlKey && e.key === 'r') {
                e.preventDefault();
                const container = document.querySelector(".container");
                container.classList.add("sign-up-mode");
                NotificationSystem.show('Switched to Register form (Ctrl+R)', 'info');
            }
            
            if (e.key === 'Escape') {
                FormValidator.clearErrors();
            }
        });
    }
    
    setupInputAnimations() {
        document.querySelectorAll('.input-field input').forEach(input => {
            input.addEventListener('focus', function() {
                this.parentElement.style.transform = 'translateY(-5px) scale(1.02)';
                this.parentElement.style.boxShadow = '0 15px 40px rgba(0, 230, 255, 0.2)';
            });
            
            input.addEventListener('blur', function() {
                this.parentElement.style.transform = '';
                this.parentElement.style.boxShadow = '';
            });
            
            // Add typing animation
            input.addEventListener('input', function() {
                if (this.value.length > 0) {
                    this.parentElement.querySelector('i').style.transform = 'scale(1.2)';
                } else {
                    this.parentElement.querySelector('i').style.transform = '';
                }
            });
        });
    }
    
    async handleSignUp() {
        FormValidator.clearErrors();
        
        const username = document.getElementById('signupUsername').value.trim();
        const email = document.getElementById('signupEmail').value.trim();
        const password = document.getElementById('signupPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        const termsAgree = document.getElementById('termsAgree').checked;
        
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
        
        LoadingSystem.show();
        
        try {
            const response = await APIService.signup(username, email, password);
            
            if (response.message) {
                NotificationSystem.show('Registration successful! Please sign in.', 'success');
                
                // Switch to sign in form with animation
                const container = document.querySelector(".container");
                container.classList.remove("sign-up-mode");
                
                // Clear form
                document.getElementById('signUpForm').reset();
                
                // Auto-fill username in login form
                document.getElementById('loginUsername').value = username;
                
                // Pulse the login form
                const loginForm = document.querySelector('.sign-in-form');
                Animations.pulseElement(loginForm);
                
            } else if (response.error) {
                NotificationSystem.show(response.error, 'error');
                
                if (response.error.includes('Username')) {
                    FormValidator.showError('signupUsername', 'Username already exists');
                } else if (response.error.includes('Email')) {
                    FormValidator.showError('signupEmail', 'Email already in use');
                }
            }
        } catch (error) {
            console.error('Signup error:', error);
            NotificationSystem.show('Error connecting to server. Please try again.', 'error');
        } finally {
            LoadingSystem.hide();
        }
    }
    
    async handleSignIn() {
        FormValidator.clearErrors();
        
        const username = document.getElementById('loginUsername').value.trim();
        const password = document.getElementById('loginPassword').value;
        const rememberMe = document.getElementById('rememberMe').checked;
        
        // Validation
        if (!username) {
            FormValidator.showError('loginUsername', 'Username is required');
            return;
        }
        
        if (!password) {
            FormValidator.showError('loginPassword', 'Password is required');
            return;
        }
        
        LoadingSystem.show();
        
        try {
            const response = await APIService.login(username, password);
            
            if (response.message) {
                NotificationSystem.show('Login successful! Redirecting...', 'success');
                
                // Store user preference
                if (rememberMe) {
                    localStorage.setItem('username', username);
                }
                
                // Animate success
                const loginBtn = document.getElementById('loginBtn');
                if (loginBtn) {
                    loginBtn.innerHTML = '<i class="fas fa-check"></i> Success!';
                    loginBtn.style.background = 'linear-gradient(135deg, #00ff9d, #00cc7a)';
                }
                
                // Redirect after animation
                setTimeout(() => {
                    window.location.href = 'jeet.html';
                }, 1500);
                
            } else if (response.error) {
                NotificationSystem.show(response.error, 'error');
                
                if (response.error.includes('Username') || response.error.includes('Invalid')) {
                    FormValidator.showError('loginUsername', 'Invalid username or password');
                    FormValidator.showError('loginPassword', 'Invalid username or password');
                }
            }
        } catch (error) {
            console.error('Login error:', error);
            NotificationSystem.show('Error connecting to server. Please try again.', 'error');
        } finally {
            LoadingSystem.hide();
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.registrationApp = new RegistrationApp();
});

// Auto-fill saved username
window.addEventListener('load', () => {
    const savedUsername = localStorage.getItem('username');
    if (savedUsername && document.getElementById('loginUsername')) {
        document.getElementById('loginUsername').value = savedUsername;
        document.getElementById('rememberMe').checked = true;
    }
});
