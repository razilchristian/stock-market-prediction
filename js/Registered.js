// Select elements
const sign_in_btn = document.querySelector("#sign-in-btn");
const sign_up_btn = document.querySelector("#sign-up-btn");
const container = document.querySelector(".container");

// Sign-up form elements
const signUpForm = document.querySelector(".sign-up-form");
const usernameInputSignUp = signUpForm.querySelector('input[placeholder="Username"]');
const emailInputSignUp = signUpForm.querySelector('input[placeholder="Email"]');
const passwordInputSignUp = signUpForm.querySelector('input[placeholder="Password"]');

// Sign-in form elements
const signInForm = document.querySelector(".sign-in-form");
const usernameInputSignIn = signInForm.querySelector('input[placeholder="Username"]');
const passwordInputSignIn = signInForm.querySelector('input[placeholder="Password"]');

// Toggle between sign-up and sign-in modes
sign_up_btn.addEventListener("click", () => {
  container.classList.add("sign-up-mode");
});

sign_in_btn.addEventListener("click", () => {
  container.classList.remove("sign-up-mode");
});

// Handle Sign-up Form Submission
signUpForm.addEventListener("submit", async (e) => {
  e.preventDefault(); // Prevent page reload

  const username = usernameInputSignUp.value.trim();
  const email = emailInputSignUp.value.trim();
  const password = passwordInputSignUp.value.trim();

  // Basic validation
  if (!username || !email || !password) {
    alert("Please fill in all fields.");
    return;
  }

  // Email format validation (basic)
  const emailRegex = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$/;
  if (!emailRegex.test(email)) {
    alert("Invalid email format.");
    return;
  }

  // Make POST request to the signup endpoint
  try {
    const response = await fetch('http://localhost:3000/signup', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, email, password }),
    });

    const data = await response.json();
    if (response.status === 201) {
      alert("User registered successfully!");
      // Clear form fields
      usernameInputSignUp.value = '';
      emailInputSignUp.value = '';
      passwordInputSignUp.value = '';
      
      // Redirect to login page after successful sign-up
      window.location.href = 'templates/login.html'; // Change this to login page (if applicable)
    } else {
      alert(`Error: ${data.error || "Something went wrong!"}`);
    }
  } catch (error) {
    console.error("Error signing up:", error);
    alert("Error signing up. Please try again.");
  }
});

// Handle Sign-in Form Submission
signInForm.addEventListener("submit", async (e) => {
  e.preventDefault(); // Prevent page reload

  const username = usernameInputSignIn.value.trim();
  const password = passwordInputSignIn.value.trim();

  // Basic validation
  if (!username || !password) {
    alert("Please fill in all fields.");
    return;
  }

  // Make POST request to the login endpoint
  try {
    const response = await fetch('http://localhost:3000/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
    });

    const data = await response.json();
    if (response.status === 200) {
      alert("Login successful!");
      
      // Clear form fields
      usernameInputSignIn.value = '';
      passwordInputSignIn.value = '';
      
      // Redirect to jeet.html after successful login
      window.location.href = 'templates/jeet.html'; // Redirect to jeet.html (adjust if needed)
    } else {
      alert(`Error: ${data.error || "Invalid username or password"}`);
    }
  } catch (error) {
    console.error("Error logging in:", error);
    alert("Error logging in. Please try again.");
  }
});
