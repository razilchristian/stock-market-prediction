// Select elements
const signInForm = document.querySelector(".sign-in-form");
const usernameInputSignIn = signInForm.querySelector('#username');
const passwordInputSignIn = signInForm.querySelector('#password');

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
      window.location.href = 'jeet.html'; // Redirect to jeet.html (adjust if needed)
    } else {
      alert(`Error: ${data.error || "Invalid username or password"}`);
    }
  } catch (error) {
    console.error("Error logging in:", error);
    alert("Error logging in. Please try again.");
  }
});
