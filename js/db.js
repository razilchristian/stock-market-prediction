const express = require('express');
const mysql = require('mysql2/promise');
const bcrypt = require('bcrypt');
const cors = require('cors');
const session = require('express-session');
const fs = require('fs');
const csv = require('csv-parser');

const app = express();

// Middleware to parse JSON
app.use(express.json());

// CORS setup to allow requests from localhost
app.use(cors({
  origin: ['http://localhost', 'http://localhost:5000'],  
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type'],
}));

// Session setup
app.use(session({
  secret: 'your-secret-key', 
  resave: false,
  saveUninitialized: true,
}));

// Database connection
const dbConfig = {
  host: 'localhost',
  user: 'root',
  password: '',
  database: 'stockmarketprediction',
};

const pool = mysql.createPool(dbConfig);

// Helper function to execute queries
const executeQuery = async (query, params = []) => {
  try {
    const [results] = await pool.execute(query, params);
    console.log('Executed query:', query, 'With params:', params);
    return results;
  } catch (error) {
    console.error('Database query error:', error.message);
    throw error;
  }
};

// Create users table
const createUserTable = async () => {
  const query = `
    CREATE TABLE IF NOT EXISTS users (
      id INT PRIMARY KEY AUTO_INCREMENT,
      username VARCHAR(50) NOT NULL,
      email VARCHAR(100) UNIQUE NOT NULL,
      password VARCHAR(255) NOT NULL,
      role VARCHAR(20) DEFAULT 'user',
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
  `;
  await executeQuery(query);
  console.log('Users table ready');
};

// Create stock predictions table
const createStockPredictionsTable = async () => {
  const query = `
    CREATE TABLE IF NOT EXISTS stock_predictions (
      id INT PRIMARY KEY AUTO_INCREMENT,
      date DATE NOT NULL,
      symbol VARCHAR(10) NOT NULL,
      open_forecast DECIMAL(10,4) NOT NULL,
      close_forecast DECIMAL(10,4) NOT NULL
    );
  `;
  await executeQuery(query);
  console.log('Stock predictions table ready');
};

// Insert a new user
const insertUser = async (username, email, password) => {
  const hashedPassword = await bcrypt.hash(password, 10);
  const query = `INSERT INTO users (username, email, password) VALUES (?, ?, ?);`;
  await executeQuery(query, [username, email, hashedPassword]);
  console.log('User created successfully');
};

// Get user by username
const getUserByUsername = async (username) => {
  const query = `SELECT * FROM users WHERE username = ?;`;
  const results = await executeQuery(query, [username]);
  return results[0];
};

// Get user by email
const getUserByEmail = async (email) => {
  const query = `SELECT * FROM users WHERE email = ?;`;
  const results = await executeQuery(query, [email]);
  return results[0];
};

// Insert stock prediction data
const insertStockPrediction = async (date, symbol, openForecast, closeForecast) => {
  const query = `
    INSERT INTO stock_predictions (date, symbol, open_forecast, close_forecast)
    VALUES (?, ?, ?, ?);
  `;
  await executeQuery(query, [date, symbol, parseFloat(openForecast), parseFloat(closeForecast)]);
};

// Load stock data from CSV
const loadStockData = async (csvFilePath) => {
  return new Promise((resolve, reject) => {
    fs.createReadStream(csvFilePath)
      .pipe(csv())
      .on('data', async (row) => {
        try {
          const { Date, Symbol, 'Open Forecast': openForecast, 'Close Forecast': closeForecast } = row;
          await insertStockPrediction(Date, Symbol, openForecast, closeForecast);
        } catch (error) {
          console.error('Error inserting row:', error.message);
        }
      })
      .on('end', () => {
        console.log('CSV file processed successfully');
        resolve();
      })
      .on('error', (error) => {
        console.error('CSV processing error:', error);
        reject(error);
      });
  });
};

// Sign-up route
app.post('/signup', async (req, res) => {
  const { username, email, password } = req.body;

  if (!username || !email || !password) {
    return res.status(400).send({ error: 'All fields are required' });
  }

  try {
    const existingUser = await getUserByUsername(username);
    if (existingUser) return res.status(400).send({ error: 'Username already exists' });

    const existingEmail = await getUserByEmail(email);
    if (existingEmail) return res.status(400).send({ error: 'Email already in use' });

    await insertUser(username, email, password);
    res.status(201).send({ message: 'User created successfully' });
  } catch (error) {
    res.status(500).send({ error: 'Error creating user' });
  }
});

// Login route
app.post('/login', async (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).send({ error: 'Username and password are required' });
  }

  try {
    const user = await getUserByUsername(username);
    if (!user) return res.status(401).send({ error: 'Invalid username or password' });

    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) return res.status(401).send({ error: 'Invalid username or password' });

    req.session.user = { id: user.id, username: user.username };
    res.status(200).send({ message: 'Login successful' });
  } catch (error) {
    res.status(500).send({ error: 'Error logging in' });
  }
});

// Logout route
app.post('/logout', (req, res) => {
  req.session.destroy((err) => {
    if (err) return res.status(500).send({ error: 'Error logging out' });
    res.status(200).send({ message: 'Logged out successfully' });
  });
});

// Fetch stock predictions
app.get('/stocks/:symbol', async (req, res) => {
  const { symbol } = req.params;
  const query = `SELECT * FROM stock_predictions WHERE symbol = ? ORDER BY date ASC;`;
  
  try {
    const results = await executeQuery(query, [symbol]);
    res.status(200).json(results);
  } catch (error) {
    res.status(500).json({ error: 'Error fetching stock data' });
  }
});

// ... your existing top-of-file code stays

// Add this new API for current user info (no removal of old code)
app.get('/me', async (req, res) => {
  if (!req.session.user) {
    return res.status(401).json({ error: 'Not logged in' });
  }
  try {
    const query = `SELECT username, email, role FROM users WHERE id = ?`;
    const [rows] = await pool.execute(query, [req.session.user.id]);
    if (!rows.length) return res.status(404).json({ error: 'User not found' });
    res.json(rows[0]);
  } catch (err) {
    console.error('Error fetching profile:', err.message);
    res.status(500).json({ error: 'Server error' });
  }
});

// Initialize database
const initDatabase = async () => {
  try {
    await createUserTable();
    await createStockPredictionsTable();
    await loadStockData('alphaanalytics.csv');  // Change to your actual CSV file
    console.log('Database initialization complete!');
  } catch (error) {
    console.error('Error initializing database:', error.message);
  }
};

// Start server
(async () => {
  await initDatabase();
})();

app.listen(3000, () => {
  console.log('Server running on http://localhost:3000');
});
