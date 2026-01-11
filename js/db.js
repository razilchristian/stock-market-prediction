const express = require('express');
const mysql = require('mysql2/promise');
const bcrypt = require('bcrypt');
const cors = require('cors');
const session = require('express-session');
const fs = require('fs');
const csv = require('csv-parser');
const yahooFinance = require('yahoo-finance2').default; // Add yfinance
require('dotenv').config();

const app = express();

// Middleware to parse JSON
app.use(express.json());

// CORS setup to allow requests from frontend
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:5000', 'http://localhost:5500', 'http://127.0.0.1:5500'],
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type'],
  credentials: true  // IMPORTANT: Allow cookies/session
}));

// Session setup
app.use(session({
  secret: process.env.SESSION_SECRET || 'alpha-analytics-secret-key-2026',
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: process.env.NODE_ENV === 'production',
    httpOnly: true,
    maxAge: 24 * 60 * 60 * 1000 // 24 hours
  }
}));

// Database connection
const dbConfig = {
  host: process.env.DB_HOST || 'localhost',
  user: process.env.DB_USER || 'root',
  password: process.env.DB_PASSWORD || '',
  database: process.env.DB_NAME || 'stockmarketprediction',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
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
      username VARCHAR(50) NOT NULL UNIQUE,
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
      close_forecast DECIMAL(10,4) NOT NULL,
      INDEX idx_symbol (symbol),
      INDEX idx_date (date)
    );
  `;
  await executeQuery(query);
  console.log('Stock predictions table ready');
};

// Create user watchlist table
const createWatchlistTable = async () => {
  const query = `
    CREATE TABLE IF NOT EXISTS user_watchlist (
      id INT PRIMARY KEY AUTO_INCREMENT,
      user_id INT NOT NULL,
      symbol VARCHAR(20) NOT NULL,
      added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
      UNIQUE KEY unique_user_symbol (user_id, symbol)
    );
  `;
  await executeQuery(query);
  console.log('Watchlist table ready');
};

// Check if stock data already exists
const checkStockDataExists = async () => {
  try {
    const query = `SELECT COUNT(*) as count FROM stock_predictions LIMIT 1`;
    const [result] = await executeQuery(query);
    return result[0].count > 0;
  } catch (error) {
    return false;
  }
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
    // Check if file exists
    if (!fs.existsSync(csvFilePath)) {
      console.log('CSV file not found, skipping data load');
      return resolve();
    }
    
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

// ================== YFINANCE API ENDPOINTS ==================

// Get stock quotes from yfinance
app.get('/api/yfinance/quote/:symbol', async (req, res) => {
  const { symbol } = req.params;
  try {
    const quote = await yahooFinance.quote(symbol);
    res.json({
      symbol: quote.symbol,
      name: quote.longName || quote.shortName || symbol,
      price: quote.regularMarketPrice || quote.price || 0,
      change: quote.regularMarketChange || quote.change || 0,
      changePercent: quote.regularMarketChangePercent || 
                    (quote.regularMarketChange ? (quote.regularMarketChange / (quote.regularMarketPrice - quote.regularMarketChange) * 100) : 0),
      currency: quote.currency || 'USD',
      marketState: quote.marketState || 'REGULAR',
      volume: quote.regularMarketVolume,
      marketCap: quote.marketCap
    });
  } catch (error) {
    console.error('yfinance error for', symbol, ':', error.message);
    res.status(500).json({ 
      error: 'Failed to fetch stock data',
      symbol: symbol,
      message: error.message 
    });
  }
});

// Get multiple stock quotes
app.get('/api/yfinance/quotes', async (req, res) => {
  const { symbols } = req.query;
  if (!symbols) {
    return res.status(400).json({ error: 'Symbols parameter required' });
  }
  
  try {
    const symbolArray = symbols.split(',');
    const quotes = await Promise.all(
      symbolArray.map(symbol => 
        yahooFinance.quote(symbol)
          .then(quote => ({
            symbol: quote.symbol,
            name: quote.longName || quote.shortName || symbol,
            price: quote.regularMarketPrice || quote.price || 0,
            change: quote.regularMarketChange || quote.change || 0,
            changePercent: quote.regularMarketChangePercent || 
                          (quote.regularMarketChange ? (quote.regularMarketChange / (quote.regularMarketPrice - quote.regularMarketChange) * 100) : 0),
            currency: quote.currency || 'USD',
            marketState: quote.marketState || 'REGULAR'
          }))
          .catch(e => ({ 
            symbol: symbol, 
            error: e.message,
            name: symbol,
            price: 0,
            change: 0,
            changePercent: 0,
            currency: 'USD',
            marketState: 'CLOSED'
          }))
      )
    );
    
    // Filter out only valid quotes for the response
    const validQuotes = quotes.filter(q => !q.error || q.price > 0);
    res.json(validQuotes);
    
  } catch (error) {
    console.error('yfinance batch error:', error.message);
    res.status(500).json({ error: 'Failed to fetch stock data', message: error.message });
  }
});

// Get historical data
app.get('/api/yfinance/historical/:symbol', async (req, res) => {
  const { symbol } = req.params;
  const { period = '1mo', interval = '1d' } = req.query;
  
  try {
    const queryOptions = { period, interval };
    
    // Handle different period formats
    if (period.includes('d') || period.includes('mo') || period.includes('y')) {
      queryOptions.period = period;
    } else {
      queryOptions.period = '1mo'; // default
    }
    
    const historical = await yahooFinance.historical(symbol, queryOptions);
    
    // Format the response
    const formattedData = historical.map(item => ({
      date: item.date,
      open: item.open,
      high: item.high,
      low: item.low,
      close: item.close,
      volume: item.volume,
      adjClose: item.adjClose
    }));
    
    res.json(formattedData);
  } catch (error) {
    console.error('yfinance historical error for', symbol, ':', error.message);
    res.status(500).json({ 
      error: 'Failed to fetch historical data',
      symbol: symbol,
      message: error.message 
    });
  }
});

// Search stocks by query
app.get('/api/yfinance/search', async (req, res) => {
  const { q } = req.query;
  
  if (!q || q.length < 2) {
    return res.status(400).json({ error: 'Search query must be at least 2 characters' });
  }
  
  try {
    const searchResults = await yahooFinance.search(q);
    res.json(searchResults.quotes || []);
  } catch (error) {
    console.error('yfinance search error:', error.message);
    res.status(500).json({ error: 'Failed to search stocks', message: error.message });
  }
});

// Get market indices
app.get('/api/yfinance/indices', async (req, res) => {
  const indices = [
    { symbol: '^NSEI', name: 'NIFTY 50', type: 'INDIAN' },
    { symbol: '^BSESN', name: 'SENSEX', type: 'INDIAN' },
    { symbol: '^IXIC', name: 'NASDAQ', type: 'US' },
    { symbol: '^GSPC', name: 'S&P 500', type: 'US' },
    { symbol: '^DJI', name: 'Dow Jones', type: 'US' },
    { symbol: '^FTSE', name: 'FTSE 100', type: 'UK' },
    { symbol: '^N225', name: 'Nikkei 225', type: 'JAPAN' }
  ];
  
  try {
    const quotes = await Promise.all(
      indices.map(async (index) => {
        try {
          const quote = await yahooFinance.quote(index.symbol);
          return {
            symbol: index.symbol,
            name: index.name,
            type: index.type,
            price: quote.regularMarketPrice || 0,
            change: quote.regularMarketChange || 0,
            changePercent: quote.regularMarketChangePercent || 0,
            currency: quote.currency || 'USD'
          };
        } catch (error) {
          console.error(`Failed to fetch ${index.symbol}:`, error.message);
          return {
            symbol: index.symbol,
            name: index.name,
            type: index.type,
            price: 0,
            change: 0,
            changePercent: 0,
            currency: 'USD',
            error: error.message
          };
        }
      })
    );
    
    res.json(quotes);
  } catch (error) {
    console.error('Market indices error:', error.message);
    res.status(500).json({ error: 'Failed to fetch market indices', message: error.message });
  }
});

// ================== AUTH & USER ENDPOINTS ==================

// Sign-up route
app.post('/signup', async (req, res) => {
  const { username, email, password } = req.body;

  if (!username || !email || !password) {
    return res.status(400).json({ error: 'All fields are required' });
  }

  try {
    const existingUser = await getUserByUsername(username);
    if (existingUser) return res.status(400).json({ error: 'Username already exists' });

    const existingEmail = await getUserByEmail(email);
    if (existingEmail) return res.status(400).json({ error: 'Email already in use' });

    await insertUser(username, email, password);
    res.status(201).json({ message: 'User created successfully' });
  } catch (error) {
    console.error('Signup error:', error);
    res.status(500).json({ error: 'Error creating user' });
  }
});

// Login route
app.post('/login', async (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({ error: 'Username and password are required' });
  }

  try {
    const user = await getUserByUsername(username);
    if (!user) return res.status(401).json({ error: 'Invalid username or password' });

    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) return res.status(401).json({ error: 'Invalid username or password' });

    // Store user info in session
    req.session.user = { 
      id: user.id, 
      username: user.username,
      email: user.email,
      role: user.role
    };
    
    console.log('User logged in:', req.session.user);
    res.status(200).json({ 
      message: 'Login successful',
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        role: user.role
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Error logging in' });
  }
});

// Logout route
app.post('/logout', (req, res) => {
  req.session.destroy((err) => {
    if (err) {
      console.error('Logout error:', err);
      return res.status(500).json({ error: 'Error logging out' });
    }
    res.status(200).json({ message: 'Logged out successfully' });
  });
});

// Check authentication status
app.get('/api/auth/status', (req, res) => {
  if (req.session.user) {
    res.json({ 
      authenticated: true, 
      user: req.session.user 
    });
  } else {
    res.json({ authenticated: false });
  }
});

// ================== STOCK PREDICTION ENDPOINTS ==================

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

// Get all predicted stocks
app.get('/stocks', async (req, res) => {
  const query = `SELECT DISTINCT symbol FROM stock_predictions ORDER BY symbol;`;
  
  try {
    const results = await executeQuery(query);
    res.status(200).json(results);
  } catch (error) {
    res.status(500).json({ error: 'Error fetching stock symbols' });
  }
});

// ================== WATCHLIST ENDPOINTS ==================

// Get user watchlist
app.get('/api/watchlist', async (req, res) => {
  if (!req.session.user) {
    return res.status(401).json({ error: 'Not logged in' });
  }
  
  try {
    const query = `SELECT symbol FROM user_watchlist WHERE user_id = ? ORDER BY added_at DESC;`;
    const results = await executeQuery(query, [req.session.user.id]);
    res.status(200).json(results.map(row => row.symbol));
  } catch (error) {
    console.error('Watchlist error:', error);
    res.status(500).json({ error: 'Error fetching watchlist' });
  }
});

// Add to watchlist
app.post('/api/watchlist/add', async (req, res) => {
  if (!req.session.user) {
    return res.status(401).json({ error: 'Not logged in' });
  }
  
  const { symbol } = req.body;
  
  if (!symbol) {
    return res.status(400).json({ error: 'Symbol is required' });
  }
  
  try {
    const query = `INSERT INTO user_watchlist (user_id, symbol) VALUES (?, ?);`;
    await executeQuery(query, [req.session.user.id, symbol]);
    res.status(201).json({ message: 'Added to watchlist' });
  } catch (error) {
    if (error.code === 'ER_DUP_ENTRY') {
      return res.status(400).json({ error: 'Already in watchlist' });
    }
    console.error('Add to watchlist error:', error);
    res.status(500).json({ error: 'Error adding to watchlist' });
  }
});

// Remove from watchlist
app.delete('/api/watchlist/remove/:symbol', async (req, res) => {
  if (!req.session.user) {
    return res.status(401).json({ error: 'Not logged in' });
  }
  
  const { symbol } = req.params;
  
  try {
    const query = `DELETE FROM user_watchlist WHERE user_id = ? AND symbol = ?;`;
    const result = await executeQuery(query, [req.session.user.id, symbol]);
    
    if (result.affectedRows === 0) {
      return res.status(404).json({ error: 'Symbol not found in watchlist' });
    }
    
    res.status(200).json({ message: 'Removed from watchlist' });
  } catch (error) {
    console.error('Remove from watchlist error:', error);
    res.status(500).json({ error: 'Error removing from watchlist' });
  }
});

// ================== USER PROFILE ENDPOINTS ==================

// Get current user info
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

// Update user profile
app.put('/api/user/profile', async (req, res) => {
  if (!req.session.user) {
    return res.status(401).json({ error: 'Not logged in' });
  }
  
  const { email, currentPassword, newPassword } = req.body;
  
  try {
    if (email) {
      // Check if email already exists
      const existingEmail = await getUserByEmail(email);
      if (existingEmail && existingEmail.id !== req.session.user.id) {
        return res.status(400).json({ error: 'Email already in use' });
      }
      
      const query = `UPDATE users SET email = ? WHERE id = ?`;
      await executeQuery(query, [email, req.session.user.id]);
      req.session.user.email = email;
    }
    
    if (newPassword) {
      if (!currentPassword) {
        return res.status(400).json({ error: 'Current password is required to change password' });
      }
      
      // Verify current password
      const user = await getUserByUsername(req.session.user.username);
      const isPasswordValid = await bcrypt.compare(currentPassword, user.password);
      if (!isPasswordValid) {
        return res.status(401).json({ error: 'Current password is incorrect' });
      }
      
      // Update password
      const hashedPassword = await bcrypt.hash(newPassword, 10);
      const query = `UPDATE users SET password = ? WHERE id = ?`;
      await executeQuery(query, [hashedPassword, req.session.user.id]);
    }
    
    res.status(200).json({ message: 'Profile updated successfully' });
  } catch (error) {
    console.error('Update profile error:', error);
    res.status(500).json({ error: 'Error updating profile' });
  }
});

// ================== HEALTH CHECK ==================

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({ 
    status: 'OK', 
    timestamp: new Date().toISOString(),
    service: 'AlphaAnalytics API'
  });
});

// ================== INITIALIZE DATABASE ==================

const initDatabase = async () => {
  try {
    await createUserTable();
    await createStockPredictionsTable();
    await createWatchlistTable();
    
    // Only load CSV data if table is empty
    const hasData = await checkStockDataExists();
    if (!hasData) {
      console.log('Loading stock data from CSV...');
      await loadStockData('alphaanalytics.csv');
    } else {
      console.log('Stock data already exists, skipping CSV load');
    }
    
    console.log('Database initialization complete!');
  } catch (error) {
    console.error('Error initializing database:', error.message);
  }
};

// ================== START SERVER ==================

const startServer = async () => {
  try {
    await initDatabase();
    
    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
      console.log(`🚀 Server running on http://localhost:${PORT}`);
      console.log(`📊 API Endpoints:`);
      console.log(`  POST /signup        - User registration`);
      console.log(`  POST /login         - User login`);
      console.log(`  POST /logout        - User logout`);
      console.log(`  GET  /me            - Current user info`);
      console.log(`  GET  /stocks/:symbol - Stock predictions`);
      console.log(`  GET  /api/yfinance/quote/:symbol - Live stock data`);
      console.log(`  GET  /api/yfinance/quotes - Multiple stock quotes`);
      console.log(`  GET  /api/yfinance/historical/:symbol - Historical data`);
      console.log(`  GET  /api/yfinance/indices - Market indices`);
      console.log(`  GET  /api/yfinance/search - Stock search`);
      console.log(`  GET  /api/watchlist  - User watchlist`);
      console.log(`  POST /api/watchlist/add - Add to watchlist`);
      console.log(`  GET  /health        - Health check`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
};

startServer();