// server.js
import express from 'express';
import mysql from 'mysql2/promise';
import bcrypt from 'bcrypt';
import db from './db.js';
const app = express();
app.use(express.json());
import { 
  createMarketTable, 
  insertSampleMarket, 
  insertUser, 
  getUserByUsername, 
  initDatabase, 
  query 
} from './db.js';

app.post('/api/signup', async (req, res) => {
  const { username, email, password } = req.body;
  try {
    await insertUser(username, email, password);
    res.json({ success: true, message: 'User created successfully' });
  } catch (err) {
    res.json({ success: false, message: err.message });
  }
});

app.post('/api/signin', async (req, res) => {
  const { username, password } = req.body;
  try {
    const user = await getUserByUsername(username);
    if (!user) {
      res.json({ success: false, message: 'Invalid credentials' });
    } else {
      const isValidPassword = await bcrypt.compare(password, user.password);
      if (!isValidPassword) {
        res.json({ success: false, message: 'Invalid credentials' });
      } else {
        res.json({ success: true, user });
      }
    }
  } catch (err) {
    res.json({ success: false, message: err.message });
  }
});
