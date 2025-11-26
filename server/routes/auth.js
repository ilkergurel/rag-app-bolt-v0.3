import express from 'express';
import jwt from 'jsonwebtoken';
import User from '../models/User.js';
import logger from '../utils/logger.js';
import { authLimiter } from '../middleware/rateLimiter.js';

const router = express.Router();

router.post('/register', authLimiter, async (req, res) => {
  try {
    const { username, password } = req.body;

    if (!username || !password) {
      logger.warn('Registration attempt with missing credentials', { ip: req.ip });
      return res.status(400).json({ message: 'Username and password are required' });
    }

    if (username.length < 3) {
      return res.status(400).json({ message: 'Username must be at least 3 characters long' });
    }

    if (password.length < 6) {
      return res.status(400).json({ message: 'Password must be at least 6 characters long' });
    }

    const existingUser = await User.findOne({ username });
    if (existingUser) {
      logger.warn('Registration attempt with existing username', { username, ip: req.ip });
      return res.status(400).json({ message: 'Username already exists' });
    }

    const user = new User({ username, password });
    await user.save();

    const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET, { expiresIn: '7d' });

    logger.info('User registered successfully', { userId: user._id, username });

    res.status(201).json({
      message: 'User created successfully',
      token,
      userId: user._id,
      username: user.username
    });
  } catch (error) {
    logger.error('Registration error:', { error: error.message, stack: error.stack });
    res.status(500).json({
      message: process.env.NODE_ENV === 'production'
        ? 'Error creating user'
        : error.message
    });
  }
});

router.post('/login', authLimiter, async (req, res) => {
  try {
    const { username, password } = req.body;

    if (!username || !password) {
      logger.warn('Login attempt with missing credentials', { ip: req.ip });
      return res.status(400).json({ message: 'Username and password are required' });
    }

    const user = await User.findOne({ username });
    if (!user) {
      logger.warn('Login attempt with non-existent username', { username, ip: req.ip });
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    const isPasswordValid = await user.comparePassword(password);
    if (!isPasswordValid) {
      logger.warn('Login attempt with invalid password', { username, ip: req.ip });
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET, { expiresIn: '7d' });

    logger.info('User logged in successfully', { userId: user._id, username });

    res.json({
      message: 'Login successful',
      token,
      userId: user._id,
      username: user.username
    });
  } catch (error) {
    logger.error('Login error:', { error: error.message, stack: error.stack });
    res.status(500).json({
      message: process.env.NODE_ENV === 'production'
        ? 'Error logging in'
        : error.message
    });
  }
});

export default router;
