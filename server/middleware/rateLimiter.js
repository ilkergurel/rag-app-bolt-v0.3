import rateLimit from 'express-rate-limit';
import logger from '../utils/logger.js';

export const apiLimiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000,
  max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100,
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
  handler: (req, res) => {
    logger.warn(`Rate limit exceeded for IP: ${req.ip}`);
    res.status(429).json({
      message: 'Too many requests from this IP, please try again later.'
    });
  }
});

export const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5,
  skipSuccessfulRequests: true,
  message: 'Too many authentication attempts, please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
  handler: (req, res) => {
    logger.warn(`Auth rate limit exceeded for IP: ${req.ip}`);
    res.status(429).json({
      message: 'Too many authentication attempts, please try again later.'
    });
  }
});

export const chatLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 20,
  message: 'Too many messages sent, please slow down.',
  standardHeaders: true,
  legacyHeaders: false,
  handler: (req, res) => {
    logger.warn(`Chat rate limit exceeded for user: ${req.userId}, IP: ${req.ip}`);
    res.status(429).json({
      message: 'Too many messages sent, please slow down.'
    });
  }
});
