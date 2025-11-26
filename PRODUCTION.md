# Production Deployment Guide

This guide covers deploying the RAG application in a production environment with all security features enabled.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Security Features](#security-features)
- [Environment Configuration](#environment-configuration)
- [SSL/HTTPS Setup](#sslhttps-setup)
- [MongoDB Authentication](#mongodb-authentication)
- [Running with PM2](#running-with-pm2)
- [Monitoring and Logging](#monitoring-and-logging)
- [Production Checklist](#production-checklist)

## Prerequisites

- Node.js 18+ installed
- MongoDB with authentication enabled
- Docker and Docker Compose
- SSL/TLS certificates (Let's Encrypt recommended)
- Domain name (for production)
- Firewall configured

## Security Features

The application includes the following production-ready security features:

### 1. Rate Limiting

- **API Rate Limiting**: 100 requests per 15 minutes per IP
- **Auth Rate Limiting**: 5 failed login attempts per 15 minutes
- **Chat Rate Limiting**: 20 messages per minute per user

### 2. Security Headers (Helmet)

- Content Security Policy
- X-Frame-Options
- X-Content-Type-Options
- Strict-Transport-Security (HSTS)
- X-XSS-Protection

### 3. Input Sanitization

- MongoDB query injection prevention
- XSS protection
- Request payload size limits (10MB)

### 4. Authentication & Authorization

- JWT tokens with 7-day expiration
- Bcrypt password hashing (10 salt rounds)
- Secure token verification on all protected routes

### 5. Error Handling & Logging

- Winston logger with daily log rotation
- Separate error, combined, and exception logs
- Production mode hides sensitive error details
- Comprehensive error tracking

### 6. CORS Configuration

- Configurable allowed origins
- Credentials support
- Preflight request handling

## Environment Configuration

### Server Configuration

Create `/server/.env` file:

```bash
# MongoDB Configuration (with authentication)
MONGODB_URI=mongodb://rag_app_user:YOUR_SECURE_PASSWORD@localhost:27017/rag_application?authSource=rag_application

# Security
JWT_SECRET=8185842a80f436012330d7a4e09970efadb3315baf95686c1ab787883b4debfac7c0e3c63a077a429439c0b36b4f4c274e2e8b7e3b99ea560e5081fba9f20c7a

# Server Configuration
PORT=5000
NODE_ENV=production

# Service URLs
PYTHON_SERVICE_URL=http://localhost:8000

# CORS Configuration (add your production domain)
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Rate Limiting
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100

# SSL/HTTPS
SSL_KEY_PATH=/etc/ssl/private/private-key.pem
SSL_CERT_PATH=/etc/ssl/certs/certificate.pem
HTTPS_PORT=443
```

### Python Service Configuration

Create `/python-service/.env` file with your RAG service configuration:

```bash
PORT=8000
OPENAI_API_KEY=your_api_key_here
VECTOR_DB_URL=your_vector_db_url
MODEL_NAME=your_model_name
```

## SSL/HTTPS Setup

### Option 1: Let's Encrypt (Recommended for Production)

Install Certbot:

```bash
sudo apt-get update
sudo apt-get install certbot
```

Obtain certificates:

```bash
sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com
```

Certificates will be stored in `/etc/letsencrypt/live/yourdomain.com/`

Update your `.env`:

```bash
SSL_KEY_PATH=/etc/letsencrypt/live/yourdomain.com/privkey.pem
SSL_CERT_PATH=/etc/letsencrypt/live/yourdomain.com/fullchain.pem
```

Set up auto-renewal:

```bash
sudo crontab -e
# Add this line:
0 0 * * * certbot renew --quiet
```

### Option 2: Self-Signed Certificate (Development Only)

Generate self-signed certificate:

```bash
cd server/scripts
chmod +x generate-ssl-cert.sh
./generate-ssl-cert.sh
```

Update your `.env` with the generated certificate paths.

## MongoDB Authentication

Follow the [MongoDB Setup Guide](./MONGODB_SETUP.md) to:

1. Enable authentication
2. Create admin user
3. Create application user with minimal privileges
4. Update connection string in `.env`

Quick setup:

```javascript
// Connect to MongoDB
mongosh

// Create admin user
use admin
db.createUser({
  user: "admin",
  pwd: "STRONG_ADMIN_PASSWORD",
  roles: ["userAdminAnyDatabase", "readWriteAnyDatabase"]
})

// Create application user
use rag_application
db.createUser({
  user: "rag_app_user",
  pwd: "STRONG_APP_PASSWORD",
  roles: [{ role: "readWrite", db: "rag_application" }]
})
```

## Running with PM2

PM2 is a production process manager for Node.js applications.

### Install PM2

```bash
cd server
npm install
```

### Start Application

```bash
# Start in production mode
npm run prod

# Or manually
NODE_ENV=production pm2 start ecosystem.config.cjs
```

### PM2 Commands

```bash
# View logs
npm run prod:logs
# or
pm2 logs rag-backend

# Stop application
npm run prod:stop
# or
pm2 stop rag-backend

# Restart application
npm run prod:restart
# or
pm2 restart rag-backend

# Monitor
pm2 monit

# Show process info
pm2 show rag-backend

# Startup script (run on boot)
pm2 startup
pm2 save
```

## Monitoring and Logging

### Log Files

Logs are stored in `/server/logs/`:

- `error-YYYY-MM-DD.log` - Error logs only
- `combined-YYYY-MM-DD.log` - All logs
- `exceptions-YYYY-MM-DD.log` - Uncaught exceptions
- `rejections-YYYY-MM-DD.log` - Unhandled promise rejections
- `pm2-error.log` - PM2 error logs
- `pm2-out.log` - PM2 output logs

### Viewing Logs

```bash
# Tail error logs
tail -f logs/error-$(date +%Y-%m-%d).log

# Tail combined logs
tail -f logs/combined-$(date +%Y-%m-%d).log

# PM2 logs
pm2 logs rag-backend

# PM2 logs with filter
pm2 logs rag-backend --err
```

### Log Rotation

Logs automatically rotate daily and are kept for 14 days. Old logs are compressed to save space.

### Health Monitoring

Check application health:

```bash
curl http://localhost:5000/health
```

Response:

```json
{
  "status": "ok",
  "message": "Backend server is running",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "uptime": 12345,
  "mongodb": "connected"
}
```

## Production Checklist

### Security

- [ ] Strong JWT_SECRET generated and set
- [ ] MongoDB authentication enabled
- [ ] SSL/TLS certificates installed
- [ ] Firewall configured (allow only necessary ports)
- [ ] CORS origins restricted to production domains
- [ ] Rate limiting configured
- [ ] Environment variables secured
- [ ] Sensitive data not in version control

### Performance

- [ ] PM2 cluster mode enabled (multiple instances)
- [ ] Compression middleware enabled
- [ ] MongoDB indexes created
- [ ] Log rotation configured
- [ ] Connection pooling configured

### Reliability

- [ ] PM2 auto-restart configured
- [ ] PM2 startup script enabled
- [ ] Error logging configured
- [ ] Health check endpoint tested
- [ ] Backup strategy implemented
- [ ] Monitoring alerts set up

### Deployment

- [ ] All dependencies installed
- [ ] Environment variables configured
- [ ] MongoDB connection tested
- [ ] Python service running
- [ ] HTTPS working correctly
- [ ] Application accessible via domain
- [ ] All services restart on reboot

## Firewall Configuration

### UFW (Ubuntu)

```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP (for Let's Encrypt validation)
sudo ufw allow 80/tcp

# Allow HTTPS
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw enable
```

### iptables

```bash
# Allow SSH
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT

# Allow HTTPS
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Save rules
sudo iptables-save > /etc/iptables/rules.v4
```

## Nginx Reverse Proxy (Optional)

For additional security and performance, use Nginx as a reverse proxy:

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Updating the Application

```bash
# Pull latest changes
git pull origin main

# Update dependencies
cd server
npm install

# Restart with PM2
npm run prod:restart

# Check logs
npm run prod:logs
```

## Troubleshooting

### Application Won't Start

```bash
# Check PM2 logs
pm2 logs rag-backend --err

# Check MongoDB connection
mongosh -u rag_app_user -p PASSWORD rag_application

# Check port availability
sudo netstat -tlnp | grep :5000
```

### High Memory Usage

```bash
# Check PM2 status
pm2 status

# Restart application
pm2 restart rag-backend

# Adjust PM2 max memory restart
pm2 start ecosystem.config.cjs --max-memory-restart 1G
```

### SSL Certificate Issues

```bash
# Test certificate
openssl s_client -connect yourdomain.com:443

# Verify certificate files
ls -la /etc/letsencrypt/live/yourdomain.com/

# Check certificate expiry
openssl x509 -in /etc/letsencrypt/live/yourdomain.com/cert.pem -noout -dates
```

## Additional Resources

- [PM2 Documentation](https://pm2.keymetrics.io/docs/usage/quick-start/)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
- [MongoDB Production Notes](https://docs.mongodb.com/manual/administration/production-notes/)
- [Express Security Best Practices](https://expressjs.com/en/advanced/best-practice-security.html)
- [Node.js Security Checklist](https://github.com/goldbergyoni/nodebestpractices#6-security-best-practices)
