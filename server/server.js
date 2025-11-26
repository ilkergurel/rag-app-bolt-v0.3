import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";
import helmet from "helmet";
import mongoSanitize from "express-mongo-sanitize";
import compression from "compression";
import https from "https";
import http from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import authRoutes from "./routes/auth.js";
import chatRoutes from "./routes/chat.js";
import logger from "./utils/logger.js";
import { apiLimiter } from "./middleware/rateLimiter.js";

// --- Determine the directory of the current file ---
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- Configure dotenv to use the .env file in the SAME directory as this script ---
dotenv.config({ path: path.join(__dirname, ".env") });

const app = express();
const PORT = process.env.PORT || 5000;
const HTTPS_PORT = process.env.HTTPS_PORT || 443;

const allowedOrigins = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(",")
  : ["http://localhost:5173", "http://localhost:3000"];

const corsOptions = {
  origin: (origin, callback) => {
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      logger.warn(`CORS blocked request from origin: ${origin}`);
      callback(new Error("Not allowed by CORS"));
    }
  },
  credentials: true,
  optionsSuccessStatus: 200,
  methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
};

app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        styleSrc: ["'self'", "'unsafe-inline'"],
        scriptSrc: ["'self'"],
        imgSrc: ["'self'", "data:", "https:"],
      },
    },
    hsts: {
      maxAge: 31536000,
      includeSubDomains: true,
      preload: true,
    },
  })
);

app.use(cors(corsOptions));
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));
app.use(mongoSanitize());
//app.use(compression());

app.use((req, res, next) => {
  logger.info(`${req.method} ${req.path}`, {
    ip: req.ip,
    userAgent: req.get("user-agent"),
  });
  next();
});

app.use("/api", apiLimiter);

const mongooseOptions = {
  serverSelectionTimeoutMS: 5000,
  socketTimeoutMS: 45000,
};

mongoose
  .connect(process.env.MONGODB_URI, mongooseOptions)
  .then(() => {
    logger.info("Connected to MongoDB successfully");
  })
  .catch((err) => {
    logger.error("MongoDB connection error:", err);
    process.exit(1);
  });

mongoose.connection.on("disconnected", () => {
  logger.warn("MongoDB disconnected. Attempting to reconnect...");
});

mongoose.connection.on("reconnected", () => {
  logger.info("MongoDB reconnected successfully");
});

app.use("/api/auth", authRoutes);
app.use("/api/chats", chatRoutes);

app.get("/health", (req, res) => {
  const health = {
    status: "ok",
    message: "Backend server is running",
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    mongodb:
      mongoose.connection.readyState === 1 ? "connected" : "disconnected",
  };
  res.json(health);
});

app.use((err, req, res, next) => {
  logger.error("Unhandled error:", {
    error: err.message,
    stack: err.stack,
    path: req.path,
    method: req.method,
  });

  res.status(err.status || 500).json({
    message:
      process.env.NODE_ENV === "production"
        ? "Internal server error"
        : err.message,
    ...(process.env.NODE_ENV !== "production" && { stack: err.stack }),
  });
});

app.use((req, res) => {
  res.status(404).json({ message: "Route not found" });
});

if (process.env.SSL_KEY_PATH && process.env.SSL_CERT_PATH) {
  try {
    const privateKey = fs.readFileSync(process.env.SSL_KEY_PATH, "utf8");
    const certificate = fs.readFileSync(process.env.SSL_CERT_PATH, "utf8");
    const credentials = { key: privateKey, cert: certificate };

    const httpsServer = https.createServer(credentials, app);
    httpsServer.listen(HTTPS_PORT, () => {
      logger.info(`HTTPS Server running on port ${HTTPS_PORT}`);
      if (process.send) {
        process.send("ready");
      }
    });

    const httpApp = express();
    httpApp.use((req, res) => {
      res.redirect(`https://${req.headers.host}${req.url}`);
    });
    http.createServer(httpApp).listen(PORT, () => {
      logger.info(`HTTP Server redirecting to HTTPS on port ${PORT}`);
    });
  } catch (error) {
    logger.error("Failed to start HTTPS server:", error);
    startHttpServer();
  }
} else {
  startHttpServer();
}

function startHttpServer() {
  const server = app.listen(PORT, () => {
    logger.info(`HTTP Server running on port ${PORT}`);
    if (process.send) {
      process.send("ready");
    }
  });

  server.on("error", (error) => {
    logger.error("Server error:", error);
    process.exit(1);
  });
}

process.on("unhandledRejection", (reason, promise) => {
  logger.error("Unhandled Rejection at:", promise, "reason:", reason);
});

process.on("uncaughtException", (error) => {
  logger.error("Uncaught Exception:", error);
  process.exit(1);
});

process.on("SIGTERM", () => {
  logger.info("SIGTERM signal received: closing HTTP server");
  mongoose.connection.close(false);
  process.exit(0);
});

process.on("SIGINT", () => {
  logger.info("SIGINT signal received: closing HTTP server");
  mongoose.connection.close(false);
  process.exit(0);
});
