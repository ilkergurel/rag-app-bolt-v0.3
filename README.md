# RAG Application

A full-stack RAG (Retrieval-Augmented Generation) chat application with streaming responses, built with React, Node.js, Python, and MongoDB.

## Architecture

This application consists of three main services:

1. **React Frontend** (`client/`) - User interface similar to ChatGPT/Gemini (JavaScript only, no TypeScript)
2. **Node.js Backend** (`server/`) - Express server with MongoDB/Mongoose for authentication and chat history
3. **Python RAG Service** (`python-service/`) - FastAPI service running in Docker for RAG processing and streaming responses

## Features

- User authentication (register/login) with MongoDB
- Multi-language support (English and Turkish)
- Chat history management (create, view, delete chats)
- Real-time streaming responses from RAG service
- Clean, modern UI with sidebar navigation
- Responsive design

## Prerequisites

- Node.js (v18 or higher)
- MongoDB (running locally or remote)
- Docker and Docker Compose
- Python 3.11+ (for local development of Python service)

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd project
```

### 2. Set up MongoDB

Make sure MongoDB is running on your local machine or update the connection string in the server `.env` file.

```bash
# Start MongoDB (if using local installation)
mongod
```

### 3. Install all dependencies

```bash
# Install dependencies for both client and server
npm run install:all
```

Or manually:

```bash
# Install client dependencies
cd client
npm install

# Install server dependencies
cd ../server
npm install
```

### 4. Configure Environment Variables

#### Server (Node.js Backend)

```bash
cd server
cp .env.example .env

# Edit .env and update values:
# MONGODB_URI=mongodb://localhost:27017/rag_application
# JWT_SECRET=your_secret_key_here
# PORT=5000
# PYTHON_SERVICE_URL=http://localhost:8000
```

#### Python Service

```bash
cd python-service

# Create .env file (optional, for your RAG configuration)
cp .env.example .env

# Add your RAG-specific environment variables
# OPENAI_API_KEY=your_key
# VECTOR_DB_URL=your_db_url
# etc.
```

## Running the Application

### Option 1: Run all services individually

#### 1. Start Python RAG Service (Docker)

```bash
# From project root directory
docker-compose up -d

# Or to see logs:
docker-compose up
```

The Python service will be available at `http://localhost:8000`

Check if it's running:
```bash
curl http://localhost:8000/health
```

#### 2. Start Backend Server

```bash
cd server
npm start

# For development with auto-reload:
npm run dev
```

The backend will run on `http://localhost:5000`

#### 3. Start Frontend

```bash
cd client
npm run dev
```

The frontend will run on `http://localhost:5173`

### Option 2: Use npm scripts from root

```bash
# Start Python service
npm run dev:python

# In another terminal, start server
npm run dev:server

# In another terminal, start client
npm run dev:client
```

## Project Structure

```
project/
├── client/                     # React frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── Auth.jsx       # Login/Register
│   │   │   ├── Chat.jsx       # Main chat container
│   │   │   ├── ChatArea.jsx   # Chat messages area
│   │   │   └── Sidebar.jsx    # Chat history sidebar
│   │   ├── context/           # React context providers
│   │   │   ├── AuthContext.jsx     # Authentication state
│   │   │   └── LanguageContext.jsx # i18n state
│   │   ├── services/          # API services
│   │   │   └── api.js         # Backend API calls
│   │   ├── App.jsx            # Main app component
│   │   ├── main.jsx           # App entry point
│   │   └── index.css          # Global styles
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
├── server/                     # Node.js Express backend
│   ├── models/                # MongoDB models
│   │   ├── User.js           # User authentication model
│   │   └── Chat.js           # Chat history model
│   ├── routes/               # API routes
│   │   ├── auth.js           # Authentication routes
│   │   └── chat.js           # Chat management routes
│   ├── middleware/           # Express middleware
│   │   └── auth.js           # JWT authentication middleware
│   ├── server.js             # Main server file
│   ├── package.json
│   └── .env.example
│
├── python-service/            # Python RAG service
│   ├── app.py                # FastAPI application
│   ├── requirements.txt      # Python dependencies
│   ├── Dockerfile            # Docker configuration
│   └── .env.example
│
├── docker-compose.yml         # Docker Compose configuration
├── package.json              # Root package.json with helper scripts
└── README.md
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user

### Chat Management
- `GET /api/chats` - Get all chats for logged-in user
- `GET /api/chats/:chatId` - Get specific chat with messages
- `POST /api/chats` - Create new chat
- `POST /api/chats/:chatId/message` - Send message (streaming response)
- `DELETE /api/chats/:chatId` - Delete chat

### Python Service
- `POST /query` - Send RAG query (streaming response)
- `GET /health` - Health check

## Implementing Your RAG Service

The Python service (`python-service/app.py`) currently has a placeholder implementation. To integrate your RAG solution:

1. Add your dependencies to `requirements.txt`
2. Replace the `generate_rag_response()` function with your RAG logic:
   - Vector database queries
   - Embedding generation
   - LLM integration
   - Response streaming

Example structure:
```python
async def generate_rag_response(query: str):
    # 1. Generate embeddings for the query
    # embeddings = your_embedding_model(query)

    # 2. Search vector database
    # docs = vector_db.search(embeddings)

    # 3. Generate response with LLM
    # response = llm.generate(query, context=docs)

    # 4. Stream the response
    for token in response:
        yield token
        await asyncio.sleep(0.01)
```

## Language Support

The application supports English and Turkish. Language files are in `client/src/context/LanguageContext.jsx`. To add more languages:

1. Add translations to the `translations` object
2. Add a language selector button in the UI

## Troubleshooting

### Backend won't start
- Check MongoDB is running: `mongosh` or `mongo`
- Verify `.env` file exists in `server/` with correct values
- Check port 5000 is not in use

### Python service not responding
- Check Docker container is running: `docker ps`
- View logs: `docker-compose logs python-service`
- Rebuild container: `docker-compose up --build`

### Frontend can't connect to backend
- Verify backend is running on port 5000
- Check browser console for CORS errors
- Ensure API URL in `client/src/services/api.js` is correct

## Development

### Client Development
```bash
cd client
npm run dev  # Vite dev server with hot reload
```

### Server Development
```bash
cd server
npm run dev  # Auto-reload on changes
```

### Python Service Development
For local development without Docker:
```bash
cd python-service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Building for Production

### Build Client
```bash
cd client
npm run build
```

The built files will be in `client/dist/`

## Production Considerations

- Change `JWT_SECRET` to a strong random string
- Use environment variables for all sensitive data
- Enable MongoDB authentication
- Use HTTPS for all services
- Implement rate limiting
- Add error logging and monitoring
- Consider using a process manager (PM2) for Node.js
- Set up proper CORS policies
- Implement proper error handling in streaming responses

## License

MIT
