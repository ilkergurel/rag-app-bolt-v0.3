import express from 'express';
import Chat from '../models/Chat.js';
import { authMiddleware } from '../middleware/auth.js';
import { chatLimiter } from '../middleware/rateLimiter.js';
import logger from '../utils/logger.js';
import axios from 'axios';

const router = express.Router();

const activeStreams = new Map();

router.get('/', authMiddleware, async (req, res) => {
  try {
    const chats = await Chat.find({ userId: req.userId })
      .sort({ updatedAt: -1 })
      .select('_id title createdAt updatedAt');

    logger.debug('Fetched chats', { userId: req.userId, count: chats.length });
    res.json(chats);
  } catch (error) {
    logger.error('Error fetching chats:', { error: error.message, userId: req.userId });
    res.status(500).json({
      message: process.env.NODE_ENV === 'production'
        ? 'Error fetching chats'
        : error.message
    });
  }
});

router.get('/:chatId', authMiddleware, async (req, res) => {
  try {
    const chat = await Chat.findOne({ _id: req.params.chatId, userId: req.userId });

    if (!chat) {
      logger.warn('Chat not found', { chatId: req.params.chatId, userId: req.userId });
      return res.status(404).json({ message: 'Chat not found' });
    }

    logger.debug('Fetched chat', { chatId: req.params.chatId, userId: req.userId });
    res.json(chat);
  } catch (error) {
    logger.error('Error fetching chat:', { error: error.message, chatId: req.params.chatId, userId: req.userId });
    res.status(500).json({
      message: process.env.NODE_ENV === 'production'
        ? 'Error fetching chat'
        : error.message
    });
  }
});

router.post('/', authMiddleware, async (req, res) => {
  try {
    const { title } = req.body;

    const chat = new Chat({
      userId: req.userId,
      title: title || 'New Chat',
      messages: []
    });

    await chat.save();
    logger.info('Chat created', { chatId: chat._id, userId: req.userId });
    res.status(201).json(chat);
  } catch (error) {
    logger.error('Error creating chat:', { error: error.message, userId: req.userId });
    res.status(500).json({
      message: process.env.NODE_ENV === 'production'
        ? 'Error creating chat'
        : error.message
    });
  }
});

router.post('/:chatId/message', authMiddleware, chatLimiter, async (req, res) => {
  const { chatId } = req.params;
  const { message } = req.body;

  const controller = new AbortController();
  activeStreams.set(chatId, {controller});

  logger.info(`Starting stream for chat: ${chatId}`);
  try {
    if (!message || message.trim().length === 0) {
      activeStreams.delete(chatId);
      return res.status(400).json({ message: 'Message cannot be empty' });
    }

    if (message.length > 5000) {
      activeStreams.delete(chatId);
      return res.status(400).json({ message: 'Message is too long (max 5000 characters)' });
    }

    const chat = await Chat.findOne({ _id: req.params.chatId, userId: req.userId });

    if (!chat) {
      logger.warn('Chat not found for message', { chatId: req.params.chatId, userId: req.userId });
      activeStreams.delete(chatId);
      return res.status(404).json({ message: 'Chat not found' });
    }

    chat.messages.push({
      role: 'user',
      content: message
    });

    if (chat.messages.length === 1 && chat.title === 'New Chat') {
      chat.title = message.substring(0, 50) + (message.length > 50 ? '...' : '');
    }

    await chat.save();
    logger.info('User message saved', { chatId: chat._id, userId: req.userId });

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.flushHeaders();

    let assistantMessage = '';
    let streamError = false;
    let buffer = ''; // Buffer for parsing JSON lines

    const cleanup = async () => {
      // Only clean the controller which belongs to the request
      const active = activeStreams.get(chatId);
      if (active && active.controller === controller) {
        activeStreams.delete(chatId);
      }
      logger.info(`Stream cleaned up for chat: ${chatId}`);

      if (!streamError && assistantMessage.trim().length > 0) {
        try {
          // Check for abort signal before saving
          const finalContent = assistantMessage;
          if (finalContent && finalContent.trim().length > 0) {
            await Chat.findOneAndUpdate(
              { _id: chatId, userId: req.userId },
              { $push: { messages: { role: 'assistant', content: finalContent } } }
            );
            logger.info('Assistant message saved', { chatId: chatId, userId: req.userId });
          } else {
            logger.info('Assistant message was empty, not saving.', { chatId: chatId });
          }
        } catch (saveError) {
          logger.error('Error saving assistant message:', { error: saveError.message, chatId: chatId });
        }
      }
    };

    const decoder = new TextDecoder();

    try {
      const pythonResponse = await fetch(`${process.env.PYTHON_SERVICE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chat_id: chatId, query: message }),
        signal: controller.signal,
      });

      if (!pythonResponse.ok) {
        throw new Error(`Python service error: ${pythonResponse.statusText}`);
      }

      // Process the stream and parse JSON events
      for await (const chunk of pythonResponse.body) {
        // Decode the binary chunk to string
        const chunkStr = decoder.decode(chunk, { stream: true });

        // Add to buffer
        buffer += chunkStr;

        // Split by newlines to get complete JSON lines
        const lines = buffer.split('\n');

        // Keep the last incomplete line in the buffer
        buffer = lines.pop() || '';

        // Process each complete line
        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            // Try to parse as JSON event
            const event = JSON.parse(line);

            if (event.type === 'progress') {
              // Emit progress event to client
              res.write(`data: ${JSON.stringify({
                type: 'progress',
                stage: event.stage
              })}\n\n`);
              if (res.flush) res.flush();

            } else if (event.type === 'chunk') {
              // This is content to display
              const content = event.content || '';
              assistantMessage += content;
              res.write(`data: ${JSON.stringify({ chunk: content })}\n\n`);
              if (res.flush) res.flush();

            } else if (event.type === 'done') {
              // Generation is complete
              // Don't send done yet, wait for references

            } else if (event.type === 'error') {
              // Error from Python service
              streamError = true;
              logger.error(`Python service error: ${event.message}`);
              res.write(`data: ${JSON.stringify({ error: event.message })}\n\n`);
              if (res.flush) res.flush();
            }

          } catch (parseError) {
            // JSON parse failed - log it
            logger.error(`[JSON PARSE ERROR] Failed to parse line: ${line.substring(0, 200)}`);
            logger.error(`[JSON PARSE ERROR] Error: ${parseError.message}`);

            // Safety: Don't forward lines that look like JSON events
            // This prevents raw JSON from appearing in the UI
            if (line.trim().startsWith('{') && line.includes('"type"')) {
              logger.warn(`[SAFETY] Skipping line that looks like malformed JSON event`);
            } else {
              // Treat as regular text chunk
              assistantMessage += line;
              res.write(`data: ${JSON.stringify({ chunk: line })}\n\n`);
              if (res.flush) res.flush();
            }
          }
        }
      }

      // End of Stream
      res.write(`data: ${JSON.stringify({ done: true, chatId: chat._id })}\n\n`);
      if (res.flush) res.flush();
      res.end();
      await cleanup();

    } catch (error) {
      if (error.name === 'AbortError') {
        logger.warn(`Stream aborted by user: ${chatId}`);
      } else {
        logger.error('Stream error:', error);
        res.write(`data: ${JSON.stringify({ error: 'Stream failed' })}\n\n`);
      }
      res.end();
      await cleanup();
    }

  } catch (error) {
    logger.error('Handler error:', error);
    if (!res.headersSent) res.status(500).json({ message: error.message });
    activeStreams.delete(chatId);
  }
});


router.post('/:chatId/stop', authMiddleware, async (req, res) => {
  const { chatId } = req.params;
  const active = activeStreams.get(chatId);

  if (active) {
    logger.info(`Received stop request for chat: ${chatId}`);

    // additionally: send a dedicated abort request to python service
    try {
      await axios.post(`${process.env.PYTHON_SERVICE_URL}/abort`, { chat_id: chatId});
      logger.info(`Abort signal sent to python for chat: ${chatId}`);
      res.status(200).json({ message: 'Stop signal sent' });
    } catch(err) {
      logger.warn(`Failed to send python abort for chat: ${chatId}`, { error: err.message });
      res.status(500).json({ message: 'Failed to send stop signal to Python' });
    }
  } else {
    logger.warn(`Stop request received, but no active stream found for chat: ${chatId}`);
    res.status(404).json({ message: 'No active stream to stop' });
  }
});


router.delete('/:chatId', authMiddleware, async (req, res) => {
  try {
    const result = await Chat.deleteOne({ _id: req.params.chatId, userId: req.userId });

    if (result.deletedCount === 0) {
      logger.warn('Chat not found for deletion', { chatId: req.params.chatId, userId: req.userId });
      return res.status(404).json({ message: 'Chat not found' });
    }

    try {
      await axios.post(`${process.env.PYTHON_SERVICE_URL}/abort`, { chat_id: req.params.chatId});
      logger.info(`Abort signal sent to python for chat: ${req.params.chatId}`);
    } catch(err) {
      logger.warn(`Failed to send python abort for chat: ${req.params.chatId}`, { error: err.message });
    }

    // If exists, stop the active flow of the deleted chat
    const active = activeStreams.get(req.params.chatId);
    if (active) {
      active.controller.abort();
       if (active.stream && !active.stream.destroyed) {
            active.stream.destroy();
       }
      activeStreams.delete(req.params.chatId);
      logger.info(`Stream stopped for deleted chat: ${req.params.chatId}`);
    }

    logger.info('Chat deleted', { chatId: req.params.chatId, userId: req.userId });
    res.json({ message: 'Chat deleted successfully' });
  } catch (error) {
    logger.error('Error deleting chat:', {
      error: error.message,
      chatId: req.params.chatId,
      userId: req.userId
    });
    res.status(500).json({
      message: process.env.NODE_ENV === 'production'
        ? 'Error deleting chat'
        : error.message
    });
  }
});

export default router;
