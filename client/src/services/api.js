const API_URL = '/api';


export const api = {
  async register(username, password) {
    const response = await fetch(`${API_URL}/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Registration failed');
    }

    return response.json();
  },

  async login(username, password) {
    const response = await fetch(`${API_URL}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Login failed');
    }

    return response.json();
  },

  async getChats(token) {
    const response = await fetch(`${API_URL}/chats`, {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      throw new Error('Failed to fetch chats');
    }

    return response.json();
  },

  async getChat(token, chatId) {
    const response = await fetch(`${API_URL}/chats/${chatId}`, {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      throw new Error('Failed to fetch chat');
    }

    return response.json();
  },

  async createChat(token, title = 'New Chat') {
    const response = await fetch(`${API_URL}/chats`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({ title }),
    });

    if (!response.ok) {
      throw new Error('Failed to create chat');
    }

    return response.json();
  },

  async sendMessage(token, chatId, message, onChunk, options = {}) {
    const { signal } = options;  // may be undefined

    // Early-abort if already signalled
    if (signal?.aborted) {
      onChunk({ done: true});
      return;
    }    

    try {
      const response = await fetch(`${API_URL}/chats/${chatId}/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ message }),
        signal
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let leftover = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunkStr = decoder.decode(value, { stream: true });
        const lines = (leftover + chunkStr).split('\n');
        leftover = lines.pop(); // possible partial line at end

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            onChunk(data);
          }
        }
      }
      // final flush of leftover
      if (leftover.startsWith('data: ')) {
        const data = JSON.parse(leftover.slice(6));
        onChunk(data);
      }

      // notify done
      onChunk({ done: true }); 

    } catch (err) {
      if (err.name === 'AbortError') {
        onChunk({ done: true});
      } else {
        throw err;
      }
    }
  },

  async stopStreaming(token, chatId) {
    try {
      // 'fetch' kullanacak şekilde düzeltildi
      const response = await fetch(
        `${API_URL}/chats/${chatId}/stop`, 
        {
          method: 'POST',
          headers: { 
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json' 
          },
          body: JSON.stringify({}) // Boş gövde
        }
      );
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.message || 'Failed to stop stream');
      }
      return response.json();
    } catch (error) {
      console.error('Error stopping stream:', error.message);
      throw error;
    }
  },  

  async deleteChat(token, chatId) {
    const response = await fetch(`${API_URL}/chats/${chatId}`, {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      throw new Error('Failed to delete chat');
    }

    return response.json();
  },
};
