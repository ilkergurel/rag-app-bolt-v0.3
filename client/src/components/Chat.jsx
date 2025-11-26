import { useState, useEffect, useContext } from 'react';
import { AuthContext } from '../context/AuthContext';
import { LanguageContext } from '../context/LanguageContext';
import { api } from '../services/api';
import Sidebar from './Sidebar';
import ChatArea from './ChatArea';
import { Globe } from 'lucide-react';
import { ReferenceContext } from '../context/ReferenceContext';
import ReferenceViewer from './ReferenceViewer';

export default function Chat() {
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [allMessages, setAllMessages] = useState({});
  const [isStreaming, setIsStreaming] = useState(false);
  const [loading, setLoading] = useState(true);
  const [streamController, setStreamController] = useState(null);
  const [streamingChatId, setStreamingChatId] = useState(null);
  const [progressStage, setProgressStage] = useState(null);

  const [selectedReference, setSelectedReference] = useState(null);
  const { token, logout, user } = useContext(AuthContext);
  const { language, changeLanguage, t } = useContext(LanguageContext);

  const messages = allMessages[currentChatId] || [];  

  useEffect(() => {
    loadChats();
  }, []);

  useEffect(() => {
    // If selected chat is not the processing chat, load
    if (currentChatId && currentChatId !== streamingChatId) {
      loadChat(currentChatId);
    }
    // If you come back to the processing chat, loadChat will not run and local (optimistic) state is kept

    // If you go to a different chat, close the reference window
    setSelectedReference(null);
  }, [currentChatId]);

  const loadChats = async () => {
    try {
      const fetchedChats = await api.getChats(token);
      setChats(fetchedChats);
    } catch (error) {
      console.error('Error loading chats:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadChat = async (chatId) => {
    try {
      const chat = await api.getChat(token, chatId);
      setAllMessages(prev => ({
        ...prev,
        [chatId]: chat.messages || []
      }));
    } catch (error) {
      console.error('Error loading chat:', error);
    }
  };

  const handleNewChat = async () => {
    try {
      const newChat = await api.createChat(token);
      newChat.title = t("newChat")   // New Chat title shall be according to the language selection
      setChats([newChat, ...chats]);
      setCurrentChatId(newChat._id);
      setAllMessages(prev => ({
        ...prev,
        [newChat._id]: []
      }));
    } catch (error) {
      console.error('Error creating chat:', error);
    }
  };

  const handleSelectChat = (chatId) => {
    setCurrentChatId(chatId);
  };

  const handleDeleteChat = async (chatId) => {
    try {
      await api.deleteChat(token, chatId);
      setChats(chats.filter((chat) => chat._id !== chatId));
      
      setAllMessages(prev => {
        const newAllMessages = { ...prev };
        delete newAllMessages[chatId];
        return newAllMessages;
      });

      if (currentChatId === chatId) {
        setCurrentChatId(null);
      }
    } catch (error) {
      console.error('Error deleting chat:', error);
    }
  };

  const handleSendMessage = async (message) => {
    if (!currentChatId) {
      const newChat = await api.createChat(token);
      newChat.title = t("newChat")   // New Chat title shall be according to the language selection
      setChats([newChat, ...chats]);
      setCurrentChatId(newChat._id);
      setAllMessages(prev => ({ ...prev, [newChat._id]: [] }));

      setTimeout(() => {
        sendMessageToChat(newChat._id, message);
      }, 100);
      return;
    }

    sendMessageToChat(currentChatId, message);
  };

  const sendMessageToChat = async (chatId, message) => {
    const userMessage = { role: 'user', content: message };
    setAllMessages(prev => ({
      ...prev,
      [chatId]: [...(prev[chatId] || []), userMessage]
    }));
    setIsStreaming(true);
    setStreamingChatId(chatId);
    setProgressStage('analyzing');

    // Create a new controller for this stream
    const controller = new AbortController();
    setStreamController(controller);

    // We do NOT use a local variable 'assistantMessage' for content updates anymore
    const initialAssistantMsg = { role: 'assistant', content: '', isStreaming: true };

    setAllMessages(prev => ({
      ...prev,
      [chatId]: [...(prev[chatId] || []), initialAssistantMsg]
    }));

    try {
      await api.sendMessage(token, chatId, message, (data) => {
        if (data.type === 'progress') {
          setProgressStage(data.stage);
        }

        if (data.chunk) {
          // Clear progress when first content chunk arrives
          if (progressStage) {
            setProgressStage(null);
          }

          setAllMessages(prev => {
            const currentMessages = prev[chatId] || [];
            // Create a deep copy of the array to ensure React detects change
            const newChatMessages = [...currentMessages];

            if (newChatMessages.length > 0) {
               // Get the last message (the assistant one)
               const lastMsg = newChatMessages[newChatMessages.length - 1];

               // Create a completely NEW object with the appended text
               // This forces React to re-render this specific message
               newChatMessages[newChatMessages.length - 1] = {
                 ...lastMsg,
                 content: lastMsg.content + data.chunk
               };
            }
            return { ...prev, [chatId]: newChatMessages };
          });
          // ----------------------------------------
        }

        if (data.done) {
          setProgressStage(null);
          setIsStreaming(false);
          setStreamController(null);
          setStreamingChatId(null);
          
          // Finalize the message (remove isStreaming flag)
          setAllMessages(prev => {
            const currentMessages = prev[chatId] || [];
            const newChatMessages = [...currentMessages];
            if (newChatMessages.length > 0) {
               const lastMsg = newChatMessages[newChatMessages.length - 1];
               newChatMessages[newChatMessages.length - 1] = { 
                 ...lastMsg, 
                 isStreaming: false 
               };
            }
            return { ...prev, [chatId]: newChatMessages };
          });        
          loadChats();
        }
        
        if (data.error) {
          console.error('Streaming error:', data.error);
          setProgressStage(null);
          setIsStreaming(false);
          setStreamController(null);
          setStreamingChatId(null);
          assistantMessage.content = 'Error: ' + data.error;
          setAllMessages(prev => {
            const newChatMessages = [...(prev[chatId] || [])];
            newChatMessages[newChatMessages.length - 1] = { ...assistantMessage, content: 'Error: ' + data.error, isStreaming: false };
            return { ...prev, [chatId]: newChatMessages };
          });
        }
      }, { signal: controller.signal });
    } catch (error) {
      if (!controller.signal.aborted) {
        console.error('Error sending message:', error);
        setAllMessages(prev => {
          const newChatMessages = [...(prev[chatId] || [])];
          newChatMessages[newChatMessages.length - 1] = { ...assistantMessage, content: 'Error: Failed to send message', isStreaming: false };
          return { ...prev, [chatId]: newChatMessages };
        });
      }
      setProgressStage(null);
      setIsStreaming(false);
      setStreamController(null);
      setStreamingChatId(null); 
    }
  };

  const onStopStreaming = async () => {
    if (streamController) {
      // 1) Cancel client-side streaming immediately
      streamController.abort();

      // 2) Optimistically update UI state so the STOP button and loader vanish
      setIsStreaming(false);
      setStreamController(null);
      const stoppedChatId = streamingChatId;
      setStreamingChatId(null);

      // 3) Also update the last assistant message in case it's still flagged as streaming
      setAllMessages(prev => {
        const chatId = stoppedChatId; 
        if (!chatId || !prev[chatId]) return prev;

        const newChatMessages = [...prev[chatId]];
        const lastIdx = newChatMessages.length - 1;
        if (lastIdx < 0) return prev;

        const lastMsg = newChatMessages[lastIdx];
        if (lastMsg.role === 'assistant' && lastMsg.isStreaming) {
          newChatMessages[lastIdx] = {
            ...lastMsg,
            isStreaming: false,
            content: lastMsg.content + ' -- '
          };
        }
        return { ...prev, [chatId]: newChatMessages };
      });

      // 4) Notify backend to stop the streaming process
      try {
        await api.stopStreaming(token, stoppedChatId);
      } catch (err) {
        console.error('Failed to send stop request:', err);
        // Optionally show a user-visible error or fallback
      }
    }
  };

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-50 dark:bg-slate-950">
        <div className="text-gray-600 dark:text-slate-400 text-lg">{t('loading')}</div>
      </div>
    );
  }

  return (
    <ReferenceContext.Provider value={{ reference: selectedReference, setReference: setSelectedReference }}>  
      <div className="h-screen flex bg-white dark:bg-slate-950">
        <Sidebar
          chats={chats}
          currentChatId={currentChatId}
          onSelectChat={handleSelectChat}
          onNewChat={handleNewChat}
          onDeleteChat={handleDeleteChat}
          onLogout={logout}
        />
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="bg-white dark:bg-slate-900 border-b border-gray-200 dark:border-slate-800 p-4 flex justify-between items-center flex-shrink-0">
            <div className="text-gray-900 dark:text-white font-semibold">
              {user?.username}
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => changeLanguage('en')}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
                  language === 'en'
                    ? 'bg-blue-600 dark:bg-slate-700 text-white'
                    : 'bg-gray-100 dark:bg-slate-800 text-gray-700 dark:text-slate-300 hover:bg-gray-200 dark:hover:bg-slate-700'
                }`}
              >
                <Globe size={16} />
                EN
              </button>
              <button
                onClick={() => changeLanguage('tr')}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
                  language === 'tr'
                    ? 'bg-blue-600 dark:bg-slate-700 text-white'
                    : 'bg-gray-100 dark:bg-slate-800 text-gray-700 dark:text-slate-300 hover:bg-gray-200 dark:hover:bg-slate-700'
                }`}
              >
                <Globe size={16} />
                TR
              </button>
            </div>
          </div>
          <div className="flex-1 flex overflow-hidden bg-gray-50 dark:bg-slate-950">
            {/* Chat Area */}
            <div className="flex-1 flex items-center justify-center overflow-auto p-4">
              <div className="w-full max-w-4xl h-full flex flex-col">
                <ChatArea
                  messages={messages}
                  onSendMessage={handleSendMessage}
                  isStreaming={isStreaming}
                  onStopStreaming={onStopStreaming}
                  selectedReference={selectedReference}
                  setSelectedReference={setSelectedReference}
                  progressStage={progressStage}
                />               
              </div>
            </div>

            {/* Reference viewer*/}
            {selectedReference && (
              <ReferenceViewer 
                reference={selectedReference}
                onClose={() => setSelectedReference(null)}
              />
            )}
          </div>
        </div>
      </div>
    </ReferenceContext.Provider>
  );
}
