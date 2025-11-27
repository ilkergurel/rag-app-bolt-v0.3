import React, { useState, useRef, useEffect, useContext } from 'react'; 
import ReactMarkdown from 'react-markdown'; 
import remarkGfm from 'remark-gfm';       
import { Send, User, Bot, Loader2, Copy, Check, Square, Brain, Database, Sparkles } from 'lucide-react';
import { LanguageContext } from '../context/LanguageContext';
import { ReferenceContext } from '../context/ReferenceContext'; 


const BotMessageContent = ({ content }) => {

  const { setReference } = useContext(ReferenceContext);
  const { t } = useContext(LanguageContext);

   // To track which file path has been copied
  const [copiedPath, setCopiedPath] = useState(null);

  // --- PART 1: Find links in the text ---

  const renderMainText = (text) => {
    const fileExtensions = 'pdf|doc|docx|txt|xlsx|xls|csv|json|xml|md|html|htm|epub|mobi|azw|azw3|jpg|jpeg|png|gif|svg|webp|bmp|mp3|mp4|avi|mov|wav|zip|rar|7z';
    const windowsPathRegex = new RegExp(
      `([A-Za-z]:[\\\\\\/](?:[^\\\\\\/:\\*\\?"<>\\|\\r\\n]+[\\\\\\/])*[^\\\\\\/:\\*\\?"<>\\|\\r\\n]+\\.(${fileExtensions}))`,
      'gi'
    );
    const unixPathRegex = new RegExp(
      `(\\/[^\\/\\0]+\\/)*[^\\/\\0]+\\.(${fileExtensions})`,
      'gi'
    );
  
    const parts = [];
    let lastIndex = 0;
    const allMatches = [];
  
    let match;
    while ((match = windowsPathRegex.exec(text)) !== null) {
      allMatches.push({ index: match.index, length: match[0].length, text: match[0], type: 'windows' });
    }
    windowsPathRegex.lastIndex = 0;
  
    while ((match = unixPathRegex.exec(text)) !== null) {
      const overlaps = allMatches.some(m => (match.index >= m.index && match.index < m.index + m.length));
      if (!overlaps) {
        allMatches.push({ index: match.index, length: match[0].length, text: match[0], type: 'unix' });
      }
    }
  
    allMatches.sort((a, b) => a.index - b.index);
  
    allMatches.forEach((match, idx) => {
      if (match.index > lastIndex) {
        parts.push(<span key={`text-${lastIndex}`}>{text.substring(lastIndex, match.index)}</span>);
      }
      const normalizedPath = match.text.replace(/\\/g, '/');
      const filePath = match.type === 'windows' ? `file:///${normalizedPath}` : `file://${normalizedPath}`;
      parts.push(
        <a
          key={`link-${match.index}-${idx}`}
          href={filePath}
          className="text-blue-600 dark:text-blue-400 hover:underline font-medium break-all"
          onClick={(e) => { e.preventDefault(); window.open(filePath, '_blank'); }}
          title={match.text}
        >
          {match.text}
        </a>
      );
      lastIndex = match.index + match.length;
    });
  
    if (lastIndex < text.length) {
      parts.push(<span key={`text-${lastIndex}`}>{text.substring(lastIndex)}</span>);
    }
    
    return parts.length > 0 ? parts : text;
  };

  // --- PART 2: Divide References ---

  // From python-service comes
  const jsonSplitToken = "\n\n__JSON_START__";
  let mainText = content;
  let references = [];
  let groupedReferences = {};

  // Control whether there are references in the context
  if (content.includes(jsonSplitToken)) {
    const parts = content.split(jsonSplitToken);
    mainText = parts[0]; // Text part is before jsonSplitToken

    // Fix: If the text contains literal \n characters (as text, not newlines), convert them to actual newlines
    // This happens when the LLM outputs the literal string "\n\n" instead of actual newline characters
    if (mainText.includes('\\n')) {
      mainText = mainText.replace(/\\n/g, '\n');
    }
    try {
      // After jsonSplitToken part is JSON data
      const jsonDataString = parts[1];
      references = JSON.parse(jsonDataString);
      
      // Group the references according to the sources
      groupedReferences = references.reduce((acc, ref) => {
        const key = ref.source; // Source (eg.: "D:/Bilgi/kitaplar/...")
        if (!acc[key]) {
          acc[key] = [];
        }
        acc[key].push(ref);
        return acc;
      }, {});
    } catch (e) {
      console.error("[BotMessageContent] Reference JSON data could not be parsed:", e);
      console.error("[BotMessageContent] JSON string was:", parts[1]?.substring(0, 200));
      // JSON error case, show output text anyway
    }
  } else {
    // No references - but still need to fix escaped newlines if present
    if (mainText.includes('\\n')) {
      mainText = mainText.replace(/\\n/g, '\n');
    }
  }

  // --- PART 3: Click Functions ---

  // This function now copies to clipboard instead of opening 'file://'
  const handleFileCopyClick = (e, filepath) => {
    e.preventDefault();
    // Copy the raw file path, not the 'file://' version
    const rawPath = filepath.replace(/\\/g, '\\'); 
    
    // Clipboard API (requires HTTPS or localhost)
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(rawPath).then(() => {
        setCopiedPath(filepath); // Set copied state
        setTimeout(() => setCopiedPath(null), 2000); // Clear "copied" status after 2s
      }).catch(err => {
        console.error('Failed to copy to clipboard:', err);
      });
    } else {
      // Fallback for insecure (HTTP) or older browsers
      // This might not work due to 'file://' restrictions but is a fallback
      try {
        const tempInput = document.createElement('textarea');
        tempInput.value = rawPath;
        document.body.appendChild(tempInput);
        tempInput.select();
        document.execCommand('copy');
        document.body.removeChild(tempInput);
        
        setCopiedPath(filepath);
        setTimeout(() => setCopiedPath(null), 2000);
      } catch (err) {
         console.error('Failed to copy using fallback method:', err);
      }
    }
  };

  // As reference point is clicked [1] (side window is opened)
  const handleReferenceClick = (e, ref) => {
    e.preventDefault();
    setReference(ref); // Via Context update the state in 'Chat.jsx'
  };

  // --- PART 4: Render ---

  return (
    <>
      {/* 1. Main Text Area */}
    <div className="leading-relaxed markdown-body">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]} // Enable standard Markdown features
        >
          {mainText}
        </ReactMarkdown>
      </div>

      {/* 2. References Area */}
      {references.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-300 dark:border-slate-600">
          <h4 className="text-sm font-semibold mb-2 text-gray-700 dark:text-slate-300">{t('references')}:</h4>
          <ul className="space-y-2">
            {Object.entries(groupedReferences).map(([filepath, refs]) => (
              <li key={filepath} className="text-sm leading-relaxed">
                
                {/* Copy Button */}
                <button
                  onClick={(e) => handleFileCopyClick(e, filepath)}
                  className={`flex text-left items-center gap-2 px-2 py-0.5 rounded-md transition-colors ${
                    copiedPath === filepath
                      ? 'bg-green-100 dark:bg-green-800 text-green-700 dark:text-green-200'
                      : 'bg-gray-100 dark:bg-slate-800 text-gray-600 dark:text-slate-300 hover:bg-gray-200 dark:hover:bg-slate-700'
                  }`}
                  title={`${t('copyPath')}: ${filepath}`}
                >
                  {copiedPath === filepath ? (
                    <Check size={14} className="flex-shrink-0"/>
                  ) : (
                    <Copy size={14} className="flex-shrink-0"/>
                  )}
                  {/* Truncate long paths to avoid layout breaking */}
                  <span className="font-medium">{filepath}</span>
                </button>
                
                {/* Clickable Reference Numbers [1, 2, 3] */}
                <span className="text-gray-700 dark:text-slate-300 ml-2">- [</span>
                {refs.map((ref, index) => (
                  <span key={ref.id}>
                    <a
                      href="#"
                      onClick={(e) => handleReferenceClick(e, ref)}
                      className="text-blue-600 dark:text-blue-400 hover:underline px-1"
                      title={`${t('showReference')}:\n${t('page')}: ${ref.page}\n${t('content')}: ${ref.content}`}
                    >
                      {index + 1} 
                    </a>
                    {index < refs.length - 1 && (
                      <span className="text-gray-700 dark:text-slate-300">,</span>
                    )}
                  </span>
                ))}
                <span className="text-gray-700 dark:text-slate-300">]</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </>
  );
};

// --- Main ChatArea Content  ---
export default function ChatArea({ messages, onSendMessage, isStreaming, onStopStreaming, selectedReference, setSelectedReference, progressStage }) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const { t } = useContext(LanguageContext);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const isScrolledToBottom = () => {
    if (!messagesContainerRef.current) return true;
    const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
    return scrollHeight - scrollTop - clientHeight < 50;
  };

  const handleScroll = () => {
    setShouldAutoScroll(isScrolledToBottom());
  };

  useEffect(() => {
    if (shouldAutoScroll) {
      scrollToBottom();
    }
  }, [messages, shouldAutoScroll]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isStreaming) {
      onSendMessage(input);
      setInput('');
      setShouldAutoScroll(true);
    }
  };

  const handleStopClick = (e) => {
    e.preventDefault();
    if (onStopStreaming) {
      onStopStreaming();
    }
  };

  const getProgressIcon = (stage) => {
    switch (stage) {
      case 'analyzing':
        return <Brain size={18} className="text-blue-600 dark:text-blue-400 animate-pulse" />;
      case 'retrieving':
        return <Database size={18} className="text-blue-600 dark:text-blue-400 animate-pulse" />;
      case 'generating':
        return <Sparkles size={18} className="text-blue-600 dark:text-blue-400 animate-pulse" />;
      default:
        return <Loader2 size={18} className="text-blue-600 dark:text-blue-400 animate-spin" />;
    }
  };

  const getProgressText = (stage) => {
    const textMap = {
      analyzing: 'progressAnalyzing',
      retrieving: 'progressRetrieving',
      generating: 'progressGenerating'
    };
    return t(textMap[stage] || 'loading');
  };

  return (
    <div className="flex-1 flex flex-col  rounded-2xl shadow-lg  overflow-hidden">
      <div
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-6 space-y-6 chat-area-scrollbar"
      >
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
             <div className="text-center text-gray-400 dark:text-slate-500">
               <Bot size={64} className="mx-auto mb-4 text-gray-300 dark:text-slate-600" />
               <h2 className="text-2xl font-semibold text-gray-600 dark:text-slate-400 mb-2">
                 {t('welcome')}
               </h2>
               <p>{t('startNewChat')}</p>
             </div>
          </div>
        ) : (
          <>
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex gap-4 ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {message.role === 'assistant' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600 dark:bg-slate-700 flex items-center justify-center">
                    {message.isStreaming ? (
                      <Loader2 size={20} className="text-white animate-spin" />
                    ) : (
                      <Bot size={20} className="text-white" />
                    )}
                  </div>
                )}
                <div
                  className={`max-w-3xl rounded-2xl ${
                    message.role === 'user'
                      ? 'bg-blue-600 dark:bg-slate-700 text-white rounded-br-sm'
                      : 'text-gray-900 dark:text-slate-100 rounded-bl-sm'
                  }`}
                >
                  {message.role === 'user' ? (
                     <p className="whitespace-pre-wrap leading-relaxed px-5 py-3">
                       {message.content}
                     </p>
                  ) : (
                    <div className="px-5 py-3">
                        {progressStage && message.isStreaming && !message.content ? (
                            /* Show progress indicator when streaming but no content yet */
                            <div className="flex items-center gap-3">
                                <div className="flex-shrink-0">
                                    {getProgressIcon(progressStage)}
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-medium text-slate-700 dark:text-slate-300 animate-pulse">
                                        {getProgressText(progressStage)}
                                    </span>
                                    <span className="flex gap-1">
                                        <span className="w-1 h-1 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce"></span>
                                        <span className="w-1 h-1 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></span>
                                        <span className="w-1 h-1 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                                    </span>
                                </div>
                            </div>
                        ) : message.content ? (
                            <BotMessageContent content={message.content} />
                        ) : (
                            /* Only show dots if content is strictly empty and no progress */
                            <div className="flex gap-1 h-6 items-center">
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
                            </div>
                        )}
                      </div>
                  )}
                </div>

                {message.role === 'user' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-300 dark:bg-slate-600 flex items-center justify-center">
                    <User size={20} className="text-gray-700 dark:text-slate-300" />
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <div className="border-t border-gray-200 dark:border-slate-800 p-4 bg-gray-50 dark:bg-slate-800">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={t('typeMessage')}
              disabled={isStreaming}
              className="flex-1 px-4 py-3 rounded-xl border border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-slate-600 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            />
            {isStreaming ? (
              // If streaming: show STOP button
              <button
                type="button" 
                onClick={handleStopClick}
                className="bg-blue-600 dark:bg-slate-700 text-white px-6 py-3 rounded-xl hover:bg-blue-700 dark:hover:bg-slate-600 transition-colors flex items-center gap-2 font-medium"
              >
                <Square size={20} />
                {t('stop')}
              </button>
            ) : (
              // If no streaming: show SEND button
              <button
                type="submit"
                disabled={!input.trim()}
                className="bg-blue-600 dark:bg-slate-700 text-white px-6 py-3 rounded-xl hover:bg-blue-700 dark:hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2 font-medium"
              >
                <Send size={20} />
                {t('send')}
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}