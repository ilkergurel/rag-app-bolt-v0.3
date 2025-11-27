import { useContext, useState } from 'react';
import { MessageSquare, Plus, Trash2, LogOut, Settings, Moon, Sun, X } from 'lucide-react';
import { LanguageContext } from '../context/LanguageContext';
import { useTheme } from '../context/ThemeContext';

export default function Sidebar({ chats, currentChatId, onSelectChat, onNewChat, onDeleteChat, onLogout }) {
  const { t } = useContext(LanguageContext);
  const { theme, setTheme } = useTheme();
  const [showPreferences, setShowPreferences] = useState(false);

  return (
    <div className="w-60 bg-white dark:bg-slate-900 text-gray-900 dark:text-white flex flex-col h-screen border-r border-gray-200 dark:border-slate-700 transition-colors">
      <div className="p-4 border-b border-gray-200 dark:border-slate-700">
        <button
          onClick={onNewChat}
          className="w-full bg-blue-600 hover:bg-blue-700 dark:bg-slate-700 dark:hover:bg-slate-600 text-white px-4 py-3 rounded-lg flex items-center justify-center gap-2 transition-colors font-medium"
        >
          <Plus size={20} />
          {t('newChat')}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-2 chat-area-scrollbar">
        {chats.length === 0 ? (
          <div className="text-gray-500 dark:text-slate-400 text-sm text-center mt-8">
            {t('noChats')}
          </div>
        ) : (
          chats.map((chat) => (
            <div
              key={chat._id}
              className={`group relative p-3 rounded-lg cursor-pointer transition-all ${
                currentChatId === chat._id
                  ? 'bg-gray-200 dark:bg-slate-700'
                  : 'hover:bg-gray-100 dark:hover:bg-slate-800'
              }`}
              onClick={() => onSelectChat(chat._id)}
            >
              <div className="flex items-start gap-2">
                <MessageSquare size={18} className="mt-0.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm truncate">{chat.title}</p>
                  <p className="text-xs text-gray-500 dark:text-slate-400 mt-1">
                    {new Date(chat.updatedAt).toLocaleDateString()}
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteChat(chat._id);
                  }}
                  className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-gray-300 dark:hover:bg-slate-600 rounded"
                  title={t('deleteChat')}
                >
                  <Trash2 size={16} />
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="p-4 border-t border-gray-200 dark:border-slate-700 space-y-2">
        <button
          onClick={() => setShowPreferences(!showPreferences)}
          className="w-full text-gray-700 dark:text-slate-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-slate-800 px-4 py-3 rounded-lg flex items-center justify-center gap-2 transition-colors"
        >
          <Settings size={20} />
          {t('preferences')}
        </button>

        <button
          onClick={onLogout}
          className="w-full text-gray-700 dark:text-slate-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-slate-800 px-4 py-3 rounded-lg flex items-center justify-center gap-2 transition-colors"
        >
          <LogOut size={20} />
          {t('logout')}
        </button>
      </div>

      {showPreferences && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-slate-800 rounded-lg p-6 w-96 max-w-md shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">{t('preferences')}</h2>
              <button
                onClick={() => setShowPreferences(false)}
                className="text-gray-500 hover:text-gray-700 dark:text-slate-400 dark:hover:text-slate-200"
              >
                <X size={24} />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                  {t('theme')}
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => setTheme('light')}
                    className={`flex items-center justify-center gap-2 px-4 py-3 rounded-lg border-2 transition-all ${
                      theme === 'light'
                        ? 'border-blue-600 bg-blue-100 text-blue-700 font-medium'
                        : 'border-gray-300 dark:border-slate-600 hover:border-gray-400 dark:hover:border-slate-500 text-gray-700 dark:text-slate-300'
                    }`}
                  >
                    <Sun size={20} />
                    <span>{t('light')}</span>
                  </button>
                  <button
                    onClick={() => setTheme('dark')}
                    className={`flex items-center justify-center gap-2 px-4 py-3 rounded-lg border-2 transition-all ${
                      theme === 'dark'
                        ? 'border-blue-600 bg-blue-100 text-blue-700 font-medium'
                        : 'border-gray-300 dark:border-slate-600 hover:border-gray-400 dark:hover:border-slate-500 text-gray-700 dark:text-slate-300'
                    }`}
                  >
                    <Moon size={20} />
                    <span>{t('dark')}</span>
                  </button>
                </div>
              </div>
            </div>

            <div className="mt-6 flex justify-end">
              <button
                onClick={() => setShowPreferences(false)}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              >
                {t('close')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
