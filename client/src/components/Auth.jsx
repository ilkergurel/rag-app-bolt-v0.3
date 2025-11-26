import { useState, useContext } from 'react';
import { AuthContext } from '../context/AuthContext';
import { LanguageContext } from '../context/LanguageContext';
import { api } from '../services/api';

// Simple regex for email validation
const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export default function Auth() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { login } = useContext(AuthContext);
  const { t, language, changeLanguage } = useContext(LanguageContext);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    // Email Validation Check 
    if (!emailRegex.test(email)) {
      setError(t('invalidEmail')); // Use a translated error
      return; // Stop execution
    }

    setLoading(true);

    try {
      let response;
      if (isLogin) {
        response = await api.login(email, password);
      } else {
        response = await api.register(email, password);
      }

      login(response.token, {
        userId: response.userId,
        username: response.username,
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-slate-900 dark:to-slate-950 flex items-center justify-center p-4 transition-colors">
      <div className="absolute top-4 right-4 flex gap-2">
        <button
          onClick={() => changeLanguage('en')}
          className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
            language === 'en'
              ? 'bg-blue-600 dark:bg-slate-700 text-white'
              : 'bg-white dark:bg-slate-800 text-gray-700 dark:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-700'
          }`}
        >
          EN
        </button>
        <button
          onClick={() => changeLanguage('tr')}
          className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
            language === 'tr'
              ? 'bg-blue-600 dark:bg-slate-700 text-white'
              : 'bg-white dark:bg-slate-800 text-gray-700 dark:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-700'
          }`}
        >
          TR
        </button>
      </div>

      <div className="w-full max-w-md">
        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl p-8 transition-colors">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2 text-center">
            {isLogin ? t('loginTitle') : t('registerTitle')}
          </h2>
          <p className="text-gray-600 dark:text-slate-400 text-center mb-8">
            {t('welcome')}
          </p>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                {t('email')}
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-slate-500 focus:border-transparent transition-all"
                required
                autoComplete="email"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                {t('password')}
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-slate-500 focus:border-transparent transition-all"
                required
                minLength={6}
              />
            </div>

            {error && (
              <div className="bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400 px-4 py-3 rounded-lg text-sm">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 dark:bg-slate-700 text-white py-3 rounded-lg font-medium hover:bg-blue-700 dark:hover:bg-slate-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? t('loading') : isLogin ? t('loginButton') : t('registerButton')}
            </button>
          </form>

          <div className="mt-6 text-center">
            <button
              onClick={() => {
                setIsLogin(!isLogin);
                setError('');
              }}
              className="text-gray-600 dark:text-slate-400 hover:text-gray-800 dark:hover:text-slate-200 text-sm transition-colors"
            >
              {isLogin ? t('dontHaveAccount') : t('alreadyHaveAccount')}{' '}
              <span className="font-medium text-blue-600 dark:text-slate-300">
                {isLogin ? t('register') : t('login')}
              </span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
