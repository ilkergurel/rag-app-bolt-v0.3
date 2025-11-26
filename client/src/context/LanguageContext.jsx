import { createContext, useState } from 'react';

export const LanguageContext = createContext();

const translations = {
  en: {
    login: 'Login',
    register: 'Register',
    email: 'Email',
    invalidEmail: 'Invalid email address',
    password: 'Password',
    logout: 'Logout',
    newChat: 'New Chat',
    deleteChat: 'Delete Chat',
    typeMessage: 'Type your message...',
    send: 'Send',
    welcome: 'Welcome to RAG Chat',
    startNewChat: 'Start a new chat to begin',
    noChats: 'No chat history yet',
    loginTitle: 'Login to your account',
    registerTitle: 'Create new account',
    alreadyHaveAccount: 'Already have an account?',
    dontHaveAccount: "Don't have an account?",
    loginButton: 'Login',
    registerButton: 'Register',
    loading: 'Loading...',
    error: 'Error',
    success: 'Success',
    preferences: 'Preferences',
    theme: 'Theme',
    light: 'Light',
    dark: 'Dark',
    close: 'Close',
    page: 'Page',
    references: 'References',
    reference: 'Reference',
    file: 'File',
    openFile: 'Open File',
    showReference: 'Show Reference',
    content: 'Content',
    stop: 'Stop',
    unknownFile: 'Unknown File',
    progressAnalyzing: 'Analyzing',
    progressRetrieving: 'Searching database',
    progressGenerating: 'Generating response'
  },

  tr: {
    login: 'Giriş Yap',
    register: 'Kayıt Ol',
    email: 'E-Posta',
    invalidEmail: 'Geçersiz e-posta adresi',
    password: 'Şifre',
    logout: 'Çıkış Yap',
    newChat: 'Yeni Sohbet',
    deleteChat: 'Sohbeti Sil',
    typeMessage: 'Mesajınızı yazın...',
    send: 'Gönder',
    welcome: 'RAG Sohbete Hoş Geldiniz',
    startNewChat: 'Başlamak için yeni bir sohbet başlatın',
    noChats: 'Henüz sohbet geçmişi yok',
    loginTitle: 'Hesabınıza giriş yapın',
    registerTitle: 'Yeni hesap oluştur',
    alreadyHaveAccount: 'Zaten hesabınız var mı?',
    dontHaveAccount: 'Hesabınız yok mu?',
    loginButton: 'Giriş Yap',
    registerButton: 'Kayıt Ol',
    loading: 'Yükleniyor...',
    error: 'Hata',
    success: 'Başarılı',
    preferences: 'Tercihler',
    theme: 'Tema',
    light: 'Açık',
    dark: 'Koyu',
    close: 'Kapat',
    page: 'Sayfa',
    references: 'Kaynaklar',
    reference: 'Kaynak',
    file: 'Dosya',
    openFile: 'Dosyayı Aç',
    showReference: 'Kaynağı Göster',
    content: 'İçerik',
    stop: 'Durdur',
    unknownFile: 'Bilinmeyen Dosya',
    progressAnalyzing: 'Analiz ediliyor',
    progressRetrieving: 'Veritabanı aranıyor',
    progressGenerating: 'Yanıt oluşturuluyor'
  }
};

export const LanguageProvider = ({ children }) => {
  const [language, setLanguage] = useState(localStorage.getItem('language') || 'en');

  const changeLanguage = (lang) => {
    setLanguage(lang);
    localStorage.setItem('language', lang);
  };

  const t = (key) => {
    return translations[language][key] || key;
  };

  return (
    <LanguageContext.Provider value={{ language, changeLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};
