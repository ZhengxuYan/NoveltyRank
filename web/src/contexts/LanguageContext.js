"use client";

import { createContext, useContext, useState, useEffect } from 'react';
import { translations } from '@/lib/translations';

const LanguageContext = createContext();

const AVAILABLE_LANGUAGES = [
  { code: 'en', label: 'English', nativeName: 'English' },
  { code: 'zh', label: 'Chinese', nativeName: '中文' },
  { code: 'es', label: 'Spanish', nativeName: 'Español' },
  { code: 'fr', label: 'French', nativeName: 'Français' },
  { code: 'de', label: 'German', nativeName: 'Deutsch' },
  { code: 'ja', label: 'Japanese', nativeName: '日本語' },
  { code: 'ko', label: 'Korean', nativeName: '한국어' },
];

export function LanguageProvider({ children }) {
  const [language, setLanguage] = useState('en');

  // Auto-detect browser language
  useEffect(() => {
    const browserLang = navigator.language.split('-')[0];
    const supportedCodes = AVAILABLE_LANGUAGES.map(l => l.code);
    
    if (supportedCodes.includes(browserLang)) {
      setLanguage(browserLang);
    }
  }, []);

  const t = translations[language] || translations['en'];

  // For backward compatibility, though direct setLanguage is preferred
  const toggleLanguage = () => {
    setLanguage(prev => prev === 'en' ? 'zh' : 'en');
  };

  return (
    <LanguageContext.Provider value={{ 
      language, 
      setLanguage, 
      toggleLanguage, // Keep for backward compat if needed, but Header will use setLanguage
      t,
      availableLanguages: AVAILABLE_LANGUAGES
    }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  return useContext(LanguageContext);
}
