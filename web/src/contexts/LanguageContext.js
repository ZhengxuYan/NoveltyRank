"use client";

import { createContext, useContext, useState } from 'react';
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
  const [language, setLanguage] = useState(() => {
    if (typeof navigator === 'undefined') return 'en';
    const browserLang = navigator.language.split('-')[0];
    const supportedCodes = AVAILABLE_LANGUAGES.map(l => l.code);
    return supportedCodes.includes(browserLang) ? browserLang : 'en';
  });

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
