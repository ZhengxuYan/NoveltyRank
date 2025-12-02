"use client";

import { LanguageProvider } from '@/contexts/LanguageContext';

export function Providers({ children }) {
  return (
    <LanguageProvider>
      {children}
    </LanguageProvider>
  );
}




