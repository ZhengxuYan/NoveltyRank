"use client";

import Link from 'next/link';
import Image from 'next/image';
import { useState, useRef, useEffect } from 'react';
import { Github, Globe, ChevronDown, Check } from 'lucide-react';
import { useLanguage } from '@/contexts/LanguageContext';

export default function Header() {
  const { t, language, setLanguage, availableLanguages } = useLanguage();
  const [isLangMenuOpen, setIsLangMenuOpen] = useState(false);
  const langMenuRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (langMenuRef.current && !langMenuRef.current.contains(event.target)) {
        setIsLangMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [langMenuRef]);

  const currentLang = availableLanguages.find(l => l.code === language) || availableLanguages[0];

  return (
    <header className="fixed w-full top-0 z-50 glass">
      <div className="w-full px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-8">
          <Link href="/" className="flex items-center gap-3 group">
            <div className="relative w-8 h-8">
              <Image 
                src="/logo.png" 
                alt="NoveltyRank Logo" 
                fill
                className="object-contain"
                priority
              />
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-lg text-white leading-none tracking-tight">Novelty<span className="text-indigo-400">Rank</span></span>
              <span className="text-[10px] font-medium text-slate-500 uppercase tracking-widest mt-0.5">AI Paper Discovery</span>
            </div>
          </Link>
          
          <nav className="hidden md:flex items-center gap-1 text-sm font-medium text-slate-400">
            <Link href="/" className="px-4 py-2 rounded-full text-white bg-white/5 hover:bg-white/10 transition-colors">{t.nav.leaderboard}</Link>
          </nav>
        </div>
        
        <div className="flex items-center gap-4">
          {/* Language Selector */}
          <div className="relative" ref={langMenuRef}>
            <button
              onClick={() => setIsLangMenuOpen(!isLangMenuOpen)}
              className="flex items-center gap-2 text-xs font-medium text-slate-400 hover:text-white transition-colors px-3 py-1.5 rounded-full hover:bg-white/5 border border-transparent hover:border-slate-800"
            >
              <Globe className="w-4 h-4" />
              <span className="hidden sm:inline">{currentLang.nativeName}</span>
              <ChevronDown className={`w-3 h-3 transition-transform ${isLangMenuOpen ? 'rotate-180' : ''}`} />
            </button>

            {isLangMenuOpen && (
              <div className="absolute right-0 mt-2 w-40 bg-slate-900 border border-slate-800 rounded-xl shadow-xl py-1 overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200">
                {availableLanguages.map((lang) => (
                  <button
                    key={lang.code}
                    onClick={() => {
                      setLanguage(lang.code);
                      setIsLangMenuOpen(false);
                    }}
                    className={`w-full px-4 py-2 text-xs text-left flex items-center justify-between hover:bg-slate-800 transition-colors
                      ${language === lang.code ? 'text-indigo-400 bg-indigo-500/10' : 'text-slate-400'}
                    `}
                  >
                    <span>{lang.nativeName}</span>
                    {language === lang.code && <Check className="w-3 h-3" />}
                  </button>
                ))}
              </div>
            )}
          </div>

           <a 
            href="https://huggingface.co/datasets/JasonYan777/novelty-ranked-preprints" 
            target="_blank" 
            rel="noopener noreferrer"
            className="hidden sm:flex items-center gap-2 text-xs font-medium text-slate-400 hover:text-white transition-colors border border-slate-800 hover:border-slate-700 px-4 py-2 rounded-full bg-slate-900/50"
          >
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            {t.nav.dataset}
          </a>
          <a
            href="https://github.com/ZhengxuYan/NoveltyRank"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 text-slate-400 hover:text-white transition-colors"
          >
            <Github className="w-5 h-5" />
          </a>
        </div>
      </div>
    </header>
  );
}
