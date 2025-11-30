"use client";

import { useState } from 'react';
import { Search, Calendar, ChevronDown } from 'lucide-react';
import { useLanguage } from '@/contexts/LanguageContext';

const CATEGORIES = [
  { label: "Overall", value: "All" },
  { label: "Computer Vision", value: "cs.CV" },
  { label: "NLP", value: "cs.CL" },
  { label: "Machine Learning", value: "cs.LG" },
  { label: "AI", value: "cs.AI" },
  { label: "Robotics", value: "cs.RO" },
  { label: "Cryptography", value: "cs.CR" }
];

const TIME_FRAMES = [
  { labelKey: "threeDays", value: 3 },
  { labelKey: "week", value: 7 },
  { labelKey: "month", value: 30 },
  { labelKey: "all", value: 3650 },
];

export default function Filters({ onFilterChange, onSearch }) {
  const { t } = useLanguage();
  const [activeCategory, setActiveCategory] = useState("All");
  const [activeDays, setActiveDays] = useState(30);
  const [searchQuery, setSearchQuery] = useState("");

  const handleCategoryClick = (cat) => {
    setActiveCategory(cat);
    onFilterChange({ category: cat, days: activeDays });
  };

  const handleDaysChange = (e) => {
    const days = parseInt(e.target.value);
    setActiveDays(days);
    onFilterChange({ category: activeCategory, days });
  };

  const handleSearchChange = (e) => {
    setSearchQuery(e.target.value);
    onSearch(e.target.value);
  };

  return (
    <div className="flex flex-col gap-6 mb-8">
      {/* 1. Search Bar (Separate Row) */}
      <div className="relative w-full">
        <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
          <Search className="h-4 w-4 text-slate-500" />
        </div>
        <input
          type="text"
          placeholder={t.filters.searchPlaceholder}
          value={searchQuery}
          onChange={handleSearchChange}
          className="w-full pl-11 pr-4 py-3 bg-slate-900/50 border border-slate-800/50 rounded-xl text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500/50 transition-all backdrop-blur-sm"
        />
      </div>

      {/* 2. Ranking Controls: Categories & Time Frame */}
      <div className="flex flex-col md:flex-row justify-between items-end md:items-center gap-4 border-b border-slate-800 pb-1">
        
        {/* Category Tabs */}
        <nav className="flex space-x-6 overflow-x-auto no-scrollbar w-full md:w-auto" aria-label="Tabs">
          {CATEGORIES.map((cat) => (
            <button
              key={cat.value}
              onClick={() => handleCategoryClick(cat.value)}
              className={`
                whitespace-nowrap pb-3 px-1 border-b-2 font-medium text-sm transition-all duration-200
                ${activeCategory === cat.value
                  ? 'border-indigo-500 text-indigo-400'
                  : 'border-transparent text-slate-500 hover:text-slate-300 hover:border-slate-700'}
              `}
            >
              {t.filters.categories[cat.value]}
            </button>
          ))}
        </nav>

        {/* Time Frame Selector (Right Aligned) */}
        <div className="relative w-full md:w-auto min-w-[200px] shrink-0 mb-1">
           <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Calendar className="h-3.5 w-3.5 text-slate-500" />
          </div>
          <select
            value={activeDays}
            onChange={handleDaysChange}
            className="w-full pl-9 pr-8 py-2 bg-slate-900 border border-slate-700 text-slate-300 text-xs font-medium rounded-lg focus:ring-1 focus:ring-indigo-500/50 focus:border-indigo-500 block appearance-none transition-all cursor-pointer hover:bg-slate-800"
          >
            {TIME_FRAMES.map((tf) => (
              <option key={tf.value} value={tf.value}>{t.filters.timeFrames[tf.labelKey]}</option>
            ))}
          </select>
          <div className="absolute inset-y-0 right-0 pr-2.5 flex items-center pointer-events-none">
            <ChevronDown className="h-3.5 w-3.5 text-slate-500" />
          </div>
        </div>

      </div>
    </div>
  );
}
