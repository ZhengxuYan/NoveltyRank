"use client";

import { useState } from 'react';
import { ChevronDown, ChevronUp, Award, Calendar, ExternalLink, FileText, Medal, Building2, Star, TrendingUp } from 'lucide-react';
import { useLanguage } from '@/contexts/LanguageContext';

export default function PaperRow({ paper, rank, compact = false }) {
  const { t } = useLanguage();
  const [isExpanded, setIsExpanded] = useState(false);

  // Parse affiliations
  const affiliations = paper.author_affiliations 
    ? paper.author_affiliations.split(';').map(s => s.trim()).filter(s => s && s.length > 0)
    : [];
  const uniqueAffiliations = [...new Set(affiliations)];
  
  // Parse metrics
  const hasTopAuthor = paper.has_top_author === true || String(paper.has_top_author).toLowerCase() === "true";
  const maxHIndex = paper.max_h_index ? Math.round(Number(paper.max_h_index)) : null;

  // Novelty Score Color - using gradients/glows
  const getScoreStyle = (score) => {
    if (score >= 2.0) return "text-emerald-400 bg-emerald-400/10 border-emerald-400/20";
    if (score >= 1.0) return "text-indigo-400 bg-indigo-400/10 border-indigo-400/20";
    return "text-slate-400 bg-slate-400/10 border-slate-400/20";
  };

  // Rank Style - Gold, Silver, Bronze text
  const getRankStyle = (r) => {
    if (r === 1) return "text-yellow-400 drop-shadow-[0_0_8px_rgba(250,204,21,0.5)]";
    if (r === 2) return "text-slate-300 drop-shadow-[0_0_8px_rgba(203,213,225,0.5)]";
    if (r === 3) return "text-amber-600 drop-shadow-[0_0_8px_rgba(217,119,6,0.5)]";
    return "text-slate-500";
  };


  // Clean acceptance details
  const cleanAcceptanceDetails = (details) => {
    if (!details) return null;
    return details
      .replace(/Conference match:\s*/i, '')
      .replace(/Comment match:\s*/i, '')
      .trim();
  };

  const acceptance = cleanAcceptanceDetails(paper.acceptance_details);

  if (compact) {
    return (
      <>
        <tr className={`group border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors ${isExpanded ? 'bg-slate-800/30' : ''}`}>
          <td className="py-3 pl-4 w-12 text-center align-top">
            <span className={`font-mono text-xs font-bold ${getRankStyle(paper.category_rank || rank)}`}>
              #{paper.category_rank || rank}
            </span>
          </td>
          <td className="py-3 px-2 overflow-hidden align-top">
            <div className="flex items-start justify-between gap-2">
              <a 
                href={paper.url} 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-sm font-medium text-slate-200 hover:text-indigo-400 transition-colors line-clamp-2 block mb-1"
                title={paper.title}
              >
                {paper.title}
              </a>
              <button 
                onClick={() => setIsExpanded(!isExpanded)}
                className="text-slate-500 hover:text-indigo-400 p-0.5 rounded-md hover:bg-indigo-500/10 transition-colors shrink-0"
              >
                {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              </button>
            </div>

            {/* Authors */}
            <div className="text-xs text-slate-400 font-medium truncate mb-1" title={paper.authors}>
              {paper.authors}
            </div>

            {/* Affiliations */}
            {uniqueAffiliations.length > 0 && (
               <div className="flex items-center gap-1.5 text-[10px] text-slate-400 mb-1.5 w-full" title={uniqueAffiliations.join('\n')}>
                 <Building2 className="w-2.5 h-2.5 shrink-0 text-slate-500" />
                 <span className="truncate">
                   {uniqueAffiliations.join(', ')}
                 </span>
               </div>
            )}

            <div className="flex items-center gap-2 flex-wrap">
              {acceptance && (
                <span className="inline-flex items-center gap-1 text-amber-500/90 text-[9px] font-medium">
                  <Award className="w-2.5 h-2.5" />
                  {acceptance}
                </span>
              )}
              {hasTopAuthor && (
                <span className="inline-flex items-center gap-0.5 text-amber-400 text-[9px] font-medium" title={`Top Author${maxHIndex ? ` (H-Index: ${maxHIndex})` : ''}`}>
                  <Star className="w-2.5 h-2.5 fill-amber-400" />
                  <span className="hidden sm:inline">Top Author</span>
                  {maxHIndex && <span className="font-mono opacity-80 ml-0.5 text-[9px]">{maxHIndex}</span>}
                </span>
              )}
              {/* Date - Minimal */}
              <span className="flex items-center gap-1 text-[10px] text-slate-400">
                 <Calendar className="w-2.5 h-2.5" />
                 {new Date(paper.published).toLocaleDateString(undefined, { month: 'numeric', day: 'numeric' })}
               </span>
            </div>
          </td>
          <td className="py-3 pr-4 text-right w-20 align-top">
            <div className="flex flex-col items-end">
              <span className={`font-mono text-xs font-bold ${paper.novelty_score >= 2.0 ? "text-emerald-400" : paper.novelty_score >= 1.0 ? "text-indigo-400" : "text-slate-400"}`}>
                {paper.novelty_score.toFixed(2)}
              </span>
              {paper.percentile && (
                <span className="text-[9px] text-slate-500 font-medium">
                  Top {100 - paper.percentile < 1 ? '<1' : Math.round(100 - paper.percentile)}%
                </span>
              )}
            </div>
          </td>
        </tr>
        
        {/* Expanded Abstract Row (Compact View) */}
        {isExpanded && (
          <tr className="bg-slate-800/20 border-b border-slate-800/50 animate-in fade-in slide-in-from-top-2 duration-200">
            <td colSpan="3" className="px-4 py-3">
              <div className="pl-8 relative">
                <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-slate-800"></div>
                <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1">{t.dashboard.table.abstract}</h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {paper.abstract}
                </p>
                <div className="mt-2 flex justify-end">
                  <a 
                     href={paper.url}
                     target="_blank"
                     rel="noopener noreferrer"
                     className="flex items-center gap-1 text-[10px] font-medium text-indigo-400 hover:text-indigo-300"
                  >
                    {t.dashboard.table.viewPaper} <ExternalLink className="w-2.5 h-2.5" />
                  </a>
                </div>
              </div>
            </td>
          </tr>
        )}
      </>
    );
  }

  return (
    <>
      <tr 
        className={`group transition-all duration-200 border-b border-slate-800/50 hover:bg-slate-800/30 ${isExpanded ? 'bg-slate-800/30' : ''}`}
      >
        {/* Rank */}
        {/* Rank */}
        <td className="w-16 py-4 pl-6 text-center align-top">
          <span className={`font-mono text-xl font-bold inline-block mt-0.5 ${getRankStyle(paper.category_rank || rank)}`}>
            #{paper.category_rank || rank}
          </span>
        </td>

        {/* Title & Metadata */}
        <td className="py-4 pr-6 align-top overflow-hidden">
          <div className="flex flex-col gap-1.5">
            <div className="flex items-start justify-between gap-4">
              <a 
                href={paper.url} 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-[15px] font-semibold text-slate-200 hover:text-indigo-400 transition-colors leading-snug line-clamp-2"
              >
                {paper.title}
              </a>
              <button 
                onClick={() => setIsExpanded(!isExpanded)}
                className="text-slate-500 hover:text-indigo-400 p-1 rounded-md hover:bg-indigo-500/10 transition-colors shrink-0"
              >
                {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
            </div>
            
            <div className="flex flex-col gap-1.5">
              <div className="flex items-start justify-between gap-4">
                 {/* ... */}
              </div>

              {/* Authors - Dedicated Line */}
              <div className="text-xs text-slate-400 font-medium truncate" title={paper.authors}>
                {paper.authors}
              </div>

              {/* Affiliations - Dedicated Line */}
              {uniqueAffiliations.length > 0 && (
                <div className="flex items-center gap-1.5 text-xs text-slate-400 w-full" title={uniqueAffiliations.join('\n')}>
                  <Building2 className="w-3 h-3 shrink-0 text-slate-500" />
                  <span className="truncate w-full text-slate-400">
                    {uniqueAffiliations.join(', ')} 
                  </span>
                </div>
              )}

              {/* Metadata Line */}
              <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400 mt-0.5">
                {/* Top Author Badge */}
                {hasTopAuthor && (
                  <div className="flex items-center gap-1.5 text-amber-400 bg-amber-400/10 px-2 py-0.5 rounded-full border border-amber-400/20 shadow-[0_0_8px_rgba(251,191,36,0.1)]" title={`Max H-Index: ${maxHIndex}`}>
                    <Star className="w-3 h-3 fill-amber-400" />
                    <span className="text-[10px] font-bold tracking-wide">{t.dashboard.table.topAuthor || "TOP AUTHOR"}</span>
                    {maxHIndex && <span className="text-[10px] font-mono opacity-80 border-l border-amber-400/30 pl-1.5 ml-0.5">{maxHIndex}</span>}
                  </div>
                )}

                <div className="flex items-center gap-1">
                  <Calendar className="w-3 h-3" />
                  <span>{new Date(paper.published).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}</span>
                </div>
                
                <div className="flex items-center gap-1 font-mono text-slate-500">
                  <FileText className="w-3 h-3" />
                  {paper.arxiv_id}
                </div>
                
                {acceptance && (
                   <span className="flex items-center gap-1 text-amber-500/90 bg-amber-500/10 px-2 py-0.5 rounded-full text-[10px] font-medium border border-amber-500/20">
                     <Award className="w-3 h-3" />
                     {acceptance}
                   </span>
                )}
              </div>
            </div>
          </div>
        </td>

        {/* Novelty Score */}
        <td className="w-32 py-4 pr-6 text-right align-top">
          <div className={`inline-flex items-center justify-center px-3 py-1 rounded-full border font-mono text-xs font-bold ${getScoreStyle(paper.novelty_score)}`}>
            {paper.novelty_score.toFixed(2)}
          </div>
          {paper.percentile && (
            <div className="mt-2 text-xs text-slate-500 font-medium">
              Top {100 - paper.percentile < 1 ? '<1' : Math.round(100 - paper.percentile)}%
            </div>
          )}
        </td>
      </tr>
      
      {/* Expanded Abstract Row */}
      {isExpanded && (
        <tr className="bg-slate-800/20 border-b border-slate-800/50 animate-in fade-in slide-in-from-top-2 duration-200">
          <td colSpan="3" className="px-6 py-4">
            <div className="pl-16 relative">
              <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-slate-800"></div>
              <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">{t.dashboard.table.abstract}</h4>
              <p className="text-sm text-slate-300 leading-relaxed max-w-4xl">
                {paper.abstract}
              </p>
              
              {/* Full Affiliations List */}
              {uniqueAffiliations.length > 0 && (
                <div className="mt-4 pb-2 border-b border-slate-800/50">
                  <h5 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-2">
                    <Building2 className="w-3 h-3" /> {t.dashboard.table.affiliations || "Author Affiliations"}
                  </h5>
                  <div className="flex flex-wrap gap-x-6 gap-y-2">
                    {uniqueAffiliations.map((affil, idx) => (
                      <span key={idx} className="text-xs text-slate-400 font-medium">
                        {affil}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="mt-4 flex flex-wrap gap-2">
                {paper.categories.split(', ').map(cat => (
                  <span key={cat} className="text-[10px] font-medium text-slate-400 bg-slate-800 px-2 py-1 rounded-md border border-slate-700 hover:border-slate-600 transition-colors cursor-default">
                    {cat}
                  </span>
                ))}
                <a 
                   href={paper.url}
                   target="_blank"
                   rel="noopener noreferrer"
                   className="ml-auto flex items-center gap-1 text-xs font-medium text-indigo-400 hover:text-indigo-300"
                >
                  {t.dashboard.table.viewPaper} <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
