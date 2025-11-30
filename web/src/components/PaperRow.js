"use client";

import { useState } from 'react';
import { ChevronDown, ChevronUp, Award, Calendar, ExternalLink, FileText, Medal } from 'lucide-react';
import { useLanguage } from '@/contexts/LanguageContext';

export default function PaperRow({ paper, rank, compact = false }) {
  const { t } = useLanguage();
  const [isExpanded, setIsExpanded] = useState(false);

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
      <tr className="group border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
        <td className="py-3 pl-4 w-12 text-center">
          <span className={`font-mono text-xs font-bold ${getRankStyle(paper.category_rank || rank)}`}>
            #{paper.category_rank || rank}
          </span>
        </td>
        <td className="py-3 px-2">
          <a 
            href={paper.url} 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-sm font-medium text-slate-200 hover:text-indigo-400 transition-colors line-clamp-2 block mb-1"
            title={paper.title}
          >
            {paper.title}
          </a>
           {acceptance && (
             <span className="inline-flex items-center gap-1 text-amber-500/90 text-[9px] font-medium">
               <Award className="w-2.5 h-2.5" />
               {acceptance}
             </span>
          )}
        </td>
        <td className="py-3 pr-4 text-right w-20">
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
    );
  }

  return (
    <>
      <tr 
        className={`group transition-all duration-200 border-b border-slate-800/50 hover:bg-slate-800/30 ${isExpanded ? 'bg-slate-800/30' : ''}`}
      >
        {/* Rank */}
        {/* Rank */}
        <td className="w-16 py-4 pl-6 text-center">
          <span className={`font-mono text-xl font-bold ${getRankStyle(paper.category_rank || rank)}`}>
            #{paper.category_rank || rank}
          </span>
        </td>

        {/* Title & Metadata */}
        <td className="py-4 pr-6">
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
            
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-slate-500">
              <span className="text-slate-400 font-medium truncate max-w-[400px]">
                {paper.authors}
              </span>
              <div className="flex items-center gap-1">
                <Calendar className="w-3 h-3" />
                <span>{new Date(paper.published).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}</span>
              </div>
              <div className="flex items-center gap-1 font-mono text-slate-600">
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
        </td>

        {/* Novelty Score */}
        <td className="w-32 py-4 pr-6 text-right">
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
