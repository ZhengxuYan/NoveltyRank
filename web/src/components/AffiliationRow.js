"use client";

import { useState } from 'react';
import { ChevronDown, ChevronUp, Building2, Trophy, Star } from 'lucide-react';
import { useLanguage } from '@/contexts/LanguageContext';

export default function AffiliationRow({ affiliation, rank, isSearching }) {
  const { t } = useLanguage();
  const [isExpanded, setIsExpanded] = useState(false);

  // Rank Style
  const getRankStyle = (r) => {
    if (r === 1) return "text-yellow-400 drop-shadow-[0_0_8px_rgba(250,204,21,0.5)]";
    if (r === 2) return "text-slate-300 drop-shadow-[0_0_8px_rgba(203,213,225,0.5)]";
    if (r === 3) return "text-amber-600 drop-shadow-[0_0_8px_rgba(217,119,6,0.5)]";
    return "text-slate-500";
  };

  const getScoreStyle = (score) => {
    if (score >= 2.0) return "text-emerald-400 bg-emerald-400/10 border-emerald-400/20";
    if (score >= 1.0) return "text-indigo-400 bg-indigo-400/10 border-indigo-400/20";
    if (score < 0) return "text-rose-400 bg-rose-400/10 border-rose-400/20";
    return "text-slate-400 bg-slate-400/10 border-slate-400/20";
  };

  return (
    <>
      <tr 
        className={`group transition-all duration-200 border-b border-slate-800/50 hover:bg-slate-800/30 ${isExpanded ? 'bg-slate-800/30' : ''}`}
      >
        <td className="w-24 py-4 pl-6 text-center align-top">
          <span className={`font-mono text-xl font-bold inline-block mt-0.5 ${getRankStyle(rank)}`}>
            #{rank}
          </span>
        </td>

        <td className="py-4 pr-6 align-top">
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3 min-w-0">
                <Building2 className="w-5 h-5 text-indigo-400 shrink-0" />
                <h3 className="text-[15px] font-semibold text-slate-200 leading-snug truncate" title={affiliation.name}>
                  {affiliation.name}
                </h3>
              </div>
              <button 
                onClick={() => setIsExpanded(!isExpanded)}
                className="text-slate-500 hover:text-indigo-400 p-1 rounded-md hover:bg-indigo-500/10 transition-colors shrink-0"
              >
                {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
            </div>

            <div className="flex items-center gap-4 text-xs text-slate-500">
              <span className="flex items-center gap-1.5 bg-slate-800/50 px-2.5 py-1 rounded-md border border-slate-800">
                <Trophy className="w-3 h-3 text-yellow-500" />
                <span className="text-slate-300 font-medium">{affiliation.paperCount}</span> Papers
              </span>
              {isSearching && (
                <span className="flex items-center gap-1.5 bg-slate-800/50 px-2.5 py-1 rounded-md border border-slate-800">
                  <span className="text-slate-400">Top</span>
                  <span className="text-emerald-400 font-medium">{affiliation.percentile}%</span>
                </span>
              )}
            </div>
          </div>
        </td>

        <td className="w-32 py-4 pr-6 text-right align-top">
          <div className={`inline-flex items-center justify-center px-3 py-1 rounded-full border font-mono text-xs font-bold ${getScoreStyle(affiliation.top10Sum / 10)}`}>
            {affiliation.top10Sum.toFixed(1)}
          </div>
          <div className="mt-1 text-[10px] text-slate-500 font-medium">Top 10 Score</div>
        </td>
      </tr>

      {/* Expanded View: Top Papers */}
      {isExpanded && (
        <tr className="bg-slate-800/20 border-b border-slate-800/50 animate-in fade-in slide-in-from-top-2 duration-200">
          <td colSpan="3" className="px-6 py-6">
            <div className="pl-16 relative">
              <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-slate-800"></div>
              <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">Top 10 Papers (used for ranking)</h4>
              <div className="space-y-3 max-h-[400px] overflow-y-auto pr-4 custom-scrollbar">
                {affiliation.topPapers.map((paper, idx) => (
                  <div key={paper.arxiv_id} className="flex items-start gap-3 group/paper">
                    <span className="font-mono text-xs text-slate-600 mt-0.5 w-4 shrink-0">#{idx + 1}</span>
                    <div className="flex-1 min-w-0">
                      <a 
                        href={paper.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm font-medium text-slate-300 hover:text-indigo-400 transition-colors truncate block"
                        title={paper.title}
                      >
                        {paper.title}
                      </a>
                      <div className="flex items-center gap-2 mt-1">
                         <span className={`font-mono text-[10px] font-bold ${getScoreStyle(paper.novelty_score)} px-1.5 py-0.5 rounded border shrink-0`}>
                            {paper.novelty_score.toFixed(2)}
                         </span>
                         <span className="text-[10px] text-slate-500 truncate max-w-[300px]" title={paper.authors}>{paper.authors}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

