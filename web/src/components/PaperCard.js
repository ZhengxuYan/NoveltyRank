"use client";

import { useState } from 'react';
import { ChevronDown, ChevronUp, Award } from 'lucide-react';

export default function PaperCard({ paper }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="bg-white border border-gray-200 rounded-sm p-4 hover:shadow-sm transition-shadow h-full flex flex-col">
      {/* Title */}
      <h3 className="text-base font-medium leading-snug mb-1">
        <a 
          href={paper.url} 
          target="_blank" 
          rel="noopener noreferrer"
          className="text-[#0056b3] hover:underline"
        >
          {paper.title}
        </a>
      </h3>

      {/* Authors */}
      {paper.authors && (
        <div className="text-xs text-gray-600 mb-2">
          {paper.authors}
        </div>
      )}

      {/* Metadata */}
      <div className="flex flex-wrap items-center gap-2 mb-3 text-xs text-gray-500">
        <span>{new Date(paper.published).toLocaleDateString()}</span>
        {paper.acceptance_details && (
           <span className="text-green-700 font-medium">
             • {paper.acceptance_details}
           </span>
        )}
        <span className="bg-gray-100 px-1.5 py-0.5 rounded text-gray-600 border border-gray-200">
          Score: {paper.novelty_score.toFixed(2)}
        </span>
      </div>

      {/* Abstract Toggle */}
      <div className="mt-auto pt-2 border-t border-gray-100">
        <button 
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 transition-colors w-full justify-center"
        >
          {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </button>
        
        {isExpanded && (
          <p className="mt-2 text-xs text-gray-600 leading-relaxed border-t border-gray-100 pt-2">
            {paper.abstract}
          </p>
        )}
      </div>
    </div>
  );
}
