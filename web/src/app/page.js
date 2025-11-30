"use client";

import { useState, useEffect, useMemo } from 'react';
import { fetchAllPapers } from '@/lib/data';
import PaperRow from '@/components/PaperRow';
import Filters from '@/components/Filters';
import { Loader2, ChevronLeft, ChevronRight, TrendingUp, Layers } from 'lucide-react';

const ITEMS_PER_PAGE = 50;

const DISPLAY_CATEGORIES = [
  { label: "Computer Vision", value: "cs.CV" },
  { label: "NLP", value: "cs.CL" },
  { label: "Machine Learning", value: "cs.LG" },
  { label: "AI", value: "cs.AI" },
  { label: "Robotics", value: "cs.RO" },
  { label: "Cryptography", value: "cs.CR" }
];

export default function Home() {
  const [papers, setPapers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [filters, setFilters] = useState({ category: "All", days: 30 });
  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);

  // Initial Data Fetch with Background Loading
  useEffect(() => {
    async function loadData() {
      try {
        const onProgress = (currentPapers) => {
          setPapers(currentPapers);
          setLoadingProgress(currentPapers.length);
          if (currentPapers.length > 0 && loading) {
            setLoading(false);
          }
        };

        await fetchAllPapers(onProgress);
        setLoading(false);
      } catch (error) {
        console.error("Failed to load papers", error);
        setLoading(false);
      }
    }
    loadData();
  }, []);

  // Filter Logic (Base filter without category if All)
  const baseFilteredPapers = useMemo(() => {
    let result = [...papers];

    // 1. Time Frame
    if (filters.days < 3650) {
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - filters.days);
      result = result.filter(p => new Date(p.published) >= cutoffDate);
    }

    // 2. Search
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter(p => {
        const titleMatch = p.title?.toLowerCase().includes(query);
        const abstractMatch = p.abstract?.toLowerCase().includes(query);
        const authorMatch = p.authors?.toLowerCase().includes(query);
        const idMatch = p.arxiv_id?.toLowerCase().includes(query);
        return titleMatch || abstractMatch || authorMatch || idMatch;
      });
    }
    
    // Sort by Novelty Score
    result.sort((a, b) => b.novelty_score - a.novelty_score);

    return result;
  }, [papers, filters.days, searchQuery]);

  // Specific Category List (when category is NOT All)
  const categoryPapers = useMemo(() => {
    if (filters.category === "All") return [];
    return baseFilteredPapers.filter(p => p.categories && p.categories.includes(filters.category));
  }, [baseFilteredPapers, filters.category]);

  // Dashboard Data (when category IS All)
  const dashboardData = useMemo(() => {
    if (filters.category !== "All") return null;
    
    return DISPLAY_CATEGORIES.map(cat => {
      const catPapers = baseFilteredPapers
        .filter(p => p.categories && p.categories.includes(cat.value))
        .slice(0, 20); // Top 20 for dashboard
      return { ...cat, papers: catPapers };
    });
  }, [baseFilteredPapers, filters.category]);


  // Pagination Logic (only for single category view)
  const totalPages = Math.ceil(categoryPapers.length / ITEMS_PER_PAGE);
  const paginatedPapers = categoryPapers.slice(
    (currentPage - 1) * ITEMS_PER_PAGE,
    currentPage * ITEMS_PER_PAGE
  );

  useEffect(() => {
    setCurrentPage(1);
  }, [filters, searchQuery]);

  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
  };

  const handleSearch = (query) => {
    setSearchQuery(query);
  };

  return (
    <div className="min-h-screen pt-24 pb-12 bg-slate-950">
      <div className="container mx-auto px-4 max-w-7xl">
        
        {/* Hero Section */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-white mb-4 tracking-tight">
            Discover the <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-emerald-400">Next Big Thing</span> in AI
          </h1>
          <p className="text-slate-400 text-lg max-w-2xl leading-relaxed">
            Our AI-powered ranking system analyzes thousands of daily arXiv preprints to identify the most novel and impactful research before it trends.
          </p>
        </div>

        {/* Filters & Search */}
        <Filters onFilterChange={handleFilterChange} onSearch={handleSearch} />

        {/* Loading State */}
        {loading && papers.length === 0 && (
          <div className="flex flex-col items-center justify-center py-32">
            <Loader2 className="w-10 h-10 text-indigo-500 animate-spin mb-4" />
            <p className="text-slate-400 text-sm font-medium animate-pulse">Analyzing research landscape...</p>
          </div>
        )}

        {/* Main Content */}
        {!loading && (
          <>
            {/* Case 1: Overall Dashboard (Multiple Lists) */}
            {filters.category === "All" ? (
              <div className="space-y-8">
                 <div className="flex items-center justify-between px-2">
                    <div className="flex items-center gap-2">
                      <Layers className="w-5 h-5 text-indigo-400" />
                      <h2 className="text-lg font-semibold text-slate-200">
                        Category Leaderboards
                      </h2>
                    </div>
                     {loadingProgress > 0 && loadingProgress < 7000 && (
                      <div className="flex items-center gap-2 text-xs text-indigo-400 bg-indigo-500/10 px-3 py-1 rounded-full border border-indigo-500/20">
                        <Loader2 className="w-3 h-3 animate-spin" />
                        Indexing... ({loadingProgress})
                      </div>
                    )}
                 </div>

                 <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in">
                   {dashboardData.map((categoryGroup) => (
                     <div key={categoryGroup.value + filters.days} className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg hover:border-slate-700 transition-colors flex flex-col h-[600px] animate-in slide-in-from-bottom-2 duration-300">
                       <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center shrink-0">
                         <h3 className="font-semibold text-slate-200">{categoryGroup.label}</h3>
                         <button 
                            onClick={() => setFilters(prev => ({ ...prev, category: categoryGroup.value }))}
                            className="text-xs text-indigo-400 hover:text-indigo-300 font-medium"
                         >
                           View All
                         </button>
                       </div>
                       
                       <div className="overflow-y-auto flex-1 custom-scrollbar">
                         <table className="w-full text-left relative">
                           <thead className="sticky top-0 bg-slate-900 z-10 text-xs font-semibold text-slate-500 uppercase tracking-wider border-b border-slate-800 shadow-sm">
                             <tr>
                               <th className="py-2 pl-4 text-center w-12 bg-slate-900">Rank</th>
                               <th className="py-2 px-2 bg-slate-900">Title</th>
                               <th className="py-2 pr-4 text-right w-20 bg-slate-900">Score</th>
                             </tr>
                           </thead>
                           <tbody className="divide-y divide-slate-800/50">
                             {categoryGroup.papers.length > 0 ? (
                               categoryGroup.papers.map((paper, index) => (
                                 <PaperRow 
                                   key={paper.arxiv_id} 
                                   paper={paper} 
                                   rank={index + 1}
                                   compact={true}
                                 />
                               ))
                             ) : (
                               <tr>
                                 <td className="px-6 py-8 text-center text-slate-500 text-sm">
                                   No papers found.
                                 </td>
                               </tr>
                             )}
                           </tbody>
                         </table>
                       </div>
                     </div>
                   ))}
                 </div>
              </div>
            ) : (
              /* Case 2: Single Category List (Full View) */
              <>
                <div className="flex items-center justify-between mb-4 px-2">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-emerald-400" />
                    <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
                      {filters.category === "All" ? "Top Novelty Rankings" : `${DISPLAY_CATEGORIES.find(c => c.value === filters.category)?.label || filters.category} Rankings`}
                    </h2>
                    <span className="text-slate-600 text-sm font-medium ml-2">
                      {categoryPapers.length} papers
                    </span>
                  </div>
                  
                  {loadingProgress > 0 && loadingProgress < 7000 && (
                     <div className="flex items-center gap-2 text-xs text-indigo-400 bg-indigo-500/10 px-3 py-1 rounded-full border border-indigo-500/20">
                       <Loader2 className="w-3 h-3 animate-spin" />
                       Indexing... ({loadingProgress})
                     </div>
                  )}
                </div>

                <div key={filters.days + filters.category} className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl shadow-black/50 ring-1 ring-white/5 animate-in fade-in duration-300">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="bg-slate-900/50 border-b border-slate-800">
                          <th className="w-16 py-4 pl-6 text-xs font-semibold text-slate-500 uppercase tracking-wider text-center">Rank</th>
                          <th className="py-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">Paper Details</th>
                          <th className="w-32 py-4 pr-6 text-xs font-semibold text-slate-500 uppercase tracking-wider text-right">Novelty Score</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/50">
                        {paginatedPapers.length > 0 ? (
                          paginatedPapers.map((paper, index) => (
                            <PaperRow 
                              key={paper.arxiv_id} 
                              paper={paper} 
                              rank={(currentPage - 1) * ITEMS_PER_PAGE + index + 1} 
                            />
                          ))
                        ) : (
                          <tr>
                            <td colSpan="3" className="px-6 py-24 text-center">
                              <div className="flex flex-col items-center justify-center">
                                <p className="text-slate-400 text-lg font-medium mb-2">No papers found</p>
                                <p className="text-slate-500 text-sm">Try adjusting your filters or search terms</p>
                              </div>
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Pagination Controls */}
                {categoryPapers.length > 0 && (
                  <div className="flex items-center justify-center gap-4 mt-8">
                    <button
                      onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                      disabled={currentPage === 1}
                      className="p-2.5 rounded-lg hover:bg-slate-800 disabled:opacity-30 disabled:hover:bg-transparent transition-all text-slate-400 border border-transparent hover:border-slate-700"
                    >
                      <ChevronLeft className="w-5 h-5" />
                    </button>
                    
                    <span className="text-sm text-slate-400 font-medium font-mono bg-slate-900 px-4 py-2 rounded-lg border border-slate-800">
                      Page {currentPage} <span className="text-slate-600">/</span> {totalPages}
                    </span>
                    
                    <button
                      onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                      disabled={currentPage === totalPages}
                      className="p-2.5 rounded-lg hover:bg-slate-800 disabled:opacity-30 disabled:hover:bg-transparent transition-all text-slate-400 border border-transparent hover:border-slate-700"
                    >
                      <ChevronRight className="w-5 h-5" />
                    </button>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
