"use client";

import { useState, useEffect, useMemo } from "react";
import { fetchAllPapers } from "@/lib/data";
import PaperRow from "@/components/PaperRow";
import Filters from "@/components/Filters";
import {
  Loader2,
  ChevronLeft,
  ChevronRight,
  TrendingUp,
  Layers,
  GraduationCap,
} from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";

const ITEMS_PER_PAGE = 50;

const DISPLAY_CATEGORIES = [
  { label: "Computer Vision", value: "cs.CV" },
  { label: "NLP", value: "cs.CL" },
  { label: "Machine Learning", value: "cs.LG" },
  { label: "AI", value: "cs.AI" },
  { label: "Robotics", value: "cs.RO" },
  { label: "Cryptography", value: "cs.CR" },
];

export default function Home() {
  const { t } = useLanguage();
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

  // Helper to calculate ranks and percentiles
  const calculateRanks = (allPapers) => {
    // Group by category
    const byCategory = {};
    allPapers.forEach(p => {
      if (!p.primary_category) return;
      const cat = p.primary_category; 
      if (!byCategory[cat]) byCategory[cat] = [];
      byCategory[cat].push(p);
    });

    // Sort and assign ranks
    const paperMap = new Map();
    
    Object.keys(byCategory).forEach(cat => {
      // Sort by novelty score descending
      byCategory[cat].sort((a, b) => b.novelty_score - a.novelty_score);
      
      const total = byCategory[cat].length;
      byCategory[cat].forEach((p, index) => {
        const rank = index + 1;
        const percentile = ((1 - (rank - 1) / total) * 100).toFixed(1);
        
        paperMap.set(p.arxiv_id, {
          ...p,
          category_rank: rank,
          category_total: total,
          percentile: percentile
        });
      });
    });

    // Return papers in original order (mapped)
    return allPapers.map(p => paperMap.get(p.arxiv_id) || p);
  };

  // 1. Time Filter & Ranking
  const rankedPapers = useMemo(() => {
    let result = [...papers];

    // Filter by Time FIRST
    if (filters.days < 3650) {
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - filters.days);
      result = result.filter((p) => new Date(p.published) >= cutoffDate);
    }

    // Then Calculate Ranks on the time-filtered set
    return calculateRanks(result);
  }, [papers, filters.days]);

  // 2. Category & Search Filter
  const baseFilteredPapers = useMemo(() => {
    let result = rankedPapers;

    // Filter by Primary Category if not "All"
    if (filters.category !== "All") {
      result = result.filter(
        (p) =>
          p.primary_category && p.primary_category.includes(filters.category)
      );
    }

    // Search
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter((p) => {
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
  }, [rankedPapers, filters.category, searchQuery]);

  // Specific Category List (when category is NOT All)
  const categoryPapers = useMemo(() => {
    if (filters.category === "All") return [];
    return baseFilteredPapers.filter(
      (p) => p.primary_category && p.primary_category.includes(filters.category)
    );
  }, [baseFilteredPapers, filters.category]);

  // Dashboard Data (when category IS All)
  const dashboardData = useMemo(() => {
    if (filters.category !== "All") return null;

    return DISPLAY_CATEGORIES.map((cat) => {
      const allCatPapers = baseFilteredPapers.filter(
        (p) => p.primary_category && p.primary_category.includes(cat.value)
      );
      const topPapers = allCatPapers.slice(0, 20); // Top 20 for dashboard

      return {
        ...cat,
        label: t.filters.categories[cat.value], // Use translated label
        papers: topPapers,
        count: allCatPapers.length,
      };
    }).sort((a, b) => b.count - a.count); // Sort categories by paper count (descending)
  }, [baseFilteredPapers, filters.category, t]);

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
      <div className="w-full px-6">
        {/* Hero Section */}
        <div className="mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-900/20 border border-red-900/30 text-red-400 text-xs font-medium mb-6">
            <GraduationCap className="w-3 h-3" />
            {t.hero.badge}
          </div>
          <h1 className="text-4xl font-bold text-white mb-4 tracking-tight">
            {t.hero.titlePrefix}{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-emerald-400">
              {t.hero.titleHighlight}
            </span>{" "}
            {t.hero.titleSuffix}
          </h1>
          <p className="text-slate-400 text-lg max-w-2xl leading-relaxed mb-6">
            {t.hero.description}
          </p>
        </div>

        {/* Filters & Search */}
        <Filters onFilterChange={handleFilterChange} onSearch={handleSearch} />

        {/* Loading State */}
        {loading && papers.length === 0 && (
          <div className="flex flex-col items-center justify-center py-32">
            <Loader2 className="w-10 h-10 text-indigo-500 animate-spin mb-4" />
            <p className="text-slate-400 text-sm font-medium animate-pulse">
              {t.dashboard.loading}
            </p>
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
                      {t.dashboard.leaderboards}
                    </h2>
                  </div>
                  {loadingProgress > 0 && loadingProgress < 7000 && (
                    <div className="flex items-center gap-2 text-xs text-indigo-400 bg-indigo-500/10 px-3 py-1 rounded-full border border-indigo-500/20">
                      <Loader2 className="w-3 h-3 animate-spin" />
                      {t.dashboard.indexing} ({loadingProgress})
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in">
                  {dashboardData.map((categoryGroup) => (
                    <div
                      key={categoryGroup.value + filters.days}
                      className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg hover:border-slate-700 transition-colors flex flex-col h-[600px] animate-in slide-in-from-bottom-2 duration-300"
                    >
                      <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center shrink-0">
                        <div className="flex items-center gap-3">
                          <h3 className="font-semibold text-slate-200">
                            {categoryGroup.label}
                          </h3>
                          <span className="text-xs font-medium text-slate-500 bg-slate-800 px-2 py-0.5 rounded-full border border-slate-700">
                            {categoryGroup.count} {t.dashboard.papers}
                          </span>
                        </div>
                        <button
                          onClick={() =>
                            setFilters((prev) => ({
                              ...prev,
                              category: categoryGroup.value,
                            }))
                          }
                          className="text-xs text-indigo-400 hover:text-indigo-300 font-medium"
                        >
                          {t.dashboard.viewAll}
                        </button>
                      </div>

                      <div className="overflow-y-auto flex-1 custom-scrollbar">
                        <table className="w-full text-left relative">
                          <thead className="sticky top-0 bg-slate-900 z-10 text-xs font-semibold text-slate-500 uppercase tracking-wider border-b border-slate-800 shadow-sm">
                            <tr>
                              <th className="py-2 pl-4 text-center w-12 bg-slate-900">
                                {t.dashboard.table.rank}
                              </th>
                              <th className="py-2 px-2 bg-slate-900">
                                {t.dashboard.table.title}
                              </th>
                              <th className="py-2 pr-4 text-right w-20 bg-slate-900">
                                {t.dashboard.table.scoreShort}
                              </th>
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
                                  {t.dashboard.noPapers}.
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
                      {filters.category === "All"
                        ? t.dashboard.topRankings
                        : `${
                            t.filters.categories[filters.category] ||
                            filters.category
                          } ${t.dashboard.rankingsSuffix}`}
                    </h2>
                    <span className="text-slate-600 text-sm font-medium ml-2">
                      {categoryPapers.length} {t.dashboard.papers}
                    </span>
                  </div>

                  {loadingProgress > 0 && loadingProgress < 7000 && (
                    <div className="flex items-center gap-2 text-xs text-indigo-400 bg-indigo-500/10 px-3 py-1 rounded-full border border-indigo-500/20">
                      <Loader2 className="w-3 h-3 animate-spin" />
                      {t.dashboard.indexing} ({loadingProgress})
                    </div>
                  )}
                </div>

                <div
                  key={filters.days + filters.category}
                  className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl shadow-black/50 ring-1 ring-white/5 animate-in fade-in duration-300"
                >
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="bg-slate-900/50 border-b border-slate-800">
                          <th className="w-16 py-4 pl-6 text-xs font-semibold text-slate-500 uppercase tracking-wider text-center">
                            {t.dashboard.table.rank}
                          </th>
                          <th className="py-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                            {t.dashboard.table.details}
                          </th>
                          <th className="w-32 py-4 pr-6 text-xs font-semibold text-slate-500 uppercase tracking-wider text-right">
                            {t.dashboard.table.score}
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/50">
                        {paginatedPapers.length > 0 ? (
                          paginatedPapers.map((paper, index) => (
                            <PaperRow
                              key={paper.arxiv_id}
                              paper={paper}
                              rank={
                                (currentPage - 1) * ITEMS_PER_PAGE + index + 1
                              }
                            />
                          ))
                        ) : (
                          <tr>
                            <td colSpan="3" className="px-6 py-24 text-center">
                              <div className="flex flex-col items-center justify-center">
                                <p className="text-slate-400 text-lg font-medium mb-2">
                                  {t.dashboard.noPapers}
                                </p>
                                <p className="text-slate-500 text-sm">
                                  {t.dashboard.noPapersSub}
                                </p>
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
                      onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                      disabled={currentPage === 1}
                      className="p-2.5 rounded-lg hover:bg-slate-800 disabled:opacity-30 disabled:hover:bg-transparent transition-all text-slate-400 border border-transparent hover:border-slate-700"
                    >
                      <ChevronLeft className="w-5 h-5" />
                    </button>

                    <span className="text-sm text-slate-400 font-medium font-mono bg-slate-900 px-4 py-2 rounded-lg border border-slate-800">
                      {t.dashboard.pagination.page} {currentPage}{" "}
                      <span className="text-slate-600">/</span> {totalPages}
                    </span>

                    <button
                      onClick={() =>
                        setCurrentPage((p) => Math.min(totalPages, p + 1))
                      }
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
