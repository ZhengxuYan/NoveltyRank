"use client";

import { useState, useEffect, useMemo } from "react";
import { fetchAllPapers } from "@/lib/data";
import PaperRow from "@/components/PaperRow";
import AffiliationRow from "@/components/AffiliationRow";
import Filters from "@/components/Filters";
import {
  Loader2,
  ChevronLeft,
  ChevronRight,
  TrendingUp,
  Layers,
  GraduationCap,
  Building2,
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
  const [viewMode, setViewMode] = useState("papers"); // "papers" or "affiliations"
  const [currentPage, setCurrentPage] = useState(1);
  const [statusMsg, setStatusMsg] = useState("");

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

        const onStatus = (msg) => {
          setStatusMsg(msg);
        };

        await fetchAllPapers(onProgress, onStatus);
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
    allPapers.forEach((p) => {
      if (!p.primary_category) return;
      const cat = p.primary_category;
      if (!byCategory[cat]) byCategory[cat] = [];
      byCategory[cat].push(p);
    });

    // Sort and assign ranks
    const paperMap = new Map();

    Object.keys(byCategory).forEach((cat) => {
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
          percentile: percentile,
        });
      });
    });

    // Return papers in original order (mapped)
    return allPapers.map((p) => paperMap.get(p.arxiv_id) || p);
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

  // 2. Category Filter (NO SEARCH) - Used for Global Ranking
  const categoryFilteredPapers = useMemo(() => {
    let result = rankedPapers;

    // Filter by Primary Category if not "All"
    if (filters.category !== "All") {
      result = result.filter(
        (p) =>
          p.primary_category && p.primary_category.includes(filters.category)
      );
    }

    // Sort by Novelty Score
    result.sort((a, b) => b.novelty_score - a.novelty_score);

    return result;
  }, [rankedPapers, filters.category]);

  // 3. Search Filter - Used for Display
  const searchFilteredPapers = useMemo(() => {
    let result = categoryFilteredPapers;

    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter((p) => {
        const titleMatch = p.title?.toLowerCase().includes(query);
        const abstractMatch = p.abstract?.toLowerCase().includes(query);
        const authorMatch = p.authors?.toLowerCase().includes(query);
        const idMatch = p.arxiv_id?.toLowerCase().includes(query);
        const affMatch = p.author_affiliations?.toLowerCase().includes(query);
        return (
          titleMatch || abstractMatch || authorMatch || idMatch || affMatch
        );
      });
    }

    return result;
  }, [categoryFilteredPapers, searchQuery]);

  // 4. Affiliation Aggregation Logic
  const affiliationData = useMemo(() => {
    if (viewMode !== "affiliations") return [];

    const affMap = new Map();

    // Use categoryFilteredPapers to calculate GLOBAL RANKS (before search)
    categoryFilteredPapers.forEach((p) => {
      if (!p.author_affiliations) return;

      const affs = p.author_affiliations
        .split(";")
        .map((s) => s.trim())
        .filter((s) => s.length > 0);
      const uniqueAffs = [...new Set(affs)]; // Count each paper only once per affiliation

      uniqueAffs.forEach((affName) => {
        if (!affMap.has(affName)) {
          affMap.set(affName, {
            name: affName,
            papers: [],
            topAuthorCount: 0,
          });
        }

        const entry = affMap.get(affName);
        entry.papers.push(p);
        if (
          p.has_top_author === true ||
          String(p.has_top_author).toLowerCase() === "true"
        ) {
          entry.topAuthorCount += 1;
        }
      });
    });

    // Process stats
    let results = Array.from(affMap.values()).map((entry) => {
      // Sort papers by novelty score descending
      const sortedPapers = entry.papers.sort(
        (a, b) => b.novelty_score - a.novelty_score
      );
      
      // Take top 10 papers (or all if less than 10)
      const topPapers = sortedPapers.slice(0, 10);
      
      // Calculate sum of top 10 papers
      const top10Sum = topPapers.reduce((sum, p) => sum + p.novelty_score, 0);

      return {
        name: entry.name,
        paperCount: entry.papers.length,
        top10Sum: top10Sum,
        topAuthorCount: entry.topAuthorCount,
        // Store top papers for display
        topPapers: topPapers,
        // Keep all papers if needed for other purposes, but UI will use topPapers
        allPapers: sortedPapers, 
      };
    });

    // REMOVED Filter: > 5 papers logic
    results = results.filter((r) => r.paperCount > 2);

    // Sort by Top 10 Sum Descending (GLOBAL RANKING)
    results.sort((a, b) => b.top10Sum - a.top10Sum);

    // Add Global Rank and Percentile
    const totalAffiliations = results.length;
    results = results.map((aff, index) => {
      const rank = index + 1;
      // Percentile: Top X% (lower is better, e.g. Rank 1 is Top 0.1%)
      const percentile = ((rank / totalAffiliations) * 100).toFixed(1);
      return {
        ...aff,
        rank,
        percentile,
      };
    });

    // STRICT SEARCH FILTER FOR AFFILIATION NAMES (After Ranking)
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      results = results.filter((r) => r.name.toLowerCase().includes(query));
    }

    return results;
  }, [categoryFilteredPapers, viewMode, searchQuery]);

  // Specific Category List (when category is NOT All)
  const categoryPapers = useMemo(() => {
    if (filters.category === "All") return [];
    return searchFilteredPapers.filter(
      (p) => p.primary_category && p.primary_category.includes(filters.category)
    );
  }, [searchFilteredPapers, filters.category]);

  // Dashboard Data (when category IS All)
  const dashboardData = useMemo(() => {
    if (filters.category !== "All") return null;

    return DISPLAY_CATEGORIES.map((cat) => {
      const allCatPapers = searchFilteredPapers.filter(
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
  }, [searchFilteredPapers, filters.category, t]);

  // Pagination Logic (only for single category view OR affiliation view)
  const currentList =
    viewMode === "affiliations" ? affiliationData : categoryPapers;
  const totalPages = Math.ceil(currentList.length / ITEMS_PER_PAGE);
  const paginatedItems = currentList.slice(
    (currentPage - 1) * ITEMS_PER_PAGE,
    currentPage * ITEMS_PER_PAGE
  );

  useEffect(() => {
    setCurrentPage(1);
  }, [filters, searchQuery, viewMode]);

  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
  };

  const handleSearch = (query) => {
    setSearchQuery(query);
  };

  const handleViewModeChange = (mode) => {
    setViewMode(mode);
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
        <Filters
          onFilterChange={handleFilterChange}
          onSearch={handleSearch}
          onViewModeChange={handleViewModeChange}
          viewMode={viewMode}
        />

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
            {/* CASE 1: AFFILIATION LEADERBOARD */}
            {viewMode === "affiliations" ? (
              <>
                <div className="flex items-center justify-between mb-4 px-2">
                  <div className="flex items-center gap-2">
                    <Building2 className="w-4 h-4 text-indigo-400" />
                    <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
                      Top Affiliations by Novelty
                    </h2>
                    <span className="text-slate-600 text-sm font-medium ml-2">
                      {affiliationData.length} institutions
                    </span>
                  </div>
                </div>

                <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl shadow-black/50 ring-1 ring-white/5 animate-in fade-in duration-300">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse table-fixed">
                      <thead>
                        <tr className="bg-slate-900/50 border-b border-slate-800">
                          <th className="w-24 py-4 pl-6 text-xs font-semibold text-slate-500 uppercase tracking-wider text-center">
                            {t.dashboard.table.rank}
                          </th>
                          <th className="py-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                            Institution
                          </th>
                          <th className="w-32 py-4 pr-6 text-xs font-semibold text-slate-500 uppercase tracking-wider text-right">
                            Top 10 Score
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/50">
                        {paginatedItems.length > 0 ? (
                          paginatedItems.map((aff, index) => (
                            <AffiliationRow
                              key={aff.name}
                              affiliation={aff}
                              rank={aff.rank} // Global Rank
                              isSearching={!!searchQuery.trim()}
                            />
                          ))
                        ) : (
                          <tr>
                            <td colSpan="3" className="px-6 py-24 text-center">
                              <div className="flex flex-col items-center justify-center">
                                <p className="text-slate-400 text-lg font-medium mb-2">
                                  No affiliations found
                                </p>
                                <p className="text-slate-500 text-sm">
                                  Try adjusting your filters (min 5 papers
                                  required)
                                </p>
                              </div>
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            ) : /* Case 2: Overall Dashboard (Multiple Lists) - ONLY if filters.category === "All" */
            filters.category === "All" ? (
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
                        <table className="w-full text-left relative table-fixed">
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
              /* Case 3: Single Category List (Full View) */
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
                    <table className="w-full text-left border-collapse table-fixed">
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
                        {paginatedItems.length > 0 ? (
                          paginatedItems.map((paper, index) => (
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
              </>
            )}

            {/* Common Pagination for Case 1 (Affiliation) and Case 3 (Single Category) */}
            {(viewMode === "affiliations" || filters.category !== "All") &&
              paginatedItems.length > 0 && (
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
      </div>
    </div>
  );
}
