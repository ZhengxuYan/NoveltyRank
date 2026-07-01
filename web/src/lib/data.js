import { paperSnapshot, SNAPSHOT_GENERATED_AT } from "./paperSnapshot.js";

const DEFAULT_PAGE_SIZE = 100;
const MAX_PAGES = 120;
const PAPER_CACHE_KEY = "noveltyrank-paper-cache-v1";
const CACHE_PAPER_LIMIT = 1000;
const ABSTRACT_CACHE_LIMIT = 700;

export function normalizePaperRows(rows = []) {
  return rows
    .map((item) => item?.row ?? item)
    .filter((paper) => paper && paper.arxiv_id && paper.title)
    .map((paper) => ({
      ...paper,
      novelty_score: Number(paper.novelty_score ?? 0),
      max_similarity:
        paper.max_similarity == null ? null : Number(paper.max_similarity),
      avg_similarity:
        paper.avg_similarity == null ? null : Number(paper.avg_similarity),
    }));
}

export function getLatestPaperDate(papers = []) {
  const dates = papers
    .map((paper) => new Date(paper.published))
    .filter((date) => !Number.isNaN(date.getTime()));

  if (dates.length === 0) return null;

  return new Date(Math.max(...dates.map((date) => date.getTime())));
}

function createStatus(source, papers, extra = {}) {
  const latestDate = getLatestPaperDate(papers);

  return {
    source,
    isStale: source !== "live",
    count: papers.length,
    latestDate: latestDate ? latestDate.toISOString() : null,
    snapshotGeneratedAt: SNAPSHOT_GENERATED_AT,
    ...extra,
  };
}

function getDefaultStorage() {
  if (typeof window === "undefined") return null;
  return window.localStorage ?? null;
}

function compactPaperForCache(paper) {
  return {
    title: paper.title,
    abstract: String(paper.abstract ?? "").slice(0, ABSTRACT_CACHE_LIMIT),
    categories: paper.categories,
    primary_category: paper.primary_category,
    published: paper.published,
    arxiv_id: paper.arxiv_id,
    url: paper.url,
    novelty_score: paper.novelty_score,
    max_similarity: paper.max_similarity,
    avg_similarity: paper.avg_similarity,
    is_accepted: paper.is_accepted,
    acceptance_details: paper.acceptance_details,
    authors: paper.authors,
    num_authors: paper.num_authors,
    max_h_index: paper.max_h_index,
    has_top_author: paper.has_top_author,
    author_affiliations: paper.author_affiliations,
  };
}

function readCachedPapers(storage) {
  if (!storage) return [];

  try {
    const raw = storage.getItem(PAPER_CACHE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return normalizePaperRows(parsed.papers);
  } catch {
    storage.removeItem?.(PAPER_CACHE_KEY);
    return [];
  }
}

function writeCachedPapers(storage, papers) {
  if (!storage || papers.length === 0) return;

  try {
    const compactPapers = papers
      .slice(0, CACHE_PAPER_LIMIT)
      .map(compactPaperForCache);
    storage.setItem(
      PAPER_CACHE_KEY,
      JSON.stringify({
        savedAt: new Date().toISOString(),
        papers: compactPapers,
      })
    );
  } catch {
    // Cache writes are best effort; rendering should never depend on storage.
  }
}

export async function fetchAllPapers(
  onProgress,
  onStatus,
  {
    fetchImpl = globalThis.fetch,
    pageSize = DEFAULT_PAGE_SIZE,
    snapshot = paperSnapshot,
    storage = getDefaultStorage(),
  } = {}
) {
  if (!fetchImpl) {
    const fallback = normalizePaperRows(snapshot);
    onProgress?.(fallback);
    onStatus?.(
      createStatus("snapshot", fallback, {
        message: "Using bundled snapshot because fetch is unavailable.",
      })
    );
    return fallback;
  }

  const allPapers = [];

  try {
    for (let page = 0; page < MAX_PAGES; page += 1) {
      const offset = page * pageSize;
      const response = await fetchImpl(
        `/api/papers?offset=${offset}&limit=${pageSize}`
      );

      if (!response.ok) {
        throw new Error(`Paper API returned ${response.status}`);
      }

      const payload = await response.json();
      const pageRows = normalizePaperRows(payload.rows);

      if (pageRows.length === 0) break;

      allPapers.push(...pageRows);
      onProgress?.([...allPapers]);
      writeCachedPapers(storage, allPapers);

      const totalRows = Number(payload.num_rows_total ?? allPapers.length);
      if (allPapers.length >= totalRows || pageRows.length < pageSize) break;
    }

    if (allPapers.length === 0) {
      throw new Error("Paper API returned no papers");
    }

    onStatus?.(createStatus("live", allPapers));
    return allPapers;
  } catch (error) {
    if (allPapers.length > 0) {
      writeCachedPapers(storage, allPapers);
      onStatus?.(
        createStatus("partial", allPapers, {
          message:
            error instanceof Error ? error.message : "Partial fetch error",
        })
      );
      return allPapers;
    }

    const cached = readCachedPapers(storage);
    if (cached.length > 0) {
      onProgress?.(cached);
      onStatus?.(
        createStatus("cache", cached, {
          message:
            error instanceof Error ? error.message : "Cached fetch error",
        })
      );
      return cached;
    }

    const fallback = normalizePaperRows(snapshot);
    onProgress?.(fallback);
    onStatus?.(
      createStatus("snapshot", fallback, {
        message: error instanceof Error ? error.message : "Unknown fetch error",
      })
    );
    return fallback;
  }
}
