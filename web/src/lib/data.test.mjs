import assert from "node:assert/strict";
import test from "node:test";

import {
  fetchAllPapers,
  getLatestPaperDate,
  normalizePaperRows,
} from "./data.js";

const samplePaper = {
  title: "A Paper",
  abstract: "A short abstract",
  categories: "cs.AI",
  primary_category: "cs.AI",
  published: "2025-12-09",
  arxiv_id: "2512.00001v1",
  url: "https://arxiv.org/abs/2512.00001v1",
  novelty_score: 1.25,
};

function createMemoryStorage(initial = {}) {
  const store = { ...initial };
  return {
    getItem: (key) => store[key] ?? null,
    setItem: (key, value) => {
      store[key] = value;
    },
    removeItem: (key) => {
      delete store[key];
    },
  };
}

test("normalizePaperRows unwraps Hugging Face row payloads", () => {
  const rows = normalizePaperRows([
    { row_idx: 0, row: samplePaper },
    { row_idx: 1, row: { ...samplePaper, arxiv_id: "2512.00002v1" } },
  ]);

  assert.equal(rows.length, 2);
  assert.equal(rows[0].title, "A Paper");
  assert.equal(rows[1].arxiv_id, "2512.00002v1");
});

test("fetchAllPapers falls back to snapshot data when the API fails", async () => {
  const progress = [];
  const statuses = [];

  const papers = await fetchAllPapers(
    (currentPapers) => progress.push(currentPapers),
    (status) => statuses.push(status),
    {
      fetchImpl: async () => ({
        ok: false,
        status: 503,
        json: async () => ({ error: "service unavailable" }),
      }),
      pageSize: 2,
      snapshot: [samplePaper],
    }
  );

  assert.equal(papers.length, 1);
  assert.equal(progress.at(-1).length, 1);
  assert.equal(statuses.at(-1).source, "snapshot");
  assert.equal(statuses.at(-1).isStale, true);
});

test("fetchAllPapers prefers cached papers over bundled snapshot on total API failure", async () => {
  const cachedPaper = { ...samplePaper, arxiv_id: "cached-paper" };
  const storage = createMemoryStorage({
    "noveltyrank-paper-cache-v1": JSON.stringify({
      savedAt: "2025-12-12T00:00:00.000Z",
      papers: [cachedPaper],
    }),
  });
  const statuses = [];

  const papers = await fetchAllPapers(
    () => {},
    (status) => statuses.push(status),
    {
      fetchImpl: async () => ({
        ok: false,
        status: 503,
        json: async () => ({ error: "service unavailable" }),
      }),
      storage,
      snapshot: [{ ...samplePaper, arxiv_id: "snapshot-paper" }],
    }
  );

  assert.equal(papers.length, 1);
  assert.equal(papers[0].arxiv_id, "cached-paper");
  assert.equal(statuses.at(-1).source, "cache");
  assert.equal(statuses.at(-1).isStale, true);
});

test("fetchAllPapers keeps partial live data when a later page fails", async () => {
  const statuses = [];
  let calls = 0;

  const papers = await fetchAllPapers(
    () => {},
    (status) => statuses.push(status),
    {
      fetchImpl: async () => {
        calls += 1;
        if (calls === 1) {
          return {
            ok: true,
            json: async () => ({
              rows: [{ row_idx: 0, row: samplePaper }],
              num_rows_total: 2,
            }),
          };
        }

        return {
          ok: false,
          status: 429,
          json: async () => ({ error: "rate limited" }),
        };
      },
      pageSize: 1,
      snapshot: [{ ...samplePaper, arxiv_id: "snapshot" }],
    }
  );

  assert.equal(papers.length, 1);
  assert.equal(papers[0].arxiv_id, "2512.00001v1");
  assert.equal(statuses.at(-1).source, "partial");
  assert.equal(statuses.at(-1).isStale, true);
});

test("getLatestPaperDate returns the newest usable published date", () => {
  const latest = getLatestPaperDate([
    { published: "not-a-date" },
    { published: "2025-12-08" },
    { published: "2025-12-09" },
  ]);

  assert.equal(latest.toISOString().slice(0, 10), "2025-12-09");
});
