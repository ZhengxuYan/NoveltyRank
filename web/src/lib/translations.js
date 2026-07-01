const en = {
  nav: {
    leaderboard: "Leaderboard",
    dataset: "Dataset",
  },
  hero: {
    badge: "Stanford CS230 Research Project",
    titlePrefix: "Discover",
    titleHighlight: "novel AI papers",
    titleSuffix: "before they get crowded",
    description:
      "NoveltyRank surfaces recent AI preprints by conceptual novelty, using ranked historical results from our Hugging Face dataset.",
  },
  filters: {
    viewMode: {
      papers: "Papers",
      affiliations: "Affiliations",
    },
    searchPlaceholder: "Search papers, authors, IDs, or affiliations...",
    categories: {
      All: "Overall",
      "cs.CV": "Computer Vision",
      "cs.CL": "NLP",
      "cs.LG": "Machine Learning",
      "cs.AI": "AI",
      "cs.RO": "Robotics",
      "cs.CR": "Cryptography",
    },
    timeFrames: {
      week: "Latest 7 days in dataset",
      month: "Latest 30 days in dataset",
      all: "All available papers",
    },
  },
  dashboard: {
    loading: "Loading ranked papers...",
    leaderboards: "Category leaderboards",
    indexing: "Indexing",
    papers: "papers",
    viewAll: "View all",
    noPapers: "No papers found",
    noPapersSub: "Try widening the time range or clearing search.",
    topRankings: "Top rankings",
    rankingsSuffix: "rankings",
    dataStatus: {
      live: "Live Hugging Face dataset",
      partial: "Partial Hugging Face dataset",
      cache: "Browser cache",
      snapshot: "Bundled snapshot",
      today: "Today",
      updated: "Dataset updated",
      stale:
        "Showing the last available ranked papers so the site remains browsable.",
    },
    pagination: {
      page: "Page",
    },
    table: {
      rank: "Rank",
      title: "Title",
      details: "Details",
      score: "Novelty Score",
      scoreShort: "Score",
      abstract: "Abstract",
      viewPaper: "View paper",
      topAuthor: "Top Author",
      affiliations: "Author Affiliations",
    },
  },
};

const zh = {
  ...en,
  nav: {
    leaderboard: "排行榜",
    dataset: "数据集",
  },
  hero: {
    badge: "Stanford CS230 Research Project",
    titlePrefix: "发现",
    titleHighlight: "更有新意的 AI 论文",
    titleSuffix: "",
    description:
      "NoveltyRank 基于已跑完并发布到 Hugging Face 的历史排名数据，展示近期 AI 预印本的概念新颖度。",
  },
  filters: {
    ...en.filters,
    viewMode: {
      papers: "论文",
      affiliations: "机构",
    },
    searchPlaceholder: "搜索论文、作者、arXiv ID 或机构...",
    categories: {
      All: "总览",
      "cs.CV": "计算机视觉",
      "cs.CL": "自然语言处理",
      "cs.LG": "机器学习",
      "cs.AI": "人工智能",
      "cs.RO": "机器人",
      "cs.CR": "密码学",
    },
    timeFrames: {
      week: "数据集中最近 7 天",
      month: "数据集中最近 30 天",
      all: "全部可用论文",
    },
  },
  dashboard: {
    ...en.dashboard,
    loading: "正在加载论文排名...",
    leaderboards: "分类排行榜",
    indexing: "索引中",
    papers: "篇论文",
    viewAll: "查看全部",
    noPapers: "没有找到论文",
    noPapersSub: "可以放宽时间范围，或清空搜索条件。",
    topRankings: "最高排名",
    rankingsSuffix: "排行榜",
    dataStatus: {
      live: "实时 Hugging Face 数据集",
      partial: "部分 Hugging Face 数据",
      cache: "浏览器缓存",
      snapshot: "内置快照",
      today: "今天",
      updated: "数据更新时间",
      stale: "正在展示最后一次可用的排名数据，保证网站不会空白。",
    },
    pagination: {
      page: "第",
    },
    table: {
      ...en.dashboard.table,
      rank: "排名",
      title: "标题",
      details: "详情",
      score: "新颖度分数",
      scoreShort: "分数",
      abstract: "摘要",
      viewPaper: "查看论文",
      topAuthor: "高影响作者",
      affiliations: "作者机构",
    },
  },
};

export const translations = {
  en,
  zh,
  es: en,
  fr: en,
  de: en,
  ja: en,
  ko: en,
};
