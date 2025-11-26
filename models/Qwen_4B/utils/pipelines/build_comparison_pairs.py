#!/usr/bin/env python3
"""CLI for generating comparison-mode DPO pairs for a specific category."""
from __future__ import annotations

import argparse
import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(repo_root)

from models.Qwen_4B.utils.pipelines.category_pipeline import (
    DEFAULT_CATEGORY,
    DEFAULT_OUTDIR,
    DEFAULT_SEED,
    DEFAULT_TEST_SPLIT,
    DEFAULT_TRAIN_SPLIT,
    CategoryPaths,
    build_comparison_pairs,
    resolve_paths,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate comparison DPO pairs for a category")
    parser.add_argument("--category", default=DEFAULT_CATEGORY, help="Primary Category to filter (e.g., cs.RO)")
    parser.add_argument("--train-input", default=DEFAULT_TRAIN_SPLIT, help="Path to cached SFT train split")
    parser.add_argument("--test-input", default=DEFAULT_TEST_SPLIT, help="Path to cached SFT test split")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory base for generated datasets")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for cycling positives when pairing")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    paths: CategoryPaths = resolve_paths(args.category, args.train_input, args.test_input, args.outdir)
    counts = build_comparison_pairs(paths, args.seed)
    logger.info("Category artifacts stored under %s", paths.root)
    logger.info(
        "Generated comparison pairs: train=%d, test=%d",
        counts.get("train_pairs", 0),
        counts.get("test_pairs", 0),
    )


if __name__ == "__main__":
    main()
