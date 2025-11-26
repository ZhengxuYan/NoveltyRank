#!/usr/bin/env python3
"""CLI for extracting an SFT slice for a specific `Primary Category`."""
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
    DEFAULT_TEST_SPLIT,
    DEFAULT_TRAIN_SPLIT,
    CategoryPaths,
    extract_sft_splits,
    resolve_paths,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract SFT train/test splits for a given category")
    parser.add_argument("--category", default=DEFAULT_CATEGORY, help="Primary Category to filter (e.g., cs.RO)")
    parser.add_argument("--train-input", default=DEFAULT_TRAIN_SPLIT, help="Path to cached SFT train split")
    parser.add_argument("--test-input", default=DEFAULT_TEST_SPLIT, help="Path to cached SFT test split")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory base for generated datasets")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    paths: CategoryPaths = resolve_paths(args.category, args.train_input, args.test_input, args.outdir)
    summary = extract_sft_splits(paths)
    logger.info("Category artifacts stored under %s", paths.root)
    logger.info("Extraction counts: train=%d, test=%d", summary.get("train", 0), summary.get("test", 0))


if __name__ == "__main__":
    main()
