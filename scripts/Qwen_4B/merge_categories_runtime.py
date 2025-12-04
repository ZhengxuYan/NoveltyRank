#!/usr/bin/env python3
"""On-demand merger for NoveltyRank category datasets without writing to disk."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from models.Qwen_4B.data_access import (
    CATEGORY_SUBDIR_DEFAULT,
    DATA_VARIANT_BASE,
    DATA_VARIANT_SIM,
    DEFAULT_ROOTS,
    EXPECTED_DPO_COLUMNS,
    TASK_CLASSIFICATION,
    TASK_COMPARISON,
    TASK_SFT,
    merge_categories,
)

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

SUPPORTED_DATASETS = {TASK_SFT, TASK_CLASSIFICATION, TASK_COMPARISON}
SUPPORTED_VARIANTS = {DATA_VARIANT_SIM, DATA_VARIANT_BASE}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview merged NoveltyRank category datasets")
    parser.add_argument("dataset_type", choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("split", choices=["train", "test"])
    parser.add_argument(
        "--category-root",
        default=None,
        help="Root directory containing category folders (defaults depend on dataset_type)",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(SUPPORTED_VARIANTS),
        default=DATA_VARIANT_SIM,
        help="Choose similarity-aware ('sim') or base ('base') caches",
    )
    parser.add_argument(
        "--category-subdir",
        default=CATEGORY_SUBDIR_DEFAULT,
        help="Subdirectory under each category for SFT data (default: sft)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--peek",
        type=int,
        default=0,
        help="Number of rows to print from the merged dataset (0 disables)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    default_root = DEFAULT_ROOTS.get((args.dataset_type, args.variant))
    if default_root is None:
        raise ValueError(
            f"No default root configured for dataset_type={args.dataset_type}, variant={args.variant}"
        )

    category_root = args.category_root or str(default_root)

    if args.dataset_type == TASK_SFT:
        expected_columns = None
    else:
        expected_columns = EXPECTED_DPO_COLUMNS

    merged = merge_categories(
        dataset_type=args.dataset_type,
        split=args.split,
        category_root=category_root,
        category_subdir=args.category_subdir,
        expected_columns=expected_columns,
    )

    logger.info("Merged dataset length: %d", len(merged))
    if args.peek > 0:
        for idx in range(min(args.peek, len(merged))):
            logger.info("Row %d: %s", idx, merged[idx])


if __name__ == "__main__":
    main()
