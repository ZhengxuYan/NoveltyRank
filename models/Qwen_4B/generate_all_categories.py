#!/usr/bin/env python3
"""Generate per-category SFT and DPO datasets from cached NoveltyRank splits."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from datasets import Dataset, load_from_disk, load_dataset

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.Qwen_4B.utils import clean_dataset
from models.Qwen_4B.utils.pipelines.constants import DEFAULT_OUTDIR, PRIMARY_CATEGORY_FIELD, WHOLE_DATASET
from models.Qwen_4B.utils.pipelines.category_pipeline import (
    build_category_artifacts,
    ensure_clean_dir,
    resolve_paths,
)

logger = logging.getLogger(__name__)

DEFAULT_RAW_TRAIN = "data_cache/raw/novelty_rank_with_similarities/train"
DEFAULT_RAW_TEST = "data_cache/raw/novelty_rank_with_similarities/test"
DEFAULT_CATEGORIES = ["cs.CV", "cs.CL", "cs.LG", "cs.RO", "cs.CR", "cs.AI"]


def _ensure_local_splits_present(
    train_path: str, test_path: str, remote_dataset: str, splits: Sequence[str] = ("train", "test")
) -> None:
    """Ensure local dataset directories exist; if missing, download from HuggingFace remote_dataset.

    The function will attempt to download the required split(s) and save them to the provided
    `train_path` / `test_path` locations. If parent directory is provided (e.g., path ends with
    'train' or 'test'), the downloaded split is saved exactly at that path.
    """
    for target_path, default_split in ((train_path, "train"), (test_path, "test")):
        abs_target = os.path.abspath(target_path)
        if os.path.exists(abs_target):
            continue

        # determine split name to request from remote
        base = os.path.basename(abs_target).lower()
        split_name = default_split if base not in splits else base

        parent = os.path.dirname(abs_target)
        os.makedirs(parent, exist_ok=True)
        logger.info("Downloading remote split '%s' from %s to %s", split_name, remote_dataset, abs_target)
        dataset = load_dataset(remote_dataset, split=split_name)
        dataset.save_to_disk(abs_target)
        logger.info("Saved remote split '%s' (%d rows) to %s", split_name, len(dataset), abs_target)


def _collect_indices(dataset: Dataset, categories: Sequence[str]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {cat: [] for cat in categories}
    column = dataset[PRIMARY_CATEGORY_FIELD]
    for idx, value in enumerate(column):
        if isinstance(value, str):
            category = value.strip()
            if category in mapping:
                mapping[category].append(idx)
    return mapping


def _save_subset(dataset: Dataset, indices: List[int], target_path: str) -> int:
    ensure_clean_dir(target_path)
    subset = dataset.select(indices) if indices else dataset.select([])
    subset.save_to_disk(target_path)
    return len(subset)


def _prepare_sft_splits(
    train_dataset: Dataset,
    test_dataset: Dataset,
    categories: Sequence[str],
    outdir: str,
    train_source: str,
    test_source: str,
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    category_subset = [cat for cat in categories if cat != WHOLE_DATASET]
    train_indices = _collect_indices(train_dataset, category_subset)
    test_indices = _collect_indices(test_dataset, category_subset)

    for category in categories:
        paths = resolve_paths(category, train_source, test_source, outdir)
        summary[category] = {}

        if category == WHOLE_DATASET:
            train_count = _save_subset(train_dataset, list(range(len(train_dataset))), paths.train_sft)
            test_count = _save_subset(test_dataset, list(range(len(test_dataset))), paths.test_sft)
            summary[category]["train"] = train_count
            summary[category]["test"] = test_count
            continue

        cat_train_indices = train_indices.get(category, [])
        cat_test_indices = test_indices.get(category, [])

        if not cat_train_indices and not cat_test_indices:
            logger.warning("Category '%s' not present after cleaning; skipping save", category)
            continue

        if cat_train_indices:
            train_count = _save_subset(train_dataset, cat_train_indices, paths.train_sft)
            summary[category]["train"] = train_count
        else:
            logger.warning("Category '%s' missing train examples", category)
            ensure_clean_dir(paths.train_sft)
            train_dataset.select([]).save_to_disk(paths.train_sft)
            summary[category]["train"] = 0

        if cat_test_indices:
            test_count = _save_subset(test_dataset, cat_test_indices, paths.test_sft)
            summary[category]["test"] = test_count
        else:
            logger.warning("Category '%s' missing test examples", category)
            ensure_clean_dir(paths.test_sft)
            test_dataset.select([]).save_to_disk(paths.test_sft)
            summary[category]["test"] = 0

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SFT and DPO datasets for every detected category"
    )
    parser.add_argument(
        "--train-input",
        default=DEFAULT_RAW_TRAIN,
        help="Path to the base train split (Hugging Face dataset saved to disk)",
    )
    parser.add_argument(
        "--test-input",
        default=DEFAULT_RAW_TEST,
        help="Path to the base test split (Hugging Face dataset saved to disk)",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Output directory base for generated datasets (default: data_cache)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Optional list of specific categories to process. If omitted, categories are discovered automatically.",
    )
    parser.add_argument(
        "--include-whole-dataset",
        action="store_true",
        help="Also generate unfiltered splits under the WHOLE_DATASET token",
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Skip extracting SFT train/test splits",
    )
    parser.add_argument(
        "--build-classification",
        action="store_true",
        help="Generate classification DPO datasets (default: disabled)",
    )
    parser.add_argument(
        "--build-comparison",
        action="store_true",
        help="Generate comparison DPO datasets (default: disabled)",
    )
    parser.add_argument(
        "--build-all-dpo",
        action="store_true",
        help="Shortcut to enable both classification and comparison DPO generation",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip generating classification DPO datasets",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip generating comparison DPO datasets",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first category that raises an exception",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="If local train/test splits are missing, download them from a remote HuggingFace dataset",
    )
    parser.add_argument(
        "--remote-dataset-name",
        default="JasonYan777/novelty-rank-with-similarities",
        help="HuggingFace dataset identifier to download missing original splits from",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    if args.categories:
        categories = [cat.strip() for cat in args.categories if cat.strip()]
        logger.info("Using categories provided via CLI: %s", ", ".join(categories))
    else:
        categories = DEFAULT_CATEGORIES.copy()
        logger.info("Defaulting to target categories: %s", ", ".join(categories))

    if args.include_whole_dataset:
        categories = sorted(set(categories) | {WHOLE_DATASET})

    if not categories:
        logger.warning("No categories to process. Nothing to do.")
        return

    run_sft = not args.skip_sft
    run_classification = bool(args.build_classification or args.build_all_dpo)
    run_comparison = bool(args.build_comparison or args.build_all_dpo)
    if args.skip_classification:
        run_classification = False
    if args.skip_comparison:
        run_comparison = False

    sft_summary: Dict[str, Dict[str, int]] = {}
    if run_sft:
        if args.download_if_missing:
            try:
                _ensure_local_splits_present(
                    args.train_input, args.test_input, args.remote_dataset_name
                )
            except Exception:  # don't fail silently on download issues
                logger.exception("Failed to download missing raw splits from remote")
                if args.fail_fast:
                    raise
        train_dataset = clean_dataset(load_from_disk(args.train_input), filter_year=False)
        test_dataset = clean_dataset(load_from_disk(args.test_input), filter_year=False)
        logger.info(
            "Cleaned datasets -> train: %d rows, test: %d rows",
            len(train_dataset),
            len(test_dataset),
        )
        sft_summary = _prepare_sft_splits(
            train_dataset,
            test_dataset,
            categories,
            args.outdir,
            args.train_input,
            args.test_input,
        )
        if sft_summary:
            logger.info("\nSFT generation summary:")
            for category, counts in sft_summary.items():
                formatted = ", ".join(f"{split}={size}" for split, size in sorted(counts.items()))
                logger.info("  %s -> %s", category, formatted or "no records saved")

    if not run_classification and not run_comparison:
        return

    for category in categories:
        try:
            train_count = sft_summary.get(category, {}).get("train", 0) if sft_summary else None
            if run_sft and train_count == 0:
                logger.warning("Skipping DPO generation for %s due to empty train split", category)
                continue
            logger.info("\n=== Building DPO artifacts for: %s ===", category)
            summary = build_category_artifacts(
                category=category,
                train_input=args.train_input,
                test_input=args.test_input,
                outdir=args.outdir,
                run_sft=False,
                run_classification=run_classification,
                run_comparison=run_comparison,
            )
            meta = summary.pop("meta", {}) if summary else {}
            if summary:
                logger.info("Artifacts stored under %s", meta.get("category_root", "<unknown>"))
                for stage, counts in summary.items():
                    formatted = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                    logger.info("%s: %s", stage, formatted)
            else:
                logger.info("No DPO stages executed for category %s", category)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to process category %s", category)
            if args.fail_fast:
                raise exc


if __name__ == "__main__":
    main()
