#!/usr/bin/env python3
"""Merge per-category similarity-aware datasets into whole-dataset splits."""
import argparse
import logging
import os
from typing import Iterable, List, Tuple

from datasets import Dataset, concatenate_datasets, load_from_disk

logger = logging.getLogger("merge_similarity_datasets")


def _list_categories(root: str) -> List[Tuple[str, str]]:
    categories: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            categories.append((name, path))
    if not categories:
        raise RuntimeError(f"No category subdirectories found in {root}")
    return categories


def _load_category_split(path: str) -> Dataset:
    logger.info("Loading dataset from %s", path)
    dataset = load_from_disk(path)
    if len(dataset) == 0:
        logger.warning("Dataset at %s is empty", path)
    return dataset


def _merge_datasets(paths: Iterable[str]) -> Dataset:
    datasets: List[Dataset] = []
    for ds_path in paths:
        datasets.append(_load_category_split(ds_path))
    if not datasets:
        raise RuntimeError("No datasets were loaded for the requested split")
    if len(datasets) == 1:
        return datasets[0]
    logger.info("Concatenating %d datasets", len(datasets))
    merged = concatenate_datasets(datasets, info=datasets[0].info)
    return merged


def _save_dataset(dataset: Dataset, target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    logger.info("Saving merged dataset with %d rows to %s", len(dataset), target_dir)
    dataset.save_to_disk(target_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge similarity-aware category datasets into whole-dataset splits")
    parser.add_argument(
        "--category-root",
        default="data_cache/similiarity_aware_categories",
        help="Directory containing per-category similarity-aware data",
    )
    parser.add_argument(
        "--category-subdir",
        default="sft",
        help="Subdirectory within each category that holds the split directories (e.g., 'sft')",
    )
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        choices=["train", "test"],
        help="Specific split(s) to merge. Defaults to both train and test if omitted.",
    )
    parser.add_argument(
        "--output-root",
        default="data_cache/similiarity_aware_whole_dataset",
        help="Root directory where merged splits will be saved",
    )
    parser.add_argument(
        "--train-subdir",
        default="train_sft_data/train_split_cleaned",
        help="Relative output path for the merged train split",
    )
    parser.add_argument(
        "--test-subdir",
        default="test_sft_data/test_split_cleaned",
        help="Relative output path for the merged test split",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    category_root = os.path.abspath(args.category_root)
    if not os.path.isdir(category_root):
        raise FileNotFoundError(f"Category root '{category_root}' does not exist")

    categories = _list_categories(category_root)
    logger.info("Found %d categories", len(categories))

    requested_splits = args.splits or ["train", "test"]

    for split_name in requested_splits:
        split_paths: List[str] = []
        for _, category_path in categories:
            if args.category_subdir in {"", "."}:
                split_dir = os.path.join(category_path, split_name)
            else:
                split_dir = os.path.join(category_path, args.category_subdir, split_name)
            if not os.path.isdir(split_dir):
                logger.warning("Skipping missing split %s", split_dir)
                continue
            split_paths.append(split_dir)
        if not split_paths:
            logger.warning("No data found for split '%s'; skipping", split_name)
            continue

        merged = _merge_datasets(split_paths)

        if split_name == "train":
            output_dir = os.path.join(args.output_root, args.train_subdir)
        else:
            output_dir = os.path.join(args.output_root, args.test_subdir)
        _save_dataset(merged, output_dir)


if __name__ == "__main__":
    main()
