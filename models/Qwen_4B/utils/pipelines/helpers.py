"""Helper utilities for preparing SFT and DPO artifacts."""
from __future__ import annotations

import os
from typing import Optional, Tuple

from datasets import load_dataset

from models.Qwen_4B.utils import clean_dataset

from .category_pipeline import CategoryPaths, build_category_artifacts, resolve_paths
from .constants import (
    DEFAULT_OUTDIR,
    DEFAULT_SEED,
    DEFAULT_TEST_SPLIT,
    DEFAULT_TRAIN_SPLIT,
    WHOLE_DATASET,
)
DEFAULT_TRAIN_CACHE_DIR = os.path.dirname(DEFAULT_TRAIN_SPLIT)
DEFAULT_TEST_CACHE_DIR = os.path.dirname(DEFAULT_TEST_SPLIT)


def ensure_base_sft_splits(
    dataset_path: str,
    train_cache_dir: str = DEFAULT_TRAIN_CACHE_DIR,
    test_cache_dir: str = DEFAULT_TEST_CACHE_DIR,
) -> Tuple[str, str]:
    """Ensure cleaned train/test splits exist locally and return their paths."""
    train_split = os.path.join(train_cache_dir, "train_split_cleaned")
    test_split = os.path.join(test_cache_dir, "test_split_cleaned")

    dataset = None

    if not os.path.exists(train_split):
        if dataset is None:
            dataset = load_dataset(dataset_path)
        os.makedirs(train_cache_dir, exist_ok=True)
        cleaned_train = clean_dataset(dataset["train"])
        cleaned_train.save_to_disk(train_split)

    if not os.path.exists(test_split):
        if dataset is None:
            dataset = load_dataset(dataset_path)
        os.makedirs(test_cache_dir, exist_ok=True)
        cleaned_test = clean_dataset(dataset["test"], filter_year=False)
        cleaned_test.save_to_disk(test_split)

    return train_split, test_split


def ensure_category_resources(
    category: str,
    dataset_path: str,
    train_split: str,
    test_split: str,
    *,
    outdir: str = DEFAULT_OUTDIR,
    seed: Optional[int] = None,
    need_sft: bool = False,
    need_classification: bool = False,
    need_comparison: bool = False,
) -> CategoryPaths:
    """Ensure category- or dataset-level artifacts exist and return the resolved paths."""
    filter_category = category != WHOLE_DATASET
    paths = resolve_paths(category, train_split, test_split, outdir)

    want_sft = need_sft or need_classification or need_comparison
    run_sft = want_sft and (
        not os.path.exists(paths.train_sft) or not os.path.exists(paths.test_sft)
    )
    run_classification = need_classification and (
        not os.path.exists(paths.train_classification)
        or not os.path.exists(paths.test_classification)
    )
    run_comparison = need_comparison and (
        not os.path.exists(paths.train_comparison)
        or not os.path.exists(paths.test_comparison)
    )

    if run_sft or run_classification or run_comparison:
        build_category_artifacts(
            category=category if filter_category else WHOLE_DATASET,
            train_input=train_split,
            test_input=test_split,
            outdir=outdir,
            seed=DEFAULT_SEED if seed is None else seed,
            run_sft=run_sft,
            run_classification=run_classification,
            run_comparison=run_comparison,
        )

    return paths
