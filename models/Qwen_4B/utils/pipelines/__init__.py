"""Pipelines for building SFT and DPO datasets."""

from .category_pipeline import (
    CategoryPaths,
    category_to_token,
    resolve_paths,
    extract_sft_splits,
    build_classification_pairs,
    build_comparison_pairs,
    build_category_artifacts,
)
from .constants import (
    CS_RO,
    CS_CV,
    DEFAULT_CATEGORY,
    DEFAULT_OUTDIR,
    DEFAULT_SEED,
    DEFAULT_TEST_SPLIT,
    DEFAULT_TRAIN_SPLIT,
    WHOLE_DATASET,
)
from .helpers import (
    DEFAULT_TEST_CACHE_DIR,
    DEFAULT_TRAIN_CACHE_DIR,
    ensure_base_sft_splits,
    ensure_category_resources,
)

__all__ = [
    "CategoryPaths",
    "category_to_token",
    "resolve_paths",
    "extract_sft_splits",
    "build_classification_pairs",
    "build_comparison_pairs",
    "build_category_artifacts",
    "CS_RO",
    "CS_CV",
    "DEFAULT_CATEGORY",
    "DEFAULT_OUTDIR",
    "DEFAULT_SEED",
    "DEFAULT_TEST_SPLIT",
    "DEFAULT_TRAIN_SPLIT",
    "WHOLE_DATASET",
    "DEFAULT_TEST_CACHE_DIR",
    "DEFAULT_TRAIN_CACHE_DIR",
    "ensure_base_sft_splits",
    "ensure_category_resources",
]
