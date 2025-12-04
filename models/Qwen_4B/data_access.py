"""Local data access helpers for NoveltyRank training."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

from datasets import Dataset, concatenate_datasets, load_from_disk

logger = logging.getLogger(__name__)

# Canonical category identifiers
WHOLE_DATASET = "whole_dataset"
CS_AI = "cs.AI"
CS_CL = "cs.CL"
CS_CR = "cs.CR"
CS_CV = "cs.CV"
CS_LG = "cs.LG"
CS_RO = "cs.RO"

DEFAULT_CATEGORIES: Sequence[str] = (CS_AI, CS_CL, CS_CR, CS_CV, CS_LG, CS_RO)

# Dataset type keys
TASK_SFT = "sft"
TASK_CLASSIFICATION = "classification"
TASK_COMPARISON = "comparison"

# Data variant keys
DATA_VARIANT_SIM = "sim"  # similarity-aware caches under data_cache/similiarity_aware_categories
DATA_VARIANT_BASE = "base"  # original caches under data_cache/categories

# Default roots for each dataset/variant combination
DEFAULT_ROOTS = {
    (TASK_SFT, DATA_VARIANT_SIM): Path("data_cache/similiarity_aware_categories"),
    (TASK_SFT, DATA_VARIANT_BASE): Path("data_cache/categories"),
    (TASK_CLASSIFICATION, DATA_VARIANT_SIM): Path("data_cache/similiarity_aware_categories"),
    (TASK_CLASSIFICATION, DATA_VARIANT_BASE): Path("data_cache/categories"),
    (TASK_COMPARISON, DATA_VARIANT_SIM): Path("data_cache/similiarity_aware_categories"),
    (TASK_COMPARISON, DATA_VARIANT_BASE): Path("data_cache/categories"),
}

EXPECTED_DPO_COLUMNS: Sequence[str] = ("prompt_conversation", "chosen", "rejected")
CATEGORY_SUBDIR_DEFAULT = "sft"

_SUPPORTED_DATASET_TYPES = {TASK_SFT, TASK_CLASSIFICATION, TASK_COMPARISON}


def normalize_category(category: str | None) -> str:
    """Normalise user-provided category strings to canonical form."""
    if not category:
        return WHOLE_DATASET
    value = category.strip()
    if value == "":
        return WHOLE_DATASET
    if value == WHOLE_DATASET:
        return WHOLE_DATASET
    if value.upper() == value:
        value = value.lower().replace("_", ".")
    return value


def category_to_token(category: str) -> str:
    """Convert a category like 'cs.CV' to its cached directory token (e.g., CS_CV)."""
    normalised = normalize_category(category)
    if normalised == WHOLE_DATASET:
        raise ValueError("'whole_dataset' does not map to a single category token")
    token = normalised.replace("/", "_").replace(".", "_").replace("-", "_")
    return token.upper()


def _list_category_dirs(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Category root '{root}' does not exist")
    return sorted(path for path in root.iterdir() if path.is_dir())


def _resolve_split_path(
    category_dir: Path,
    dataset_type: str,
    split: str,
    category_subdir: str,
) -> Path:
    if dataset_type == TASK_SFT:
        return category_dir / category_subdir / split
    if dataset_type == TASK_CLASSIFICATION:
        return category_dir / "dpo" / TASK_CLASSIFICATION / split
    if dataset_type == TASK_COMPARISON:
        return category_dir / "dpo" / TASK_COMPARISON / split
    raise ValueError(f"Unsupported dataset_type '{dataset_type}'")


def _ensure_column_alignment(
    dataset: Dataset,
    expected_columns: Sequence[str] | None,
    *,
    source: Path,
) -> Dataset:
    if expected_columns is None:
        return dataset
    actual = list(dataset.column_names)
    expected = list(expected_columns)
    if set(actual) != set(expected):
        raise ValueError(
            f"Column mismatch for {source}: expected columns {expected}, saw {actual}"
        )
    if actual != expected:
        dataset = dataset.select_columns(expected)
    return dataset


def merge_categories(
    *,
    dataset_type: str,
    split: str,
    category_root: str | Path,
    category_subdir: str = CATEGORY_SUBDIR_DEFAULT,
    expected_columns: Sequence[str] | None = None,
) -> Dataset:
    if dataset_type not in _SUPPORTED_DATASET_TYPES:
        raise ValueError(f"Unsupported dataset_type '{dataset_type}'")

    root_path = Path(category_root).expanduser().resolve()
    datasets: List[Dataset] = []
    missing: List[Path] = []

    for category_dir in _list_category_dirs(root_path):
        split_path = _resolve_split_path(category_dir, dataset_type, split, category_subdir)
        if not split_path.exists():
            missing.append(split_path)
            continue
        ds = load_from_disk(str(split_path))
        ds = _ensure_column_alignment(ds, expected_columns, source=split_path)
        datasets.append(ds)

    if missing:
        missing_lines = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing expected dataset splits while merging categories:\n"
            f"{missing_lines}"
        )

    if not datasets:
        raise RuntimeError(
            f"No datasets loaded for dataset_type={dataset_type}, split={split} under {root_path}"
        )

    if len(datasets) == 1:
        merged = datasets[0]
    else:
        merged = concatenate_datasets(datasets, info=datasets[0].info)
    logger.info(
        "Merged %d categories for %s/%s -> %d rows",
        len(datasets),
        dataset_type,
        split,
        len(merged),
    )
    return merged


def load_category_dataset(
    *,
    dataset_type: str,
    split: str,
    category: str,
    variant: str = DATA_VARIANT_SIM,
    category_subdir: str = CATEGORY_SUBDIR_DEFAULT,
    expected_columns: Sequence[str] | None = None,
) -> Dataset:
    """Load a dataset for a specific category (or merge all when category is whole_dataset)."""
    key = (dataset_type, variant)
    if key not in DEFAULT_ROOTS:
        raise ValueError(f"Unsupported dataset_type/variant combination: {key}")
    root = DEFAULT_ROOTS[key]

    if normalize_category(category) == WHOLE_DATASET:
        return merge_categories(
            dataset_type=dataset_type,
            split=split,
            category_root=root,
            category_subdir=category_subdir,
            expected_columns=expected_columns,
        )

    token = category_to_token(category)
    split_path = _resolve_split_path(root / token, dataset_type, split, category_subdir)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Expected dataset at '{split_path}'. Ensure data generation has been run first."
        )
    ds = load_from_disk(str(split_path))
    return _ensure_column_alignment(ds, expected_columns, source=split_path)


__all__ = [
    "WHOLE_DATASET",
    "CS_AI",
    "CS_CL",
    "CS_CR",
    "CS_CV",
    "CS_LG",
    "CS_RO",
    "DEFAULT_CATEGORIES",
    "TASK_SFT",
    "TASK_CLASSIFICATION",
    "TASK_COMPARISON",
    "DATA_VARIANT_SIM",
    "DATA_VARIANT_BASE",
    "DEFAULT_ROOTS",
    "CATEGORY_SUBDIR_DEFAULT",
    "EXPECTED_DPO_COLUMNS",
    "normalize_category",
    "category_to_token",
    "merge_categories",
    "load_category_dataset",
]
