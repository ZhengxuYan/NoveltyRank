#!/usr/bin/env python3
"""Generate classification or comparison DPO pairs for selected NoveltyRank categories."""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

from datasets import Dataset, load_from_disk

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.Qwen_4B.data_access import category_to_token  # noqa: E402
from models.Qwen_4B.utils import create_comparison_example, generate_classification_dpo_pairs  # noqa: E402

logger = logging.getLogger(__name__)

TASK_CLASSIFICATION = "classification"
TASK_COMPARISON = "comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert per-category SFT datasets into DPO pairs",
    )
    parser.add_argument(
        "--task",
        choices=[TASK_CLASSIFICATION, TASK_COMPARISON],
        required=True,
        help="Type of DPO pairs to generate",
    )
    parser.add_argument(
        "--category-root",
        default="data_cache/similiarity_aware_categories",
        help="Root directory containing category subdirectories",
    )
    parser.add_argument(
        "--category-subdir",
        default="sft",
        help="Subdirectory within each category that holds the source splits",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Base directory to store generated DPO pairs. Defaults to the category root",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Explicit list of categories to process (e.g., cs.CV cs.RO or CS_CV)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (used for pairing and upsampling)",
    )
    parser.add_argument(
        "--include-similarity-report",
        action="store_true",
        help="Include similarity report text in classification prompts",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs instead of skipping",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def _normalize_categories(raw: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for item in raw:
        value = item.strip()
        if not value:
            continue
        if value.upper() == value:
            normalized.append(value.lower().replace("_", "."))
        else:
            normalized.append(value)
    return sorted(set(normalized))


def _discover_categories(root: Path, category_subdir: str) -> List[str]:
    if not root.exists():
        logger.warning("Category root %s missing", root)
        return []
    categories: List[str] = []
    for entry in sorted(path for path in root.iterdir() if path.is_dir()):
        candidate = entry / category_subdir / "train"
        if candidate.exists():
            categories.append(entry.name)
    return categories


def _generate_comparison_pairs(dataset: Dataset, *, seed: int) -> Dataset:
    positives = dataset.filter(lambda row: str(row.get("label")) == "1")
    negatives = dataset.filter(lambda row: str(row.get("label")) == "0")

    if len(positives) == 0 or len(negatives) == 0:
        logger.warning(
            "Skipping comparison generation: positives=%d, negatives=%d",
            len(positives),
            len(negatives),
        )
        return Dataset.from_list([])

    rng = random.Random(seed)
    neg_indices = list(range(len(negatives)))
    pairs = []

    for idx in range(len(positives)):
        neg_idx = rng.choice(neg_indices)
        pairs.append(
            create_comparison_example(
                positives[idx],
                negatives[neg_idx],
            )
        )

    return Dataset.from_list(pairs)


def _save_dataset(ds: Dataset, target_dir: Path, overwrite: bool) -> None:
    if ds is None:
        return
    if target_dir.exists():
        if overwrite:
            shutil.rmtree(target_dir)
        else:
            logger.info("Skipping existing dataset at %s", target_dir)
            return
    target_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(target_dir))


def _iter_with_progress(items: List[str], *, desc: str) -> Iterable[str]:
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(items, desc=desc)
    except Exception:  # pragma: no cover - tqdm optional
        logger.info("Processing %s: %s", desc.lower(), ", ".join(items))
        return items


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    category_root = Path(args.category_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else category_root

    if args.categories:
        requested = _normalize_categories(args.categories)
    else:
        discovered = _discover_categories(category_root, args.category_subdir)
        requested = _normalize_categories(discovered)

    if not requested:
        logger.error("No categories to process")
        return

    category_iter = _iter_with_progress(list(requested), desc="Categories")

    for category in category_iter:
        token = category_to_token(category)
        input_dir = category_root / token / args.category_subdir
        if not input_dir.exists():
            logger.warning("Missing source data for %s at %s", category, input_dir)
            continue

        logger.info("\n=== %s (%s) ===", category, token)
        split_iter = _iter_with_progress(["train", "test"], desc=f"{token} splits")
        for split in split_iter:
            source_path = input_dir / split
            if not source_path.exists():
                logger.warning("Skipping missing split %s", source_path)
                continue

            try:
                dataset = load_from_disk(str(source_path))
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Failed to load %s: %s", source_path, exc)
                continue

            if len(dataset) == 0:
                logger.warning("Split %s is empty; skipping", source_path)
                continue

            logger.info("Loaded %d rows from %s", len(dataset), source_path)

            if args.task == TASK_CLASSIFICATION:
                dpo_pairs = generate_classification_dpo_pairs(
                    dataset,
                    include_similarity_report=args.include_similarity_report,
                )
            else:
                dpo_pairs = _generate_comparison_pairs(dataset, seed=args.seed)

            logger.info("Generated %d %s pairs", len(dpo_pairs), args.task)

            target_dir = output_root / token / "dpo" / args.task / split
            try:
                _save_dataset(dpo_pairs, target_dir, overwrite=args.overwrite)
            except FileExistsError as exc:
                logger.warning("%s", exc)
                continue
            logger.info("Saved %s pairs to %s", args.task, target_dir)


if __name__ == "__main__":
    main()
