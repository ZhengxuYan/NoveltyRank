#!/usr/bin/env python3
"""Utility to generate similarity reports for NoveltyRank datasets using Qwen3-235B via Tinker."""
import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, Value, load_dataset, load_from_disk
from dotenv import load_dotenv

try:  # Allow running as a module or as a script
    from .metadata import ArxivMetadataIndex, SimilarPaper
    from .prompts import build_similarity_prompt
    from .sampler import TinkerSampler
except ImportError:  # pragma: no cover - executed only for direct script usage
    CURRENT_DIR = os.path.dirname(__file__)
    REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from embedding.similiarity_report.metadata import ArxivMetadataIndex, SimilarPaper
    from embedding.similiarity_report.prompts import build_similarity_prompt
    from embedding.similiarity_report.sampler import TinkerSampler

load_dotenv()
logger = logging.getLogger("similarity_report")


class ProgressBar:
    """Simple ASCII progress bar for async generation loop."""

    def __init__(self, total: int, bar_width: int = 30) -> None:
        self.total = max(total, 1)
        self.bar_width = bar_width
        self.count = 0
        self._lock = asyncio.Lock()

    async def update(self) -> None:
        async with self._lock:
            self.count += 1
            percent = min(self.count / self.total, 1.0)
            filled = int(self.bar_width * percent)
            bar = "#" * filled + "-" * (self.bar_width - filled)
            sys.stdout.write(
                f"\rProgress [{bar}] {self.count}/{self.total} ({percent * 100:.1f}%)"
            )
            sys.stdout.flush()
            if self.count >= self.total:
                sys.stdout.write("\n")
                sys.stdout.flush()


def ensure_original_dataset(
    dataset_name: str,
    splits: List[str],
    cache_root: str,
) -> List[str]:
    """Download the original dataset splits if missing and return their local paths."""
    local_paths: List[str] = []
    abs_cache_root = os.path.abspath(cache_root)
    os.makedirs(abs_cache_root, exist_ok=True)

    for split in splits:
        split_path = os.path.join(abs_cache_root, split)
        if os.path.exists(split_path):
            logger.info("Original split '%s' found at %s", split, split_path)
            local_paths.append(split_path)
            continue

        logger.info(
            "Downloading original dataset %s split '%s' to %s", dataset_name, split, split_path
        )
        dataset = load_dataset(dataset_name, split=split)
        dataset.save_to_disk(split_path)
        local_paths.append(split_path)
        logger.info(
            "Saved original dataset split '%s' with %d rows to %s",
            split,
            len(dataset),
            split_path,
        )

    return local_paths


async def generate_similarity_report(
    entry: Dict[str, Any],
    sampler: TinkerSampler,
    metadata_index: ArxivMetadataIndex,
    *,
    preview_prompt: bool = False,
    preview_output: bool = False,
) -> Dict[str, Any]:
    similars_raw = entry.get("top_10_similar") or []
    similar_papers = [SimilarPaper.from_raw(raw, metadata_index.get) for raw in similars_raw]

    messages = build_similarity_prompt(entry, similar_papers)
    if preview_prompt:
        def _get_value(obj: Any, key: str) -> str:
            if hasattr(obj, key):
                return getattr(obj, key)
            if isinstance(obj, dict):
                return obj.get(key, "")
            return ""

        system_prompt = next(
            (_get_value(msg, "content") for msg in messages if _get_value(msg, "role") == "system"),
            "",
        )
        user_prompt = next(
            (_get_value(msg, "content") for msg in messages if _get_value(msg, "role") == "user"),
            "",
        )
        logger.info(
            "Prompt preview for %s\n--- SYSTEM ---\n%s\n--- USER ---\n%s",
            entry.get("arXiv ID", ""),
            system_prompt,
            user_prompt,
        )
    try:
        report = await sampler.generate(messages)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to generate report for %s", entry.get("arXiv ID"))
        report = f"ERROR: {exc}"

    if preview_output:
        logger.info(
            "Generated report preview for %s\n%s",
            entry.get("arXiv ID", ""),
            report,
        )

    enriched_entry: Dict[str, Any] = dict(entry)
    enriched_entry["top_10_similar"] = [paper.to_serializable() for paper in similar_papers]
    enriched_entry["similarity_report"] = report

    return enriched_entry


async def process_dataset(
    dataset: Dataset,
    sampler: TinkerSampler,
    metadata_index: ArxivMetadataIndex,
    max_concurrency: int,
    preview_prompt_count: int,
    preview_output_count: int,
    result_handler: Callable[[int, Dict[str, Any]], None],
) -> int:
    semaphore = asyncio.Semaphore(max_concurrency)
    total = len(dataset)
    progress_bar = ProgressBar(total)
    logger.info("Generating reports for %d records", total)

    async def process_entry(position: int, entry: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        try:
            async with semaphore:
                enriched = await generate_similarity_report(
                    entry,
                    sampler,
                    metadata_index,
                    preview_prompt=position < preview_prompt_count,
                    preview_output=position < preview_output_count,
                )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Unexpected failure for %s", entry.get("arXiv ID"))
            enriched = dict(entry)
            enriched.setdefault("arXiv ID", "")
            enriched.setdefault("Title", "")
            enriched.setdefault("Abstract", "")
            enriched["top_10_similar"] = entry.get("top_10_similar", [])
            enriched["similarity_report"] = f"ERROR: {exc}"
        await progress_bar.update()
        return position, enriched

    tasks = [asyncio.create_task(process_entry(i, dataset[i])) for i in range(total)]

    processed = 0
    for task in asyncio.as_completed(tasks):
        position, enriched = await task
        result_handler(position, enriched)
        processed += 1

    logger.info("Processed %d records", processed)

    return processed


def _make_json_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _make_json_serializable(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_make_json_serializable(item) for item in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()  # Works for datetime-like objects
        except Exception:  # pragma: no cover - best-effort conversion
            pass
    return str(value)


def slice_dataset(dataset: Dataset, offset: int, limit: Optional[int]) -> Dataset:
    total = len(dataset)
    start = max(0, offset)
    stop = total if limit is None else min(total, start + max(0, limit))
    if start >= stop:
        return Dataset.from_list([])
    indices = list(range(start, stop))
    return dataset.select(indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate similarity reports using Qwen3 via Tinker")
    parser.add_argument(
        "--split-root",
        default=None,
        help="Root directory containing per-split subdirectories (e.g., train/test). When provided, individual --train-split/--output-dir arguments are ignored and reports will be generated for each detected split.",
    )
    parser.add_argument(
        "--train-split",
        default="data_cache/categories/CS_CV/sft/train",
        help="Path to the target split saved with HuggingFace datasets",
    )
    parser.add_argument(
        "--test-split",
        default="data_cache/whole_dataset/test_sft_data/test_split_cleaned",
        help="Optional path to the cleaned test split to help resolve neighbour metadata",
    )
    parser.add_argument(
        "--output-dir",
        default="data_cache/similiarity_aware_categories/CS_CV/sft/train",
        help="Directory to save the enriched dataset",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory to store generated splits when --split-root is supplied. Defaults to mirroring --split-root under a similiarity-aware directory.",
    )
    parser.add_argument(
        "--category-root",
        default=None,
        help="Root directory containing per-category subdirectories. Each category will be processed by locating the split directory specified via --category-subdir.",
    )
    parser.add_argument(
        "--category-subdir",
        default="sft",
        help="Subdirectory within each category that stores the splits (e.g., 'sft'). Use '.' to treat the category directory itself as the split root.",
    )
    parser.add_argument(
        "--category",
        dest="categories",
        action="append",
        help="Limit processing to the provided category name(s) when using --category-root. Can be supplied multiple times.",
    )
    parser.add_argument("--offset", type=int, default=0, help="Start index within the dataset")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to process (default: all remaining)",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-235B-A22B-Instruct-2507",
        help="Base model name registered with Tinker",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional Tinker checkpoint URI (tinker://...) if using fine-tuned weights",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling p")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling cutoff (-1 disables)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of new tokens in the generated report",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5000,
        help="Maximum number of concurrent generation requests",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--preview-prompts",
        type=int,
        default=0,
        help="Number of prompts to log for inspection (starting from offset).",
    )
    parser.add_argument(
        "--preview-outputs",
        type=int,
        default=0,
        help="Number of generated reports to log (starting from offset).",
    )
    parser.add_argument(
        "--metadata-cache",
        default="embedding/similiarity_report/cache/metadata_index.pkl",
        help="Optional path to cache the metadata index for faster reloads.",
    )
    parser.add_argument(
        "--extra-metadata-split",
        action="append",
        help="Additional dataset paths to include when building the metadata index.",
    )
    parser.add_argument(
        "--original-dataset",
        default="JasonYan777/novelty-rank-with-similarities",
        help="HuggingFace dataset identifier for the original data.",
    )
    parser.add_argument(
        "--original-splits",
        nargs="+",
        default=["train", "test"],
        help="Original dataset splits to download for metadata lookup.",
    )
    parser.add_argument(
        "--original-cache-dir",
        default="data_cache/raw/novelty_rank_with_similarities",
        help="Local directory to cache downloaded original dataset splits.",
    )
    return parser.parse_args()


async def async_main(
    args: argparse.Namespace,
    train_split: str,
    test_split: Optional[str],
    output_dir: str,
    extra_metadata_splits: Optional[Iterable[str]] = None,
) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("Loading base dataset from %s", train_split)
    base_dataset = load_from_disk(train_split)
    target_dataset = slice_dataset(base_dataset, args.offset, args.limit)
    if len(target_dataset) == 0:
        logger.warning("No records to process (offset=%d, limit=%s)", args.offset, str(args.limit))
        return
    logger.info("Prepared dataset with %d records", len(target_dataset))
    base_features = target_dataset.features.copy()
    enriched_features = base_features.copy()
    enriched_features["similarity_report"] = Value("string")
    enriched_features["_position"] = Value("int64")

    default_metadata = [
        "data_cache/whole_dataset/train_sft_data/train_split_cleaned",
        "data_cache/whole_dataset/test_sft_data/test_split_cleaned",
    ]
    extra_splits: List[str] = []
    if extra_metadata_splits:
        extra_splits.extend(extra_metadata_splits)
    elif args.extra_metadata_split:
        extra_splits.extend(args.extra_metadata_split)
    else:
        extra_splits.extend(default_metadata)

    original_paths = ensure_original_dataset(
        dataset_name=args.original_dataset,
        splits=args.original_splits,
        cache_root=args.original_cache_dir,
    )

    metadata_roots: List[str] = []
    for path in [train_split, test_split, *extra_splits, *original_paths]:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if abs_path not in metadata_roots:
            metadata_roots.append(abs_path)
    metadata_roots.sort()

    cache_path = args.metadata_cache or None
    metadata_index = ArxivMetadataIndex(
        dataset_roots=metadata_roots,
        cache_path=cache_path,
    )
    logger.info("Metadata index ready with %d entries", len(metadata_index))

    sampler = TinkerSampler(
        model_name=args.model_name,
        model_path=args.model_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    with tempfile.TemporaryDirectory(prefix="similarity_reports_") as tmpdir:
        jsonl_path = os.path.join(tmpdir, "reports.jsonl")
        logger.debug("Writing intermediate results to %s", jsonl_path)

        with open(jsonl_path, "w", encoding="utf-8") as sink:
            def _handle_result(position: int, enriched: Dict[str, Any]) -> None:
                enriched_with_position = dict(enriched)
                enriched_with_position["_position"] = position
                serializable = _make_json_serializable(enriched_with_position)
                sink.write(json.dumps(serializable, ensure_ascii=False) + "\n")

            processed = await process_dataset(
                target_dataset,
                sampler,
                metadata_index,
                args.max_concurrency,
                preview_prompt_count=max(0, args.preview_prompts),
                preview_output_count=max(0, args.preview_outputs),
                result_handler=_handle_result,
            )
            sink.flush()

        logger.info("Loading %d intermediate records from %s", processed, jsonl_path)
        enriched_dataset = load_dataset(
            "json",
            data_files=jsonl_path,
            split="train",
            features=enriched_features,
        )
        enriched_dataset = enriched_dataset.sort("_position")
        enriched_dataset = enriched_dataset.remove_columns("_position")

        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving enriched dataset with %d entries to %s", len(enriched_dataset), output_dir)
        enriched_dataset.save_to_disk(output_dir)


def _derive_default_output_root(split_root: str) -> str:
    split_root = os.path.abspath(split_root)
    parts = split_root.split(os.sep)
    derived_parts = parts.copy()
    try:
        categories_idx = derived_parts.index("categories")
        derived_parts[categories_idx] = "similiarity_aware_categories"
    except ValueError:
        derived_parts.append("similarity_reports")
    return os.sep.join(derived_parts)


def _enumerate_split_dirs(split_root: str) -> List[Tuple[str, str]]:
    split_root = os.path.abspath(split_root)
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"split-root directory '{split_root}' does not exist")
    entries: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(split_root)):
        candidate = os.path.join(split_root, name)
        if os.path.isdir(candidate):
            entries.append((name, candidate))
    if not entries:
        raise RuntimeError(f"No subdirectories found under split-root '{split_root}'")
    return entries


def _pick_secondary_split(split_dirs: List[Tuple[str, str]], primary_name: str) -> Optional[str]:
    for name, path in split_dirs:
        if name == "test" and name != primary_name:
            return path
    for name, path in split_dirs:
        if name != primary_name:
            return path
    return None


def _process_split_root(args: argparse.Namespace, split_root: str, output_root_override: Optional[str]) -> None:
    split_dirs = _enumerate_split_dirs(split_root)
    output_root = (
        os.path.abspath(output_root_override)
        if output_root_override
        else _derive_default_output_root(split_root)
    )
    os.makedirs(output_root, exist_ok=True)

    additional_metadata = set(args.extra_metadata_split or [])
    for _, path in split_dirs:
        additional_metadata.add(path)
    sorted_additional = sorted(additional_metadata)

    for split_name, split_path in split_dirs:
        logger.info("Processing split '%s' from %s", split_name, split_path)
        target_output = os.path.join(output_root, split_name)
        metadata_splits = [root for root in sorted_additional if root != split_path]
        test_split = _pick_secondary_split(split_dirs, split_name)
        asyncio.run(
            async_main(
                args,
                train_split=split_path,
                test_split=test_split,
                output_dir=target_output,
                extra_metadata_splits=metadata_splits,
            )
        )


def main() -> None:
    args = parse_args()
    if args.category_root:
        category_root = os.path.abspath(args.category_root)
        if not os.path.isdir(category_root):
            raise FileNotFoundError(f"category-root directory '{category_root}' does not exist")

        requested: Optional[set[str]] = set(args.categories or []) if args.categories else None
        processed_any = False
        for name in sorted(os.listdir(category_root)):
            category_dir = os.path.join(category_root, name)
            if not os.path.isdir(category_dir):
                continue
            if requested and name not in requested:
                continue

            subdir = args.category_subdir or ""
            if subdir in {"", "."}:
                split_root = category_dir
            else:
                split_root = os.path.join(category_dir, subdir)

            if not os.path.isdir(split_root):
                logger.warning(
                    "Skipping category '%s'; expected split directory %s is missing",
                    name,
                    split_root,
                )
                continue

            if args.output_root:
                base_output = os.path.join(os.path.abspath(args.output_root), name)
                if subdir not in {"", "."}:
                    base_output = os.path.join(base_output, subdir)
            else:
                base_output = None

            _process_split_root(args, split_root, base_output)
            processed_any = True

        if not processed_any:
            raise RuntimeError("No categories were processed under the provided category-root")
        return

    if args.split_root:
        _process_split_root(args, args.split_root, args.output_root)
        return

    asyncio.run(
        async_main(
            args,
            train_split=args.train_split,
            test_split=args.test_split,
            output_dir=args.output_dir,
            extra_metadata_splits=args.extra_metadata_split,
        )
    )


if __name__ == "__main__":
    main()
