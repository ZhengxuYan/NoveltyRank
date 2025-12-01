"""Utility helpers for the similarity-aware CV pipeline."""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from datasets import Dataset, concatenate_datasets

logger = logging.getLogger(__name__)


def _extract_similarity_report(example: Dict[str, Any]) -> str:
    """Return a cleaned similarity report block."""
    report = example.get("similarity_report")
    if not report:
        return "No similarity report available."
    if isinstance(report, list):
        report = "\n\n".join(str(item).strip() for item in report)
    return str(report).strip() or "No similarity report available."


def create_sft_example(example: Dict[str, Any]) -> Tuple[str, str]:
    """Build the single-turn SFT prompt and label, integrating similarity insights."""
    title = example.get("Title", "")
    authors = example.get("Authors", "")
    abstract = example.get("Abstract", "")
    max_sim = example.get("max_similarity", example.get("Max similarity", "N/A"))
    avg_sim = example.get("average_similarity", example.get("avg_similarity", "N/A"))
    similarity_report = _extract_similarity_report(example)
    label = str(example.get("label", ""))

    user_prompt = (
        "You are an expert AI researcher and senior conference reviewer (NeurIPS/ICLR level).\n"
        "Your goal is to assess the conceptual novelty of a new research paper.\n\n"
        "---\n"
        "### Conceptual Novelty Primer\n"
        "Conceptual novelty captures fundamental shifts in scientific thinking. Consider:\n"
        "- Problem Formulation: Does it redefine the task or introduce a new problem?\n"
        "- Methodological Innovation: Does it propose a new class of algorithms or frameworks?\n"
        "- Theoretical Insight: Does it supply a unifying or surprising theoretical lens?\n"
        "- Cross-Disciplinary Import: Does it bring transformative ideas from another domain?\n\n"
        "Incremental tweaks (hyperparameters, small architecture edits, dataset swaps) are not novel.\n\n"
        "---\n"
        "### Paper Metadata\n"
        f"Title: {title}\n"
        f"Authors: {authors}\n"
        f"Abstract: {abstract}\n"
        f"Max similarity to prior work: {max_sim}\n"
        f"Average similarity to prior work: {avg_sim}\n\n"
        "---\n"
        "### Similarity Report (Aggregated)\n"
        f"{similarity_report}\n\n"
        "---\n"
        "### Decision Instructions\n"
        "1. Synthesize the similarity report with the paper metadata.\n"
        "2. Judge whether the work represents a conceptually novel contribution.\n"
        "3. Output '1' if the paper is conceptually novel and likely to influence future research.\n"
        "4. Output '0' if the contribution is incremental, derivative, or lacks conceptual novelty.\n"
        "Respond with a single digit (0 or 1).\n"
    )

    return user_prompt, label


def create_classification_example(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap the SFT prompt as a DPO-style classification pair."""
    user_prompt, label = create_sft_example(paper)
    canonical = "1" if label == "1" else "0"
    alternate = "0" if canonical == "1" else "1"

    return {
        "prompt_conversation": [{"role": "user", "content": user_prompt}],
        "chosen": [{"role": "assistant", "content": canonical}],
        "rejected": [{"role": "assistant", "content": alternate}],
    }


def _balance_dataset_by_upsampling(dataset: Dataset) -> Dataset:
    """Upsample the minority class so positive and negative counts match."""
    pos_ds = dataset.filter(lambda x: str(x.get("label")) == "1")
    neg_ds = dataset.filter(lambda x: str(x.get("label")) == "0")

    if len(pos_ds) == 0 or len(pos_ds) >= len(neg_ds):
        logger.info("No upsampling required (positives >= negatives or missing positives).")
        return dataset

    repeat_times = len(neg_ds) // len(pos_ds)
    remainder = len(neg_ds) % len(pos_ds)

    clones = [pos_ds] * repeat_times
    if remainder:
        clones.append(pos_ds.select(range(remainder)))

    upsampled_pos = concatenate_datasets(clones)
    combined = concatenate_datasets([upsampled_pos, neg_ds]).shuffle(seed=42)
    logger.info("Balanced dataset size: %d", len(combined))
    return combined


def generate_classification_dpo_pairs(dataset: Dataset) -> Dataset:
    """Create DPO classification pairs with balanced labels."""
    balanced = _balance_dataset_by_upsampling(dataset)
    original_columns = balanced.column_names
    return balanced.map(
        create_classification_example,
        remove_columns=original_columns,
        desc="Generating similarity-aware classification pairs",
    )
