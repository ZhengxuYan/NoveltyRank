import logging
import random
from collections import deque
from typing import Dict, Any
from datasets import Dataset, concatenate_datasets
from datetime import datetime

logger = logging.getLogger(__name__)

def clean_dataset(dataset: Dataset, filter_year=True) -> Dataset:
    """
    Clean the dataset by removing null similarities, filtering for 2024 dates, and rescaling scores.
    Optimized to use batched mapping and column-based stats calculation.
    """
    
    def is_valid_and_2024(example: Dict[str, Any]) -> bool:
        """Checks if similarity scores are valid and if the 'Updated Date' year is 2024."""
        # 1. Check for similarity field validity
        similarity_valid = (
            example.get("max_similarity") not in [None, "null"]
            and example.get("avg_similarity") not in [None, "null"]
        )
        
        if filter_year is False:
            return similarity_valid
        
        # 2. Check if the year of 'Updated Date' is 2024
        updated_date_str = example.get("Updated Date")
        year_is_2024 = False
        
        if updated_date_str and isinstance(updated_date_str, str):
            try:
                # Attempt to parse the date string. Format is assumed to be 'YYYY-MM-DD'
                date_object = datetime.strptime(updated_date_str, '%Y-%m-%d')
                if date_object.year == 2024:
                    year_is_2024 = True
            except ValueError:
                pass
        
        return similarity_valid and year_is_2024

    # Filter out invalid entries and non-2024 years
    dataset = dataset.filter(is_valid_and_2024)

    if len(dataset) == 0:
        return dataset

    # OPTIMIZATION: Access columns directly instead of iterating rows
    # HF Datasets allows dataset['col_name'], which is much faster than list comprehension over rows
    max_vals = dataset["max_similarity"]
    avg_vals = dataset["avg_similarity"]

    # Calculate stats
    max_similarity = max([float(x) for x in max_vals]) if max_vals else 1.0
    min_similarity = min([float(x) for x in avg_vals]) if avg_vals else 0.0
    denom = max_similarity - min_similarity if max_similarity != min_similarity else 1.0

    # OPTIMIZATION: Use batched=True to reduce function call overhead
    def _rescale_batched(batch):
        # Batch is a dict of lists, we can use list comprehensions which are faster
        batch["max_similarity"] = [(float(x) - min_similarity) / denom for x in batch["max_similarity"]]
        batch["avg_similarity"] = [(float(x) - min_similarity) / denom for x in batch["avg_similarity"]]
        return batch

    dataset = dataset.map(_rescale_batched, batched=True)
    return dataset


def _extract_similarity_report(example: Dict[str, Any]) -> str:
    """Return a cleaned similarity report block if available."""
    report = example.get("similarity_report")
    if not report:
        return "No similarity report available."
    if isinstance(report, list):
        report = "\n\n".join(str(item).strip() for item in report)
    report_str = str(report).strip()
    if not report_str:
        return "No similarity report available."

    # Remove any embedded verdict lines to avoid leaking the gold label.
    filtered_lines = []
    for line in report_str.splitlines():
        if line.strip().startswith("Novelty Verdict"):
            continue
        filtered_lines.append(line)

    cleaned_report = "\n".join(filtered_lines).replace("<|im_end|>", "").strip()
    return cleaned_report or "No similarity report available."


def format_paper_block(title: str, authors: str, abstract: str, max_sim: float, avg_sim: float) -> str:
    """Helper to format a single paper's details."""
    return f"""
Title: {title}
Authors: {authors}
Abstract: {abstract}
Max similarity to prior work: {max_sim:.4f}
Average similarity to prior work: {avg_sim:.4f}
"""


def create_comparison_example(novel_paper: Dict, random_paper: Dict) -> Dict:
    """
    Constructs a DPO comparison prompt between a Novel paper (Positive) and a Random/Not-Novel paper (Negative).
    """
    
    novel_text = format_paper_block(
        novel_paper.get("Title", ""), novel_paper.get("Authors", ""), novel_paper.get("Abstract", ""),
        novel_paper["max_similarity"], novel_paper["avg_similarity"]
    )
    
    random_text = format_paper_block(
        random_paper.get("Title", ""), random_paper.get("Authors", ""), random_paper.get("Abstract", ""),
        random_paper["max_similarity"], random_paper["avg_similarity"]
    )

    # Randomly decide if the Novel paper is A or B
    is_novel_A = random.random() < 0.5

    if is_novel_A:
        # Novel is A
        paper_a_block = novel_text
        paper_b_block = random_text
        chosen_response = "A"
        rejected_response = "B"
    else:
        # Novel is B
        paper_a_block = random_text
        paper_b_block = novel_text
        chosen_response = "B"
        rejected_response = "A"
    # CV-focused prompt: emphasize vision-specific novelty dimensions and provide CV few-shots
    prompt_content = f"""
You are an expert computer-vision researcher and senior conference reviewer (CVPR/ICCV/NeurIPS level).
Your goal is to compare the *conceptual novelty* of two computer-vision research papers (not just surface/benchmark improvements).

---
What counts as Conceptual Novelty in Computer Vision?
- Problem formulation: introduces a new problem or reframes a visual task (e.g., novel 3D reconstruction objective, new cross-modal alignment task).
- Methodological innovation: proposes a new class of models or training paradigms for visual data (e.g., transformer architectures adapted to image patches, implicit neural representations like NeRF, diffusion for image synthesis).
- Representation or inductive bias: introduces a new way to represent visual content (e.g., continuous scene representations, equivariant networks, geometric priors).
- Evaluation / benchmarking shift: defines a fundamentally different evaluation protocol that reveals new capabilities (not just adding another dataset split).
- Cross-modal or cross-disciplinary transfer that opens new research directions (e.g., physics-informed rendering for vision, neuro-inspired perceptual priors).

Incremental contributions (e.g., slightly higher ImageNet accuracy, simple augmentation tweaks, hyperparameter tuning, or re-running existing methods on another dataset) are NOT considered conceptually novel.

---
Step-by-step reasoning (use these as your guide and mention the strongest signal):
1) Extract the core technical idea from each paper's title and abstract.
2) Check whether the idea represents a new task, representation, learning paradigm, or major architectural shift.
3) Use similarity metrics as supportive evidence (high similarity tilts toward incremental), but prioritize conceptual signals (new objective, representation, or theory).
4) Choose which paper is more conceptually novel; answer only with 'A' or 'B'.

--- EXAMPLES (computer-vision focused)
Example 1:
Paper A: Introduces Vision Transformer (ViT) — treats images as a sequence of patches and applies a pure transformer backbone, changing core architecture for vision.
Paper B: Reports small regularization and augmentation tweaks to ResNet training that marginally improve accuracy.
Reasoning: A introduces a new architectural paradigm for visual representation -> Novel.
Output: A

Example 2:
Paper A: Proposes Neural Radiance Fields (NeRF) — an implicit continuous 3D scene representation enabling view synthesis.
Paper B: Improves an existing multi-view stereo pipeline with a better post-processing filter.
Reasoning: NeRF introduces a fundamentally new representation and rendering paradigm -> Novel.
Output: A

Example 3:
Paper A: Applies an off-the-shelf transformer to a small medical imaging dataset with minor changes.
Paper B: Proposes a new contrastive objective that aligns multi-resolution feature maps and demonstrates broad transfer across many vision tasks.
Reasoning: B defines a new learning objective with broad implications -> Novel.
Output: B

---
### Paper A
{paper_a_block}

---
### Paper B
{paper_b_block}

---
Output only the single letter 'A' or 'B'.
"""
    
    return {
        "prompt_conversation": [{"role": "user", "content": prompt_content}],
        "chosen": [{"role": "assistant", "content": chosen_response}],
        "rejected": [{"role": "assistant", "content": rejected_response}],
    }


def generate_comparison_dpo_pairs(dataset: Dataset, *, seed: int = 42) -> Dataset:
    """
    Converts a dataset containing individual labeled papers into pairwise DPO examples.
    Optimized to use dataset.map() instead of slow Python list appending.
    """
    
    # 1. Create filtered views
    # Note: Accessing columns is fast, we can optimize filtering slightly if needed, 
    # but .filter() is usually sufficient.
    pos_samples = dataset.filter(lambda x: str(x.get('label')) == "1")
    neg_samples = dataset.filter(lambda x: str(x.get('label')) == "0")
    
    num_neg = len(neg_samples)
    
    if len(pos_samples) == 0 or num_neg == 0:
        logger.warning("Dataset missing either positive or negative samples. Cannot form pairs.")
        return Dataset.from_list([])

    # OPTIMIZATION: Instead of a loop + list append + Dataset.from_list (which serializes/deserializes),
    # use .map() on the positive samples to generate the pairs directly.
    
    indices = list(range(len(pos_samples)))
    random.Random(seed).shuffle(indices)
    cycle_queue = deque(indices)

    pairs = []
    for neg_sample in neg_samples:
        pos_idx = cycle_queue[0]
        pos_sample = pos_samples[pos_idx]
        pairs.append(create_comparison_example(pos_sample, neg_sample))
        cycle_queue.rotate(-1)

    return Dataset.from_list(pairs)


def create_classification_example(
    paper: Dict,
    *,
    include_similarity_report: bool = False,
) -> Dict:
    """
    Constructs a DPO classification prompt for a single paper.
    Task: Is this paper novel? (Label 1) or Not (Label 0).
    """
    
    # Build the user prompt and get the canonical label using the shared prompt generator
    user_prompt, label = create_sft_example(
        paper,
        include_similarity_report=include_similarity_report,
    )

    # Ensure label is canonical '0' or '1'
    chosen_response = label if str(label) in ("0", "1") else "0"
    rejected_response = "1" if chosen_response == "0" else "0"

    return {
        "prompt_conversation": [{"role": "user", "content": user_prompt}],
        "chosen": [{"role": "assistant", "content": chosen_response}],
        "rejected": [{"role": "assistant", "content": rejected_response}],
    }


def create_sft_example(
    example: dict[str, Any],
    *,
    include_similarity_report: bool = False,
) -> tuple[str, str]:
    """
    Generate a single-turn classification prompt (user-facing) using a structured template
    and return the canonical label as a string. Optionally includes similarity analysis.
    """
    title = example.get("Title", "")
    authors = example.get("Authors", "")
    abstract = example.get("Abstract", "")
    max_sim = example.get("max_similarity", example.get("Max similarity", "N/A"))
    avg_sim = example.get("avg_similarity", example.get("Average similarity", "N/A"))
    label = str(example.get("label", ""))

    concept_primer = (
        "You are an expert AI researcher and senior conference reviewer (NeurIPS/ICLR level).\n"
        "Your goal is to judge whether the submission introduces a conceptually novel idea."\
        " Conceptual novelty captures fundamental shifts in scientific thinking."
    )

    rubric = (
        "---\n"
        "### Conceptual Novelty Primer\n"
        "Consider the following signals:\n"
        "- Problem Formulation: Does it redefine an existing task or introduce a new one?\n"
        "- Methodological Innovation: Does it propose a new class of algorithms or training paradigm?\n"
        "- Theoretical Insight: Does it deliver a unifying or surprising theoretical lens?\n"
        "- Cross-Disciplinary Import: Does it import a transformative idea from another domain?\n\n"
        "Incremental tweaks (hyperparameters, surface-level architecture edits, dataset swaps) are not novel.\n"
    )

    metadata_block = (
        "---\n"
        "### Paper Metadata\n"
        f"Title: {title}\n"
        f"Authors: {authors}\n"
        f"Abstract: {abstract}\n"
        f"Max similarity to prior work: {max_sim}\n"
        f"Average similarity to prior work: {avg_sim}\n"
    )

    if include_similarity_report:
        similarity_report = _extract_similarity_report(example)
        similarity_section = (
            "---\n"
            "### Similarity Report (Aggregated)\n"
            f"{similarity_report}\n"
        )
    else:
        similarity_section = ("---\n")

    decision_block = (
        "---\n"
        "### Decision Instructions\n"
        "1. Synthesize the available evidence (abstract + similarity signals).\n"
        "2. Decide whether the work represents a conceptually novel contribution.\n"
        "3. Output '1' if the paper is conceptually novel and likely to influence future research.\n"
        "4. Output '0' if the contribution is incremental, derivative, or lacks conceptual novelty.\n"
        "Respond with a single digit (0 or 1).\n"
    )

    few_shot = (
        "---\n"
        "### Reference Decisions\n"
        "Example 1:\n"
        "Title: Differentiable Logic for Robotics\n"
        "Abstract: Introduces a framework that composes continuous control policies with symbolic logic programs to enable reasoning-guided motion planning.\n"
        "Similarity scores: max=0.61 | avg=0.48\n"
        "Reasoning: Combines two previously disjoint paradigms (continuous control and symbolic reasoning) into a unified differentiable architecture -> Novel.\n"
        "Output: 1\n\n"
        "Example 2:\n"
        "Title: Better Hyperparameters for BERT Fine-Tuning\n"
        "Abstract: Reports extensive sweeps over learning rates and batch sizes for BERT on GLUE benchmarks.\n"
        "Similarity scores: max=0.89 | avg=0.81\n"
        "Reasoning: Purely empirical tuning without a new formulation or architecture -> Not Novel.\n"
        "Output: 0\n\n"
        "Example 3:\n"
        "Title: Physical Priors for Diffusion Models\n"
        "Abstract: Incorporates symbolic conservation laws into diffusion model training to improve controllable generation.\n"
        "Similarity scores: max=0.67 | avg=0.58\n"
        "Reasoning: Introduces a cross-disciplinary inductive bias that reshapes the generative objective -> Novel.\n"
        "Output: 1"
    )

    user_prompt = "\n\n".join([
        concept_primer,
        rubric,
        metadata_block,
        similarity_section,
        decision_block,
        few_shot,
    ])

    return user_prompt, label

def _balance_dataset_by_upsampling(dataset: Dataset) -> Dataset:
    """
    Balances a dataset by upsampling the minority class (assumed to be '1').
    """
    pos_ds = dataset.filter(lambda x: str(x.get('label')) == "1")
    neg_ds = dataset.filter(lambda x: str(x.get('label')) == "0")
    
    n_pos = len(pos_ds)
    n_neg = len(neg_ds)
    
    logger.info(f"Class Distribution - Positive: {n_pos}, Negative: {n_neg}")

    if n_pos == 0 or n_pos >= n_neg:
        logger.info("No upsampling needed or no positive samples found.")
        return dataset

    logger.info(f"Upsampling Positive class from {n_pos} to match Negative class {n_neg}...")
    
    repeat_times = n_neg // n_pos
    remainder = n_neg % n_pos
    
    datasets_to_concat = [pos_ds] * repeat_times
    if remainder > 0:
        datasets_to_concat.append(pos_ds.select(range(remainder)))
        
    upsampled_pos_ds = concatenate_datasets(datasets_to_concat)
    
    balanced_dataset = concatenate_datasets([upsampled_pos_ds, neg_ds]).shuffle(seed=42)
    
    logger.info(f"Final Balanced Dataset Size: {len(balanced_dataset)}")
    return balanced_dataset


def generate_classification_dpo_pairs(
    dataset: Dataset,
    *,
    include_similarity_report: bool = False,
) -> Dataset:
    """
    Generates Balanced DPO pairs for a Single-Paper Classification task.
    It upsamples the positive examples to match the count of negative examples.
    """
    # 1. Balance the dataset
    balanced_dataset = _balance_dataset_by_upsampling(dataset)
    
    # 2. Map to DPO format
    original_cols = balanced_dataset.column_names
    def _map_classification(example: Dict[str, Any]) -> Dict[str, Any]:
        return create_classification_example(
            example,
            include_similarity_report=include_similarity_report,
        )

    dpo_dataset = balanced_dataset.map(
        _map_classification,
        remove_columns=original_cols,
        desc="Generating Balanced Classification DPO pairs"
    )
    
    return dpo_dataset
