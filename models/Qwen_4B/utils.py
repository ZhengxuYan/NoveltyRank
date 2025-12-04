import logging
import random
from typing import Dict, List, Any
from datasets import Dataset
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


def format_paper_block(title: str, field: str, abstract: str, max_sim: float, avg_sim: float) -> str:
    """Helper to format a single paper's details."""
    return f"""
Title: {title}
Field: {field}
Abstract: {abstract}
Max similarity to prior work: {max_sim:.4f}
Average similarity to prior work: {avg_sim:.4f}
"""


def create_comparison_example(novel_paper: Dict, random_paper: Dict) -> Dict:
    """
    Constructs a DPO comparison prompt between a Novel paper (Positive) and a Random/Not-Novel paper (Negative).
    """
    
    novel_text = format_paper_block(
        novel_paper.get("Title", ""), novel_paper.get("Primary Category", ""), novel_paper.get("Abstract", ""),
        novel_paper["max_similarity"], novel_paper["avg_similarity"]
    )
    
    random_text = format_paper_block(
        random_paper.get("Title", ""), random_paper.get("Primary Category", ""), random_paper.get("Abstract", ""),
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

    prompt_content = f"""
You are an expert AI researcher and senior conference reviewer (NeurIPS/ICLR level). 
Your goal is to compare the **conceptual novelty** of two research papers, not just their surface similarity.

---
### What is Conceptual Novelty?
Conceptual novelty is not just about being different; it's about introducing a fundamental shift in thinking. Consider these dimensions:
-   **Problem Formulation:** Does the paper define a new problem or re-frame an existing one in a completely new way?
-   **Methodological Innovation:** Does it propose a new class of models, algorithms, or frameworks (e.g., Attention, GANs, Diffusion Models)? This is more than just tweaking an existing architecture.
-   **Theoretical Insight:** Does it provide a new theoretical understanding that unifies disparate concepts or explains a phenomenon in a new light?
-   **Cross-Disciplinary Application:** Does it successfully import a concept from another field, creating a new line of inquiry (e.g., applying concepts from physics to machine learning)?

Incremental work, such as hyperparameter tuning, minor architectural tweaks, or applying a known method to a new dataset without significant adaptation, is **not** considered conceptually novel.

---
### Step-by-step Reasoning Guide:
1.  **Understand** each paper's core contribution from its title and abstract.
2.  **Synthesize** the provided similarity scores. High similarity might suggest incremental work, but isn't conclusive. A truly novel idea might still build on existing concepts.
3.  **Evaluate** each paper's potential for a *paradigm shift* based on the definition of novelty above.
4.  **Compare** them directly. Even if both are good, one might be more foundational or conceptually deeper than the other.
5.  **Decide** which paper, A or B, is more conceptually novel and output only the corresponding letter.

---
### Few-shot Examples

**Example 1:**
*   **Paper A:** Introduces the Transformer architecture, replacing recurrence with self-attention. (Max sim: 0.68, Avg sim: 0.55)
*   **Paper B:** Modifies BERT with small regularization tweaks for better stability. (Max sim: 0.91, Avg sim: 0.82)
*   **Reasoning:** Paper A introduces a new architectural paradigm (Methodological Innovation), while B is an incremental improvement.
*   **Output:** A

**Example 2:**
*   **Paper A:** Proposes a new method for image generation using Variational Autoencoders with a novel divergence metric. (Max sim: 0.75, Avg sim: 0.65)
*   **Paper B:** Introduces Generative Adversarial Networks (GANs), a new framework where two neural networks contest with each other. (Max sim: 0.70, Avg sim: 0.60)
*   **Reasoning:** While both are generative models, GANs introduced a fundamentally new and highly influential adversarial training paradigm (Methodological Innovation).
*   **Output:** B

---
### Your Task

Compare the following two papers and identify which one demonstrates higher conceptual novelty.

---
### Paper A
{paper_a_block}

---
### Paper B
{paper_b_block}

---
### Final Output
Which paper is more novel? Output 'A' or 'B'.
Output only the letter.
"""
    
    return {
        "prompt_conversation": [{"role": "user", "content": prompt_content}],
        "chosen": [{"role": "assistant", "content": chosen_response}],
        "rejected": [{"role": "assistant", "content": rejected_response}],
    }


def generate_comparison_dpo_pairs(dataset: Dataset) -> Dataset:
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
    
    def _generate_pair_row(pos_sample):
        # Select a random negative sample
        # HF Dataset allows random access by index which is reasonably fast
        random_neg_index = random.randrange(num_neg)
        neg_sample = neg_samples[random_neg_index]
        
        return create_comparison_example(pos_sample, neg_sample)

    # We remove the original columns so the output dataset only contains the DPO structure
    # (prompt_conversation, chosen, rejected)
    original_cols = pos_samples.column_names
    
    dpo_dataset = pos_samples.map(
        _generate_pair_row, 
        remove_columns=original_cols,
        desc="Generating DPO pairs"
    )
    
    return dpo_dataset


def create_classification_example(paper: Dict) -> Dict:
    """
    Constructs a DPO classification prompt for a single paper.
    Task: Is this paper novel? (Label 1) or Not (Label 0).
    """
    
    # Use the shared SFT prompt generator to avoid duplication
    user_prompt, label = create_sft_example(paper)

    chosen_response = label if str(label) in ("0", "1") else "0"
    rejected_response = "1" if chosen_response == "0" else "0"

    return {
        "prompt_conversation": [{"role": "user", "content": user_prompt}],
        "chosen": [{"role": "assistant", "content": chosen_response}],
        "rejected": [{"role": "assistant", "content": rejected_response}],
    }


def create_sft_example(example: dict[str, str]) -> list[dict[str, str]]:
    """
    Generates the full multi-turn prompt (including few-shot examples and reasoning steps)
    and the ground truth label from a single paper example.
    """
    title = example.get("Title", "")
    field = example.get("Primary Category", "")  # Will be used as field
    abstract = example.get("Abstract", "")
    max_sim = example.get("max_similarity", "N/A")
    avg_sim = example.get("average_similarity", "N/A")
    label = example.get("label", "")
    
    # generate user prompt
    user_prompt = f"""
                You are an expert AI researcher and senior conference reviewer (NeurIPS/ICLR level). 
                Your goal is to **assess the conceptual novelty** of a new research paper.

                ---
                ### What is Conceptual Novelty?
                Conceptual novelty is not just about being different; it's about introducing a fundamental shift in thinking. Consider these dimensions:
                -   **Problem Formulation:** Does the paper define a new problem or re-frame an existing one in a completely new way?
                -   **Methodological Innovation:** Does it propose a new class of models, algorithms, or frameworks (e.g., Attention, GANs, Diffusion Models)? This is more than just tweaking an existing architecture.
                -   **Theoretical Insight:** Does it provide a new theoretical understanding that unifies disparate concepts or explains a phenomenon in a new light?
                -   **Cross-Disciplinary Application:** Does it successfully import a concept from another field, creating a new line of inquiry (e.g., applying concepts from physics to machine learning)?

                Incremental work, such as hyperparameter tuning, minor architectural tweaks, or applying a known method to a new dataset without significant adaptation, is **not** considered conceptually novel.
            
                ---
                ### Step-by-step reasoning:
                1. **Understand** the paper’s main idea, contribution, and context from its title and abstract.
                2. **Compare** it conceptually against prior literature (based on the given similarity metrics).
                3. **Evaluate** whether the paper represents a conceptually novel contribution based on the definition above.
                4. **Predict** whether this project is likely to have *high influence* or *define a new paradigm*.
                5. Finally, output a binary decision:
                    - Output `'1'` if the paper is conceptually novel and likely to influence future research,
                    - Output `'0'` otherwise.
            
                ---
                ### Input
                Title: {title}
                Field: {field}
                Abstract: {abstract}
                Max similarity to prior work: {max_sim}
                Average similarity to prior work: {avg_sim}
            
                ---
                ### Few-shot Examples
            
            **Example 1**
            Title: "Attention Is All You Need"
            Abstract: Introduces the Transformer architecture, replacing recurrence with attention mechanisms for sequence modeling.
            Max similarity: 0.68 | Avg similarity: 0.55  
            **Reasoning:** This paper introduced a new architectural paradigm (Methodological Innovation).
            **Output:** 1
            
            **Example 2**
            Title: "A Slightly Improved BERT Model with Layer-wise Dropout"
            Abstract: Modifies BERT with small regularization tweaks and hyperparameter tuning for better stability.
            Max similarity: 0.91 | Avg similarity: 0.82  
            **Reasoning:** Highly similar to prior work and lacks new conceptual insights. This is incremental.
            **Output:** 0
            
            **Example 3**
            Title: "Aligning LLMs with Value-Constrained Reinforcement Learning"
            Abstract: Proposes a reinforcement learning approach with explicit value constraints to align LLM behavior.
            Max similarity: 0.73 | Avg similarity: 0.64  
            **Reasoning:** Builds on known RLHF methods but introduces a novel constrained optimization view (Problem Formulation). Moderately novel.
            **Output:** 1
            
            ---
            Now, reason through the given paper step by step and output only '1' or '0'.
            """
    return (user_prompt, str(label))
        

from datasets import Dataset, concatenate_datasets


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


def generate_classification_dpo_pairs(dataset: Dataset) -> Dataset:
    """
    Generates Balanced DPO pairs for a Single-Paper Classification task.
    It upsamples the positive examples to match the count of negative examples.
    """
    # 1. Balance the dataset
    balanced_dataset = _balance_dataset_by_upsampling(dataset)
    
    # 2. Map to DPO format
    original_cols = balanced_dataset.column_names
    dpo_dataset = balanced_dataset.map(
        create_classification_example,
        remove_columns=original_cols,
        desc="Generating Balanced Classification DPO pairs"
    )
    
    return dpo_dataset


# --- Pipeline helpers moved from utils.pipelines ---
import os
import shutil
import random
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque
from datasets import load_from_disk


# Constants (kept here for convenience by pipeline helpers)
WHOLE_DATASET = "whole_dataset"
PRIMARY_CATEGORY_FIELD = "Primary Category"
LABEL_FIELD = "label"
DEFAULT_OUTDIR = "data_cache"
DEFAULT_SEED = 42
DEFAULT_TEST_SPLIT = "data_cache/whole_dataset/test_sft_data/test_split_cleaned"
DEFAULT_TRAIN_SPLIT = "data_cache/whole_dataset/train_sft_data/train_split_cleaned"


@dataclass
class CategoryPaths:
    category: str
    token: str
    train_source: str
    test_source: str
    root: str
    train_sft: str
    test_sft: str
    train_classification: str
    test_classification: str
    train_comparison: str
    test_comparison: str


def category_to_token(category: str) -> str:
    token = category.replace("/", "_").replace(".", "_").replace("-", "_")
    return token.upper()


def ensure_clean_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def resolve_paths(category: str, train_source: str, test_source: str, outdir: str) -> CategoryPaths:
    token = category_to_token(category)
    category_root = os.path.join(outdir, "categories", token)
    train_sft_dir = os.path.join(category_root, "sft", "train")
    test_sft_dir = os.path.join(category_root, "sft", "test")
    train_class_dir = os.path.join(category_root, "dpo", "classification", "train")
    test_class_dir = os.path.join(category_root, "dpo", "classification", "test")
    train_comp_dir = os.path.join(category_root, "dpo", "comparison", "train")
    test_comp_dir = os.path.join(category_root, "dpo", "comparison", "test")
    return CategoryPaths(
        category=category,
        token=token,
        train_source=train_source,
        test_source=test_source,
        root=category_root,
        train_sft=train_sft_dir,
        test_sft=test_sft_dir,
        train_classification=train_class_dir,
        test_classification=test_class_dir,
        train_comparison=train_comp_dir,
        test_comparison=test_comp_dir,
    )


def require_dataset(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected {label} at '{path}'. Run the prerequisite stage or adjust the input paths."
        )


def load_and_filter(dataset_path: str, category: str, filter_category: bool) -> Dataset:
    logger.info("Loading dataset from %s", dataset_path)
    ds = load_from_disk(dataset_path)
    if not filter_category:
        logger.info("Filter disabled; using entire dataset from %s", dataset_path)
        return ds

    if PRIMARY_CATEGORY_FIELD not in ds.column_names:
        raise RuntimeError(f"Expected column '{PRIMARY_CATEGORY_FIELD}' not found in dataset")

    logger.info("Filtering for category '%s'", category)
    filtered = ds.filter(lambda row: str(row.get(PRIMARY_CATEGORY_FIELD, "")).strip() == category)
    logger.info("Selected %d rows from %s", len(filtered), dataset_path)
    return filtered


def save_dataset(ds: Dataset, path: str):
    ensure_clean_dir(path)
    ds.save_to_disk(path)
    logger.info("Saved dataset to %s", path)


def extract_sft_splits(paths: CategoryPaths, filter_category: bool = True) -> dict:
    counts = {}
    test_filtered = load_and_filter(paths.test_source, paths.category, filter_category)
    save_dataset(test_filtered, paths.test_sft)
    counts["test"] = len(test_filtered)

    train_filtered = load_and_filter(paths.train_source, paths.category, filter_category)
    if len(train_filtered) == 0:
        logger.warning("No train examples found for category '%s'", paths.category)
    else:
        save_dataset(train_filtered, paths.train_sft)
    counts["train"] = len(train_filtered)
    return counts


def build_classification_pairs(paths: CategoryPaths) -> dict:
    counts = {}

    require_dataset(paths.train_sft, "train SFT split")
    logger.info("Loading train subset from %s", paths.train_sft)
    train_ds = load_from_disk(paths.train_sft)
    logger.info("Loaded %d train rows", len(train_ds))

    logger.info("Generating balanced classification DPO pairs for train subset")
    train_pairs = generate_classification_dpo_pairs(train_ds)
    logger.info("Generated %d train classification pairs", len(train_pairs))

    ensure_clean_dir(paths.train_classification)
    train_pairs.save_to_disk(paths.train_classification)
    counts["train_pairs"] = len(train_pairs)
    logger.info("Saved train classification pairs to %s", paths.train_classification)

    require_dataset(paths.test_sft, "test SFT split")
    logger.info("Loading test subset from %s", paths.test_sft)
    test_ds = load_from_disk(paths.test_sft)
    logger.info("Loaded %d test rows", len(test_ds))

    logger.info("Converting test subset to classification DPO format without balancing")
    test_pairs = test_ds.map(
        create_classification_example,
        remove_columns=test_ds.column_names,
        desc="Generating classification DPO pairs for test subset",
    )
    logger.info("Generated %d test classification pairs", len(test_pairs))

    ensure_clean_dir(paths.test_classification)
    test_pairs.save_to_disk(paths.test_classification)
    counts["test_pairs"] = len(test_pairs)
    logger.info("Saved test classification pairs to %s", paths.test_classification)

    return counts


def build_comparison_pairs(paths: CategoryPaths, seed: int) -> dict:
    counts = {}

    def build_for_split(split_name: str, src_path: str, out_path: str) -> int:
        require_dataset(src_path, f"{split_name} SFT split")
        logger.info("Loading %s subset from %s", split_name, src_path)
        ds = load_from_disk(src_path)
        logger.info("Loaded %d %s rows", len(ds), split_name)

        positives = ds.filter(lambda x: str(x.get(LABEL_FIELD, "")) == "1")
        negatives = ds.filter(lambda x: str(x.get(LABEL_FIELD, "")) == "0")
        logger.info("%s: positives=%d, negatives=%d", split_name, len(positives), len(negatives))

        if len(positives) == 0 or len(negatives) == 0:
            logger.warning("%s: Missing classes, skipping comparison pair generation", split_name)
            ensure_clean_dir(out_path)
            Dataset.from_list([]).save_to_disk(out_path)
            return 0

        indices = list(range(len(positives)))
        random.Random(seed).shuffle(indices)
        cycle_queue: Deque[int] = deque(indices)

        pairs = []
        for neg_sample in negatives:
            pos_idx = cycle_queue[0]
            pos_sample = positives[pos_idx]
            pairs.append(create_comparison_example(pos_sample, neg_sample))
            cycle_queue.rotate(-1)

        logger.info("%s: Generated %d comparison pairs", split_name, len(pairs))
        ensure_clean_dir(out_path)
        Dataset.from_list(pairs).save_to_disk(out_path)
        return len(pairs)

    counts["train_pairs"] = build_for_split("train", paths.train_sft, paths.train_comparison)
    counts["test_pairs"] = build_for_split("test", paths.test_sft, paths.test_comparison)
    return counts


def build_category_artifacts(
    category: str,
    train_input: str = DEFAULT_TRAIN_SPLIT,
    test_input: str = DEFAULT_TEST_SPLIT,
    outdir: str = DEFAULT_OUTDIR,
    seed: int = DEFAULT_SEED,
    run_sft: bool = True,
    run_classification: bool = True,
    run_comparison: bool = True,
) -> dict:
    filter_category = category != WHOLE_DATASET
    paths = resolve_paths(category, train_input, test_input, outdir)
    summary = {}

    if run_sft:
        summary["sft"] = extract_sft_splits(paths, filter_category=filter_category)
    if run_classification:
        summary["classification"] = build_classification_pairs(paths)
    if run_comparison:
        summary["comparison"] = build_comparison_pairs(paths, seed)

    if summary:
        summary["meta"] = {"category_root": paths.root}

    return summary


def ensure_base_sft_splits(
    dataset_path: str,
    train_cache_dir: str = os.path.dirname(DEFAULT_TRAIN_SPLIT),
    test_cache_dir: str = os.path.dirname(DEFAULT_TEST_SPLIT),
) -> tuple[str, str]:
    train_split = os.path.join(train_cache_dir, "train_split_cleaned")
    test_split = os.path.join(test_cache_dir, "test_split_cleaned")

    dataset = None

    if not os.path.exists(train_split):
        if dataset is None:
            from datasets import load_dataset

            dataset = load_dataset(dataset_path)
        os.makedirs(train_cache_dir, exist_ok=True)
        cleaned_train = clean_dataset(dataset["train"])
        cleaned_train.save_to_disk(train_split)

    if not os.path.exists(test_split):
        if dataset is None:
            from datasets import load_dataset

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
    seed: int | None = None,
    need_sft: bool = False,
    need_classification: bool = False,
    need_comparison: bool = False,
) -> CategoryPaths:
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