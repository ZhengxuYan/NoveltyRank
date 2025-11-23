import logging
import random
from typing import Dict, List, Any
from datasets import Dataset

logger = logging.getLogger(__name__)

def clean_dataset(dataset: Dataset) -> Dataset:
    """
    Clean the dataset by removing null similarities and rescaling scores.
    """
    # Filter out invalid entries
    dataset = dataset.filter(
        lambda example: example.get("max_similarity") not in [None, "null"]
        and example.get("avg_similarity") not in [None, "null"]
    )

    # Rescale similarity scores from 0-1 to min-max
    max_similarity_values = [float(item["max_similarity"]) for item in dataset]
    avg_similarity_values = [float(item["avg_similarity"]) for item in dataset]

    max_similarity = max(max_similarity_values) if max_similarity_values else 1.0
    min_similarity = min(avg_similarity_values) if avg_similarity_values else 0.0
    denom = max_similarity - min_similarity if max_similarity != min_similarity else 1.0

    def _rescale(item):
        item["max_similarity"] = (float(item["max_similarity"]) - min_similarity) / denom
        item["avg_similarity"] = (float(item["avg_similarity"]) - min_similarity) / denom
        return item

    dataset = dataset.map(_rescale)
    return dataset


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
    
    Randomization:
    To prevent position bias (e.g., the model always picking 'A'), we randomly 
    assign the novel paper to be Paper A or Paper B.
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

    prompt_content = f"""
You are an expert AI researcher and senior conference reviewer. 
Your goal is to compare the **conceptual novelty** of two research papers.

Compare the following two papers based on their potential for paradigm shifts and conceptual depth, not just surface similarity.

---
### Paper A
{paper_a_block}

---
### Paper B
{paper_b_block}

---
### Task
Identify which paper demonstrates higher conceptual novelty.
Output 'A' if Paper A is more novel.
Output 'B' if Paper B is more novel.

Output only the letter ('A' or 'B').
"""
    
    return {
        "prompt_conversation": [{"role": "user", "content": prompt_content}],
        "chosen": [{"role": "assistant", "content": chosen_response}],
        "rejected": [{"role": "assistant", "content": rejected_response}],
    }


def generate_dpo_pairs_from_hf(dataset: Dataset) -> Dataset:
    """
    Converts a dataset containing individual labeled papers into pairwise DPO examples.
    Strategy: Pair every Positive sample (label=1) with a random Negative sample (label=0).
    """
    # Convert to list for easier handling (memory permitting)
    data_list = list(dataset)
    
    pos_samples = [x for x in data_list if str(x.get('label')) == "1"]
    neg_samples = [x for x in data_list if str(x.get('label')) == "0"]
    
    if not pos_samples or not neg_samples:
        logger.warning("Dataset missing either positive or negative samples. Cannot form pairs.")
        return Dataset.from_list([])

    dpo_data = []
    
    for pos in pos_samples:
        # Randomly select a negative sample to contrast against
        neg = random.choice(neg_samples)
        dpo_example = create_comparison_example(pos, neg)
        dpo_data.append(dpo_example)
    
    return Dataset.from_list(dpo_data)