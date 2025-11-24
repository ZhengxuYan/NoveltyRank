import logging
import random
from typing import Dict, List, Any
from datasets import Dataset
from datetime import datetime

logger = logging.getLogger(__name__)

def clean_dataset(dataset: Dataset) -> Dataset:
    """
    Clean the dataset by removing null similarities, filtering for 2024 dates, and rescaling scores.
    """
    
    def is_valid_and_2024(example: Dict[str, Any]) -> bool:
        """Checks if similarity scores are valid and if the 'Updated Date' year is 2024."""
        
        # 1. Check for similarity field validity
        similarity_valid = (
            example.get("max_similarity") not in [None, "null"]
            and example.get("avg_similarity") not in [None, "null"]
        )
        
        # 2. Check if the year of 'Updated Date' is 2024
        updated_date_str = example.get("Updated Date")
        year_is_2024 = False
        
        # Ensure 'Updated Date' is a non-empty string
        if updated_date_str and isinstance(updated_date_str, str):
            try:
                # Attempt to parse the date string. Format is assumed to be 'YYYY-MM-DD'
                date_object = datetime.strptime(updated_date_str, '%Y-%m-%d')
                if date_object.year == 2024:
                    year_is_2024 = True
            except ValueError:
                # If date format is incorrect, the year filter fails
                pass
        
        return similarity_valid and year_is_2024

    # Filter out invalid entries and non-2024 years
    dataset = dataset.filter(is_valid_and_2024)

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
    Converts a dataset containing individual labeled papers into pairwise DPO examples 
    in a memory-efficient manner by avoiding conversion to a full Python list.

    Strategy: Pair every Positive sample (label=1) with a random Negative sample (label=0).
    """
    
    # 1. Create filtered views (still Dataset objects). 
    # This is memory efficient as it only stores references/indices, not full copies.
    pos_samples = dataset.filter(lambda x: str(x.get('label')) == "1")
    neg_samples = dataset.filter(lambda x: str(x.get('label')) == "0")
    
    num_neg = len(neg_samples)
    
    if len(pos_samples) == 0 or num_neg == 0:
        logger.warning("Dataset missing either positive or negative samples. Cannot form pairs.")
        return Dataset.from_list([])

    dpo_data: List[Dict[str, Any]] = []
    
    # 2. Iterate over the Positive samples (efficient iteration over the Dataset view).
    for pos_idx, pos in enumerate(pos_samples):
        # 3. Randomly select an INDEX from the Negative samples.
        # This avoids loading all negative samples into memory simultaneously.
        
        # Select a random index
        random_neg_index = random.randrange(num_neg)
        
        # Access the specific negative sample using the index. 
        # This fetches only the required single row from the Dataset view.
        neg = neg_samples[random_neg_index]
        
        # 4. Create the DPO example pair
        dpo_example = create_comparison_example(pos, neg)
        dpo_data.append(dpo_example)
    
    # 5. Convert the resulting list of comparison examples back to a Dataset.
    return Dataset.from_list(dpo_data)