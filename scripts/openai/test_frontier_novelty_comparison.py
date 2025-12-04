import os
import random
from typing import Dict, List
from datasets import load_dataset
from pydantic import BaseModel, Field
from bespokelabs import curator
from dotenv import load_dotenv

load_dotenv()

# Set Random Seed
random.seed(42)

# Define the output structure for the LLM
class NoveltyChoice(BaseModel):
    more_novel_paper_id: int = Field(
        description="The ID (1 or 2) of the paper that is more novel."
    )
    reasoning: str = Field(
        description="A brief explanation of why this paper is considered more novel."
    )

# Define the Curator LLM class
class NoveltyJudge(curator.LLM):
    response_format = NoveltyChoice

    def prompt(self, input: dict) -> List[Dict[str, str]]:
        paper1 = input["paper1"]
        paper2 = input["paper2"]
        
        system_prompt = """You are an expert researcher and reviewer in the field of computer science and machine learning.
Your task is to evaluate two research papers based on their titles and abstracts and determine which one is MORE NOVEL.
Novelty implies introducing new ideas, methods, or significant improvements over existing work, rather than just incremental changes.
"""
        
        user_prompt = f"""Please analyze the following two papers and decide which one is more novel.

Paper 1:
Title: {paper1['title']}
Abstract: {paper1['abstract']}

Paper 2:
Title: {paper2['title']}
Abstract: {paper2['abstract']}

Return the ID (1 or 2) of the more novel paper and your reasoning.
"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse(self, input: dict, response: NoveltyChoice) -> Dict:
        return {
            **input,
            "predicted_novel_id": response.more_novel_paper_id,
            "predicted_reasoning": response.reasoning,
            "is_correct": (response.more_novel_paper_id == 1 and input["novel_paper_idx"] == 1) or \
                          (response.more_novel_paper_id == 2 and input["novel_paper_idx"] == 2)
        }

def create_pairs(dataset):
    """
    Creates pairs of (novel, non-novel) papers from the same category.
    Returns a list of dicts representing the pairs.
    """
    # Group by category
    category_map = {}
    for item in dataset:
        cat = item.get("Primary Category")
        label = item.get("label")
        
        if cat not in category_map:
            category_map[cat] = {"novel": [], "not_novel": []}
        
        if label == 1:
            category_map[cat]["novel"].append(item)
        elif label == 0:
            category_map[cat]["not_novel"].append(item)
    
    pairs = []
    
    for cat, papers in category_map.items():
        novel_papers = papers["novel"]
        not_novel_papers = papers["not_novel"]
        
        # Shuffle to ensure random pairing
        random.shuffle(novel_papers)
        random.shuffle(not_novel_papers)
        
        # Pair up as many as possible
        min_len = min(len(novel_papers), len(not_novel_papers))
        
        for i in range(min_len):
            novel_p = novel_papers[i]
            not_novel_p = not_novel_papers[i]
            
            # Randomly assign position 1 or 2 to the novel paper to avoid position bias
            if random.random() < 0.5:
                pairs.append({
                    "paper1": {"title": novel_p["Title"], "abstract": novel_p["Abstract"]},
                    "paper2": {"title": not_novel_p["Title"], "abstract": not_novel_p["Abstract"]},
                    "novel_paper_idx": 1, # Paper 1 is the novel one
                    "primary_category": cat
                })
            else:
                pairs.append({
                    "paper1": {"title": not_novel_p["Title"], "abstract": not_novel_p["Abstract"]},
                    "paper2": {"title": novel_p["Title"], "abstract": novel_p["Abstract"]},
                    "novel_paper_idx": 2, # Paper 2 is the novel one
                    "primary_category": cat
                })
                
    return pairs

import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-5.1", help="Model name to use")
    args = parser.parse_args()

    safe_model_name = args.model.replace("/", "_")
    CACHE_FILE = f"scripts/openai/results_cache_{safe_model_name}.jsonl"
    
    print("Loading dataset...")
    dataset = load_dataset("JasonYan777/novelty-rank-with-similarities", split="test")
    
    print("Creating pairs...")
    pairs = create_pairs(dataset)
    print(f"Created {len(pairs)} pairs for evaluation.")
    
    if not pairs:
        print("No pairs created. Check dataset labels and categories.")
        return

    results_list = []
    
    # Check for cache
    if os.path.exists(CACHE_FILE):
        print(f"Loading results from cache: {CACHE_FILE}")
        with open(CACHE_FILE, "r") as f:
            for line in f:
                results_list.append(json.loads(line))
        print(f"Loaded {len(results_list)} results from cache.")
        
        # Verify cache matches current pairs count (optional warning)
        if len(results_list) != len(pairs):
            print(f"Warning: Cache size ({len(results_list)}) does not match current pairs ({len(pairs)}).")
            
    else:
        model_name = args.model
        print(f"Initializing NoveltyJudge with {model_name}...")
        judge = NoveltyJudge(model_name=model_name)
        
        print("Running batch inference...")
        # Curator handles batching and parallelism automatically
        results = judge(pairs)
        
        # Handle CuratorResponse
        if hasattr(results, "dataset"):
            results_list = list(results.dataset)
        else:
            results_list = list(results)
            
        # Save to cache
        print(f"Saving results to cache: {CACHE_FILE}")
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            for res in results_list:
                f.write(json.dumps(res) + "\n")

    # Calculate agreement by category
    category_stats = {}
    
    for res in results_list:
        cat = res.get("primary_category", "Unknown")
        is_correct = res["is_correct"]
        
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
            
        category_stats[cat]["total"] += 1
        if is_correct:
            category_stats[cat]["correct"] += 1
            
    # Print Results Table
    print("\n" + "="*50)
    print("AGREEMENT RATES (Frontier Model)")
    print("="*50)
    print(f"{'Category':<30} | {'Agreement':<10} | {'Count':<10}")
    print("-" * 56)
    
    total_correct = 0
    total_count = 0
    
    for cat, stats in sorted(category_stats.items()):
        acc = stats["correct"] / stats["total"]
        print(f"{cat:<30} | {acc:.2%}    | {stats['total']}")
        total_correct += stats["correct"]
        total_count += stats["total"]
        
    print("-" * 56)
    overall_acc = total_correct / total_count if total_count > 0 else 0
    print(f"{'COMBINED':<30} | {overall_acc:.2%}    | {total_count}")
    print("="*50)

if __name__ == "__main__":
    main()
