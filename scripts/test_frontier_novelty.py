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
                    "category": cat
                })
            else:
                pairs.append({
                    "paper1": {"title": not_novel_p["Title"], "abstract": not_novel_p["Abstract"]},
                    "paper2": {"title": novel_p["Title"], "abstract": novel_p["Abstract"]},
                    "novel_paper_idx": 2, # Paper 2 is the novel one
                    "category": cat
                })
                
    return pairs

def main():
    print("Loading dataset...")
    dataset = load_dataset("JasonYan777/novelty-rank-with-similarities", split="test")
    
    print("Creating pairs...")
    pairs = create_pairs(dataset)
    print(f"Created {len(pairs)} pairs for evaluation.")
    
    if not pairs:
        print("No pairs created. Check dataset labels and categories.")
        return

    model_name = "gpt-5.1"
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
    
    # Calculate agreement
    correct_count = 0
    total_count = len(results_list)
    
    for res in results_list:
        if res["is_correct"]:
            correct_count += 1
            
    agreement_pct = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    print("\n" + "="*30)
    print(f"Results Summary")
    print("="*30)
    print(f"Total Pairs Evaluated: {total_count}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Agreement (Accuracy): {agreement_pct:.2f}%")
    print("="*30)

    # Optional: Save detailed results to a file
    # output_file = "novelty_test_results.jsonl"
    # import json
    # with open(output_file, "w") as f:
    #    for res in results:
    #        f.write(json.dumps(res) + "\n")
    # print(f"Detailed results saved to {output_file}")

if __name__ == "__main__":
    main()
