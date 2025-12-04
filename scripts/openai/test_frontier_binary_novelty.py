import os
import random
import json
import argparse
from typing import Dict, List
from datasets import load_dataset
from pydantic import BaseModel, Field
from bespokelabs import curator
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

load_dotenv()

# Set Random Seed
random.seed(42)

# Define the output structure for the LLM
class BinaryNoveltyDecision(BaseModel):
    is_novel: bool = Field(
        description="True if the paper is considered novel, False otherwise."
    )
    reasoning: str = Field(
        description="A brief explanation of why this paper is considered novel or not novel."
    )

# Define the Curator LLM class
class BinaryNoveltyJudge(curator.LLM):
    response_format = BinaryNoveltyDecision

    def prompt(self, input: dict) -> List[Dict[str, str]]:
        paper = input["paper"]
        
        system_prompt = """You are an expert researcher and reviewer in the field of computer science and machine learning.
Your task is to evaluate a research paper based on its title and abstract and determine if it is NOVEL.
Novelty implies introducing new ideas, methods, or significant improvements over existing work, rather than just incremental changes.
"""
        
        user_prompt = f"""Please analyze the following paper and decide if it is novel.

Title: {paper['title']}
Abstract: {paper['abstract']}

Return True if the paper is novel, and False otherwise, along with your reasoning.
"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse(self, input: dict, response: BinaryNoveltyDecision) -> Dict:
        return {
            **input,
            "predicted_is_novel": response.is_novel,
            "predicted_reasoning": response.reasoning,
            "is_correct": response.is_novel == (input["label"] == 1)
        }

def prepare_data(dataset):
    """
    Prepares the dataset for binary classification.
    Returns a list of dicts representing individual papers.
    """
    data_points = []
    
    for item in dataset:
        data_points.append({
            "paper": {"title": item["Title"], "abstract": item["Abstract"]},
            "label": item["label"], # 1 for novel, 0 for not novel
            "primary_category": item.get("Primary Category", "Unknown")
        })
                
    return data_points

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-5.1", help="Model name to use")
    args = parser.parse_args()

    safe_model_name = args.model.replace("/", "_")
    CACHE_FILE = f"scripts/openai/results_cache_binary_{safe_model_name}.jsonl"
    
    print("Loading dataset...")
    dataset = load_dataset("JasonYan777/novelty-rank-with-similarities", split="test")
    
    print("Preparing data...")
    data_points = prepare_data(dataset)
    print(f"Prepared {len(data_points)} papers for evaluation.")
    
    if not data_points:
        print("No data points prepared. Check dataset.")
        return

    results_list = []
    
    # Check for cache
    if os.path.exists(CACHE_FILE):
        print(f"Loading results from cache: {CACHE_FILE}")
        with open(CACHE_FILE, "r") as f:
            for line in f:
                results_list.append(json.loads(line))
        print(f"Loaded {len(results_list)} results from cache.")
        
        # Verify cache matches current data count (optional warning)
        if len(results_list) != len(data_points):
            print(f"Warning: Cache size ({len(results_list)}) does not match current data ({len(data_points)}).")
            
    else:
        model_name = args.model
        print(f"Initializing BinaryNoveltyJudge with {model_name}...")
        judge = BinaryNoveltyJudge(model_name=model_name)
        
        print("Running batch inference...")
        # Curator handles batching and parallelism automatically
        results = judge(data_points)
        
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

    # Calculate metrics by category
    category_stats = {}
    all_y_true = []
    all_y_pred = []
    
    for res in results_list:
        cat = res.get("primary_category", "Unknown")
        y_true = 1 if res["label"] == 1 else 0
        y_pred = 1 if res["predicted_is_novel"] else 0
        
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        
        if cat not in category_stats:
            category_stats[cat] = {"y_true": [], "y_pred": []}
            
        category_stats[cat]["y_true"].append(y_true)
        category_stats[cat]["y_pred"].append(y_pred)
            
    # Print Results Table
    print("\n" + "="*90)
    print(f"BINARY CLASSIFICATION METRICS (Model: {args.model})")
    print("="*90)
    print(f"{'Category':<30} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Count':<8}")
    print("-" * 90)
    
    for cat, stats in sorted(category_stats.items()):
        y_t = stats["y_true"]
        y_p = stats["y_pred"]
        
        acc = accuracy_score(y_t, y_p)
        prec = precision_score(y_t, y_p, zero_division=0)
        rec = recall_score(y_t, y_p, zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)
        count = len(y_t)
        
        print(f"{cat:<30} | {acc:.2%}   | {prec:.2%}   | {rec:.2%}   | {f1:.2%}   | {count}")
        
    print("-" * 90)
    
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_prec = precision_score(all_y_true, all_y_pred, zero_division=0)
    overall_rec = recall_score(all_y_true, all_y_pred, zero_division=0)
    overall_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    total_count = len(all_y_true)
    
    print(f"{'COMBINED':<30} | {overall_acc:.2%}   | {overall_prec:.2%}   | {overall_rec:.2%}   | {overall_f1:.2%}   | {total_count}")
    print("="*90)

if __name__ == "__main__":
    main()
