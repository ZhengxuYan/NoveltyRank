import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Import from the training script
from scripts.siamese_scibert.train_pairwise import SiameseSciBERT, Config, PairwiseNoveltyDataset, get_device, set_seed

def evaluate_agreement():
    # Setup
    set_seed(Config.RANDOM_SEED)
    device = get_device()
    print(f"Using device: {device}")

    # Load Model
    print("Loading model...")
    model = SiameseSciBERT(Config)
    model_path = "models/siamese_scibert_multimodal/best_siamese_model_v3.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load Dataset
    print("Loading dataset...")
    dataset_name = "JasonYan777/novelty-rank-with-similarities"
    hf_ds = load_dataset(dataset_name)
    test_df = hf_ds["test"].to_pandas()

    # Rename columns to match expected format (same as in train_pairwise.py)
    rename_map = {
        "classification_embedding": "Classification_embedding",
        "proximity_embedding": "Proximity_embedding"
    }
    test_df = test_df.rename(columns=rename_map)
    
    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.SCIBERT_MODEL)

    # Create Dataset and Loader
    # Note: is_train=False ensures we generate pairs for testing (1:1 or 1:all depending on implementation, 
    # but looking at train_pairwise.py, is_train=False uses max(len(novel), len(not_novel)) which is good)
    test_ds = PairwiseNoveltyDataset(test_df, tokenizer, Config.MAX_LEN, Config, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Evaluating on {len(test_ds)} pairs...")

    # Tracking results
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            batch_a = {k: v.to(device) for k, v in batch["paper_a"].items()}
            batch_b = {k: v.to(device) for k, v in batch["paper_b"].items()}
            
            # Forward pass
            score_a, score_b = model(batch_a, batch_b)
            
            # Calculate agreement (Score A > Score B)
            # Since Dataset always yields (Novel, Not Novel), we expect Score A > Score B
            agreement = (score_a > score_b).cpu().numpy().flatten()
            
            # We need to track categories to aggregate later.
            # The Dataset doesn't yield metadata by default, but we can infer or modify it.
            # However, modifying the Dataset class in train_pairwise.py might be invasive.
            # A better approach: The Dataset generates pairs in a deterministic order if shuffle=False (which it is for test_ds).
            # But wait, PairwiseNoveltyDataset iterates through category_map. dictionaries are insertion ordered in Python 3.7+.
            # Let's verify the order or just reconstruct the pairs locally to be sure.
            # Actually, `test_ds.pairs` is accessible!
            pass

    # Since we need category information for each prediction, and the DataLoader shuffles/batches,
    # we should iterate through the dataset directly or access `test_ds.pairs` to map back.
    # But `DataLoader` with `shuffle=False` preserves order.
    # Let's collect all predictions first.
    
    all_agreements = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            batch_a = {k: v.to(device) for k, v in batch["paper_a"].items()}
            batch_b = {k: v.to(device) for k, v in batch["paper_b"].items()}
            score_a, score_b = model(batch_a, batch_b)
            agreement = (score_a > score_b).cpu().numpy().flatten()
            all_agreements.extend(agreement)
            
    # Now map back to categories
    # test_ds.pairs is a list of (row_a, row_b)
    if len(all_agreements) != len(test_ds.pairs):
        print(f"Warning: Mismatch in predictions {len(all_agreements)} vs pairs {len(test_ds.pairs)}")
        
    category_stats = {}
    
    for idx, (row_a, row_b) in enumerate(test_ds.pairs):
        is_correct = all_agreements[idx]
        cat = row_a["Primary Category"] # Both should be same category
        
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
            
        category_stats[cat]["total"] += 1
        if is_correct:
            category_stats[cat]["correct"] += 1

    # Print Results
    print("\n" + "="*50)
    print("AGREEMENT RATES")
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
    evaluate_agreement()
