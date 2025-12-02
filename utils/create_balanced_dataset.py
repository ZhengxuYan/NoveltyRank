import argparse
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
import pandas as pd
import numpy as np

def create_balanced_dataset(input_dataset_name, output_dataset_name, push_to_hub=False):
    print(f"Loading dataset: {input_dataset_name}")
    dataset = load_dataset(input_dataset_name)
    
    balanced_splits = {}
    
    for split in dataset.keys():
        print(f"Processing split: {split}")
        ds = dataset[split]
        df = ds.to_pandas()
        
        # Separate by label
        novel_df = df[df['label'] == 1]
        non_novel_df = df[df['label'] == 0]
        
        n_novel = len(novel_df)
        n_non_novel = len(non_novel_df)
        
        print(f"  Novel papers: {n_novel}")
        print(f"  Non-novel papers: {n_non_novel}")
        
        if n_non_novel < n_novel:
            print(f"  Warning: Fewer non-novel papers than novel papers in {split}. Using all non-novel papers.")
            sampled_non_novel_df = non_novel_df
        else:
            # Randomly sample non-novel papers to match novel count
            sampled_non_novel_df = non_novel_df.sample(n=n_novel, random_state=42)
            print(f"  Sampled {n_novel} non-novel papers.")
            
        # Combine and shuffle
        balanced_df = pd.concat([novel_df, sampled_non_novel_df])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"  Balanced split size: {len(balanced_df)}")
        
        # Convert back to Dataset
        balanced_splits[split] = Dataset.from_pandas(balanced_df)
        
        # Preserve features if possible (sometimes pandas conversion messes up types)
        # But for simple types it should be fine.
        
    balanced_dataset = DatasetDict(balanced_splits)
    
    if push_to_hub:
        print(f"Pushing to Hub: {output_dataset_name}")
        balanced_dataset.push_to_hub(output_dataset_name)
    else:
        print("Dry run complete. Use --push to push to Hub.")
        
    return balanced_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a balanced dataset.")
    parser.add_argument("--input_dataset", type=str, default="JasonYan777/novelty-rank-with-similarities", help="Input dataset name.")
    parser.add_argument("--output_dataset", type=str, default="JasonYan777/novelty-rank-balanced", help="Output dataset name.")
    parser.add_argument("--push", action="store_true", help="Push to Hugging Face Hub.")
    
    args = parser.parse_args()
    
    create_balanced_dataset(args.input_dataset, args.output_dataset, args.push)
