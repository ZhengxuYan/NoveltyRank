import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
import os

def manual_update():
    csv_file = "novelty_ranking_updated.csv"
    dataset_name = "JasonYan777/novelty-rank-with-similarities-final"
    
    print(f"Reading {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} papers from CSV.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Loading reference dataset {dataset_name}...")
    try:
        ds = load_dataset(dataset_name)
        test_ds = ds["test"]
        print(f"Loaded {len(test_ds)} existing papers in test split.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Identify new papers
    # Normalize IDs
    import re
    def normalize_id(aid):
        return re.sub(r'v\d+$', '', str(aid))
        
    existing_ids = set(test_ds.to_pandas()["arXiv ID"].apply(normalize_id))
    df["normalized_id"] = df["arxiv_id"].apply(normalize_id)
    
    new_papers_df = df[~df["normalized_id"].isin(existing_ids)].copy()
    print(f"Found {len(new_papers_df)} new papers to append.")
    
    if len(new_papers_df) == 0:
        print("No new papers found. Dataset is up to date.")
        return

    # Prepare new papers dataframe
    # Map columns
    rename_map = {
        "arxiv_id": "arXiv ID",
        "url": "arXiv URL",
        "title": "Title",
        "authors": "Authors",
        "abstract": "Abstract",
        "categories": "Categories",
        "primary_category": "Primary Category",
        "published": "Publication Date"
    }
    # Only rename if they exist (the CSV might already have correct names if it was saved from final_df)
    # Check if CSV has "arXiv ID" or "arxiv_id"
    if "arXiv ID" in df.columns:
        print("CSV seems to have correct column names already.")
        # Just use the new_papers_df as is, but filter columns
    else:
        print("Renaming columns...")
        new_papers_df = new_papers_df.rename(columns=rename_map)
        
    # Ensure columns match reference features
    ref_features = test_ds.features.copy()
    if "__index_level_0__" in ref_features:
        del ref_features["__index_level_0__"]
        
    # Clean up reference dataset if needed
    if "__index_level_0__" in test_ds.column_names:
        print("Removing __index_level_0__ from reference dataset...")
        test_ds = test_ds.remove_columns(["__index_level_0__"])
        
    # Prepare new data columns
    for col in ref_features.keys():
        if col not in new_papers_df.columns:
            # Try to map from known alternatives
            if col == "PDF URL" and "arXiv ID" in new_papers_df.columns:
                 new_papers_df[col] = new_papers_df["arXiv ID"].apply(lambda x: f"https://arxiv.org/pdf/{x}.pdf" if x else None)
            elif col == "Updated Date" and "updated_date" in new_papers_df.columns:
                new_papers_df[col] = new_papers_df["updated_date"]
            elif col == "Author Affiliations" and "affiliations" in new_papers_df.columns:
                new_papers_df[col] = new_papers_df["affiliations"]
            elif col == "Journal Reference" and "journal_ref" in new_papers_df.columns:
                new_papers_df[col] = new_papers_df["journal_ref"]
            elif col == "DOI" and "doi" in new_papers_df.columns:
                new_papers_df[col] = new_papers_df["doi"]
            elif col == "Comment" and "comment" in new_papers_df.columns:
                new_papers_df[col] = new_papers_df["comment"]
            else:
                new_papers_df[col] = None
                
    # Select only relevant columns
    new_papers_df = new_papers_df[list(ref_features.keys())]
    
    # Convert to Dataset with FIX
    print("Converting to Dataset...")
    new_ds = Dataset.from_pandas(new_papers_df, preserve_index=False)
    
    if "__index_level_0__" in new_ds.column_names:
        new_ds = new_ds.remove_columns(["__index_level_0__"])
        
    new_ds = new_ds.cast(ref_features)
    
    print(f"Appending {len(new_ds)} papers...")
    try:
        updated_test_ds = concatenate_datasets([test_ds, new_ds])
        ds["test"] = updated_test_ds
        
        print(f"Pushing to {dataset_name}...")
        ds.push_to_hub(dataset_name)
        print("Success!")
        
    except Exception as e:
        print(f"Error updating dataset: {e}")

if __name__ == "__main__":
    manual_update()
