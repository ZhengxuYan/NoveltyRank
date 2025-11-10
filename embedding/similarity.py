"""
Compute similarity between paper embeddings and create train/val/test splits using FAISS.

New Logic:
1. Load dataset from HuggingFace (with embeddings already generated)
2. Build a FAISS index over proximity embeddings
3. For each paper published in 2024-2025, query nearest neighbors and filter to papers
   from the past 6 months
4. Store top 10 similar papers (with embeddings and scores)
5. Add similarity columns to dataset
6. Split: Train (2024) / Validation (2025 H1) / Test (2025 H2)
7. Push to new HuggingFace dataset

Output columns:
- 'classification_embedding': Feature embeddings for classification
- 'proximity_embedding': Proximity embeddings for similarity
- 'top_10_similar': List of dicts containing embeddings and similarity scores
- 'max_similarity': Maximum similarity score
- 'avg_similarity': Average similarity score
"""

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from datetime import timedelta
from tqdm import tqdm
import argparse
import faiss


def compute_similarities_for_papers(df, year_filter=None, search_k=200):
    """
    Compute similarities for papers published in specified years using FAISS.

    Args:
        df: DataFrame with papers and embeddings
        year_filter: List of years to process (for example [2024, 2025])
        search_k: Number of nearest neighbors to query from FAISS
                  before filtering by time window

    Returns:
        DataFrame with similarity columns added
    """
    print("\n" + "=" * 80)
    print("Computing Similarities with FAISS")
    print("=" * 80)

    # Convert Publication Date to datetime
    df["Publication Date"] = pd.to_datetime(df["Publication Date"])

    # Precompute arrays
    pub_dates = df["Publication Date"].to_numpy()
    arxiv_ids = df["arXiv ID"].to_numpy()

    print("Building embedding matrix for FAISS index...")
    emb_all = np.vstack([
        np.asarray(e, dtype=np.float32)
        for e in df["proximity_embedding"].values
    ]).astype("float32")

    dim = emb_all.shape[1]
    print(f"Embedding matrix shape: {emb_all.shape} (dim = {dim})")

    # Normalize for cosine similarity and build index
    faiss.normalize_L2(emb_all)
    index = faiss.IndexFlatIP(dim)
    index.add(emb_all)
    print("FAISS index built and populated")

    # Initialize new columns
    df["top_10_similar"] = None
    df["max_similarity"] = None
    df["avg_similarity"] = None

    # Filter papers to process
    if year_filter:
        process_mask = pd.to_datetime(df["Publication Date"]).dt.year.isin(year_filter)
        print(f"Processing papers from years: {year_filter}")
    else:
        process_mask = df["Publication Date"].notna()
        print("Processing all papers with valid publication dates")

    target_indices = np.where(process_mask.values)[0]
    print(f"Total papers to process: {len(target_indices)}")
    print()

    # Process each paper
    for idx in tqdm(target_indices, desc="Computing similarities"):
        pub_date = pub_dates[idx]

        if pd.isna(pub_date):
            df.at[idx, "top_10_similar"] = []
            df.at[idx, "max_similarity"] = None
            df.at[idx, "avg_similarity"] = None
            continue

        # Define comparison window: 6 months before publication date
        half_year_before = pub_date - np.timedelta64(180, "D")

        # Query FAISS for nearest neighbors
        query_vec = emb_all[idx:idx+1].copy()
        faiss.normalize_L2(query_vec)
        D, I = index.search(query_vec, search_k)  # similarities and indices

        candidates = []
        for sim, j in zip(D[0], I[0]):
            if j == idx:
                continue
            j_date = pub_dates[j]
            # Only keep papers in the 6 months before this paper
            if not (half_year_before <= j_date < pub_date):
                continue
            candidates.append((j, float(sim)))
            if len(candidates) >= 10:
                break

        if len(candidates) == 0:
            # No papers to compare with
            df.at[idx, "top_10_similar"] = []
            df.at[idx, "max_similarity"] = None
            df.at[idx, "avg_similarity"] = None
            continue

        # Build list of dicts with embeddings and scores
        top_k_similar = []
        for j, sim in candidates:
            top_k_similar.append(
                {
                    "arxiv_id": arxiv_ids[j],
                    "similarity_score": sim,
                    "embedding": df.at[j, "proximity_embedding"],
                    "publication_date": str(pd.to_datetime(pub_dates[j]).date()),
                }
            )

        # Store results
        df.at[idx, "top_10_similar"] = top_k_similar
        df.at[idx, "max_similarity"] = float(top_k_similar[0]["similarity_score"])
        df.at[idx, "avg_similarity"] = float(
            np.mean([x["similarity_score"] for x in top_k_similar])
        )

    return df


def create_splits(df):
    """
    Create train, validation, test splits based on publication year.

    Train: 2024 papers
    Validation: 2025 H1 papers (Jan Jun)
    Test: 2025 H2 papers (Jul Dec)

    Args:
        df: DataFrame with papers

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("\n" + "=" * 80)
    print("Creating Train/Validation/Test Splits")
    print("=" * 80)

    # Ensure Publication Date is datetime
    df["Publication Date"] = pd.to_datetime(df["Publication Date"])

    # Train: 2024 papers
    train_mask = df["Publication Date"].dt.year == 2024
    train_df = df[train_mask].reset_index(drop=True)

    # Validation and Test: 2025 papers (split 50/50)
    papers_2025 = df[df["Publication Date"].dt.year == 2025].reset_index(drop=True)

    split_idx = len(papers_2025) // 2
    val_df = papers_2025.iloc[:split_idx].reset_index(drop=True)
    test_df = papers_2025.iloc[split_idx:].reset_index(drop=True)

    print(f"Train set (2024): {len(train_df)} papers")
    print(f"Validation set (2025 H1): {len(val_df)} papers")
    print(f"Test set (2025 H2): {len(test_df)} papers")
    print(f"Total: {len(train_df) + len(val_df) + len(test_df)} papers")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute similarities with FAISS and create train/val/test splits"
    )
    parser.add_argument(
        "--source-dataset",
        type=str,
        default="JasonYan777/novelty-rank-dataset",
        help="Source HuggingFace dataset (with embeddings)",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        default="JasonYan777/novelty-rank-with-similarities",
        help="Output HuggingFace dataset name",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Do not push to HuggingFace (only process locally)",
    )
    parser.add_argument(
        "--save-local",
        action="store_true",
        help="Save locally as pickle files",
    )
    parser.add_argument(
        "--search-k",
        type=int,
        default=200,
        help="Number of neighbors FAISS returns before time filtering",
    )
    args = parser.parse_args()

    # Load dataset from HuggingFace
    print("=" * 80)
    print("Loading Dataset from HuggingFace")
    print("=" * 80)
    print(f"Dataset: {args.source_dataset}")

    ds = load_dataset(args.source_dataset)
    df = pd.DataFrame(ds["train"])

    print(f"\nLoaded {len(df)} papers")
    print(f"Columns: {list(df.columns)}")

    # Check for required columns
    required_cols = [
        "proximity_embedding",
        "classification_embedding",
        "Publication Date",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"\nError: Missing required columns: {missing_cols}")
        print("Please run embeddings.py first to generate embeddings!")
        return

    # Compute similarities for 2024 and 2025 papers
    df_with_similarities = compute_similarities_for_papers(
        df,
        year_filter=[2024, 2025],
        search_k=args.search_k,
    )

    # Filter to only keep 2024 and 2025 papers
    df_filtered = (
        df_with_similarities[
            pd.to_datetime(df_with_similarities["Publication Date"]).dt.year.isin([2024, 2025])
        ]
        .copy()
        .reset_index(drop=True)
    )

    print("\n" + "=" * 80)
    print("Filtered Dataset")
    print("=" * 80)
    print(f"Kept papers from 2024 2025: {len(df_filtered)} papers")
    print("Papers from other years excluded since they do not have full history")

    # Create train/val/test splits
    train_df, val_df, test_df = create_splits(df_filtered)

    # Save locally if requested
    if args.save_local:
        import os

        os.makedirs("output", exist_ok=True)

        print("\n" + "=" * 80)
        print("Saving Local Pickle Files")
        print("=" * 80)

        train_df.to_pickle("output/data_train.pkl")
        print("Saved output/data_train.pkl")

        val_df.to_pickle("output/data_val.pkl")
        print("Saved output/data_val.pkl")

        test_df.to_pickle("output/data_test.pkl")
        print("Saved output/data_test.pkl")

    # Push to HuggingFace
    if not args.no_push:
        print("\n" + "=" * 80)
        print("Pushing to HuggingFace")
        print("=" * 80)
        print(f"Output dataset: {args.output_dataset}")

        # Convert DataFrames to Datasets
        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
        test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

        dataset_dict = DatasetDict(
            {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
        )

        dataset_dict.push_to_hub(args.output_dataset, private=False)

        print("Dataset pushed successfully!")
        print(f"View at: https://huggingface.co/datasets/{args.output_dataset}")

    # Final summary
    print("\n" + "=" * 80)
    print("ALL PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Train set: {len(train_df)} papers (2024)")
    print(f"Validation set: {len(val_df)} papers (2025 H1)")
    print(f"Test set: {len(test_df)} papers (2025 H2)")
    print(f"Total: {len(train_df) + len(val_df) + len(test_df)} papers")
    print("\nNew columns added:")
    print("  top_10_similar: List of similar papers (embeddings and scores)")
    print("  max_similarity: Maximum similarity score")
    print("  avg_similarity: Average similarity score")
    print("=" * 80)


if __name__ == "__main__":
    main()
