import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import argparse
import faiss
import os

def compute_similarities_for_papers(df, all_papers_df, embedding="proximity", search_k=200):
    print("\n" + "=" * 80)
    print("=" * 80)

    # Convert Publication Date to datetime for both dataframes
    df = df.copy()
    all_papers_df = all_papers_df.copy()
    df["Publication Date"] = pd.to_datetime(df["Publication Date"])
    all_papers_df["Publication Date"] = pd.to_datetime(
        all_papers_df["Publication Date"]
    )

    # Build FAISS index from ALL papers (for comprehensive search)
    print("Building embedding matrix for FAISS index from all papers...")
    emb_all = np.vstack(
        [
            np.asarray(e, dtype=np.float32)
            for e in all_papers_df[embedding].values
        ]
    ).astype("float32")

    dim = emb_all.shape[1]
    print(f"Index embedding matrix shape: {emb_all.shape} (dim = {dim})")
    print(
        f"Date range in index: {all_papers_df['Publication Date'].min()} to {all_papers_df['Publication Date'].max()}"
    )

    # Normalize for cosine similarity and build index
    faiss.normalize_L2(emb_all)
    index = faiss.IndexFlatIP(dim)
    index.add(emb_all)
    print("FAISS index built and populated")

    # Precompute arrays from index dataframe
    index_pub_dates = all_papers_df["Publication Date"].to_numpy()
    index_arxiv_ids = all_papers_df["arXiv ID"].to_numpy()

    # Extract embedding name prefix (remove "_embedding" suffix if present)
    emb_prefix = embedding.replace("_embedding", "")

    # Initialize new columns for the subset we're processing
    df[f"{emb_prefix}_top_10_sim"] = None
    df[f"{emb_prefix}_max_sim"] = None
    df[f"{emb_prefix}_avg_sim"] = None

    print(f"\nProcessing {len(df)} papers to compute similarities")
    print()

    # Build query embeddings for the papers we're processing
    query_embs = np.vstack(
        [np.asarray(e, dtype=np.float32) for e in df[embedding].values]
    ).astype("float32")
    faiss.normalize_L2(query_embs)

    # Process each paper
    for idx in tqdm(range(len(df)), desc="Computing similarities"):
        pub_date = df.iloc[idx]["Publication Date"]
        current_arxiv_id = df.iloc[idx]["arXiv ID"]

        if pd.isna(pub_date):
            df.at[df.index[idx], f"{emb_prefix}_top_10_sim"] = []
            df.at[df.index[idx], f"{emb_prefix}_max_sim"] = None
            df.at[df.index[idx], f"{emb_prefix}_avg_sim"] = None
            continue

        # Define comparison window: 1 year before publication date
        one_year_before = pub_date - pd.Timedelta(days=365)

        # Query FAISS for nearest neighbors
        query_vec = query_embs[idx : idx + 1].copy()
        distances, indices = index.search(
            query_vec, search_k
        )  # similarities and indices

        candidates = []
        for sim, j in zip(distances[0], indices[0]):
            # Skip if it's the same paper
            if index_arxiv_ids[j] == current_arxiv_id:
                continue

            j_date = index_pub_dates[j]
            # Only keep papers in the 1 year before this paper
            if not (one_year_before <= j_date < pub_date):
                continue
            candidates.append((j, float(sim)))
            if len(candidates) >= 10:
                break

        if len(candidates) == 0:
            # No papers to compare with
            df.at[df.index[idx], f"{emb_prefix}_top_10_sim"] = []
            df.at[df.index[idx], f"{emb_prefix}_max_sim"] = None
            df.at[df.index[idx], f"{emb_prefix}_avg_sim"] = None
            continue

        # Build list of dicts with embeddings and scores
        top_k_similar = []
        for j, sim in candidates:
            top_k_similar.append(
                {
                    "arxiv_id": index_arxiv_ids[j],
                    "similarity_score": sim,
                    "embedding": all_papers_df.iloc[j][embedding],
                    "publication_date": str(pd.to_datetime(index_pub_dates[j]).date()),
                }
            )

        # Store results
        df.at[df.index[idx], f"{emb_prefix}_top_10_sim"] = top_k_similar
        df.at[df.index[idx], f"{emb_prefix}_max_sim"] = float(
            top_k_similar[0]["similarity_score"]
        )
        df.at[df.index[idx], f"{emb_prefix}_avg_sim"] = float(
            np.mean([x["similarity_score"] for x in top_k_similar])
        )

    return df


def create_splits(df, split_date="2025-03-15"):
    print("\n" + "=" * 80)
    print("Creating Train/Test Splits")
    print("=" * 80)

    # Ensure Publication Date is datetime
    df["Publication Date"] = pd.to_datetime(df["Publication Date"])
    split_date = pd.to_datetime(split_date)

    # Train: Papers before March 1, 2025
    train_mask = df["Publication Date"] < split_date
    train_df = df[train_mask].reset_index(drop=True)

    # Test: Papers from March 1, 2025 onwards
    test_mask = df["Publication Date"] >= split_date
    test_df = df[test_mask].reset_index(drop=True)

    print(f"Split date: {split_date.date()}")
    print(f"Train set (before {split_date.date()}): {len(train_df)} papers")
    print(f"Test set (from {split_date.date()}): {len(test_df)} papers")
    print(f"Total: {len(train_df) + len(test_df)} papers")

    # Show year distribution
    if len(train_df) > 0:
        train_years = train_df["Publication Date"].dt.year.value_counts().sort_index()
        print("\nTrain set year distribution:")
        for year, count in train_years.items():
            print(f"  {int(year)}: {count}")

    if len(test_df) > 0:
        test_years = test_df["Publication Date"].dt.year.value_counts().sort_index()
        print("\nTest set year distribution:")
        for year, count in test_years.items():
            print(f"  {int(year)}: {count}")

    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute similarities with FAISS and create train/test splits"
    )
    parser.add_argument(
        "--source-dataset",
        type=str,
        default="JasonYan777/novelty-rank-embeddings",
        help="Source HuggingFace dataset (with embeddings and train/test splits)",
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
    parser.add_argument(
        "--split-date",
        type=str,
        default="2025-03-01",
        help="Date cutoff for train/test split (default: 2025-03-01)",
    )
    args = parser.parse_args()

    # Load dataset from HuggingFace
    print("=" * 80)
    print("Loading Dataset from HuggingFace")
    print("=" * 80)
    print(f"Dataset: {args.source_dataset}")

    ds = load_dataset(args.source_dataset)

    # Check if dataset has train/test splits
    if "train" not in ds or "test" not in ds:
        print("\nError: Dataset must have 'train' and 'test' splits!")
        print("Please run create_novelty_dataset.py first to create the splits.")
        return
    
    train_df = pd.DataFrame(ds["train"])
    test_df = pd.DataFrame(ds["test"])

    print(f"\nLoaded {len(train_df)} train papers")
    print(f"Loaded {len(test_df)} test papers")
    print(f"Total: {len(train_df) + len(test_df)} papers")
    print(f"Columns: {list(train_df.columns)}")

    # Check for required columns
    required_cols = [
        "proximity_embedding",
        "classification_embedding",
        "scibert_embedding",
        "Publication Date",
    ]
    missing_cols = [col for col in required_cols if col not in train_df.columns]

    if missing_cols:
        print(f"\nError: Missing required columns: {missing_cols}")
        print("Please run embeddings.py first to generate embeddings!")
        return

    # Combine train and test to build comprehensive FAISS index
    all_papers_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"\nCombined {len(all_papers_df)} papers for FAISS index")
    print(
        f"Date range: {pd.to_datetime(all_papers_df['Publication Date']).min()} to {pd.to_datetime(all_papers_df['Publication Date']).max()}"
    )

    # Define embeddings to process
    embeddings = ["proximity_embedding", "classification_embedding", "scibert_embedding"]

    # Compute similarities for each embedding type
    for embedding in embeddings:
        print("\n" + "=" * 80)
        print(f"Processing TRAIN set for {embedding}")
        print("=" * 80)
        train_df = compute_similarities_for_papers(
            train_df,
            all_papers_df,
            embedding=embedding,
            search_k=args.search_k,
        )

        print("\n" + "=" * 80)
        print(f"Processing TEST set for {embedding}")
        print("=" * 80)
        test_df = compute_similarities_for_papers(
            test_df,
            all_papers_df,
            embedding=embedding,
            search_k=args.search_k,
        )

    all_papers_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"\nCombined {len(all_papers_df)} papers with similarity features")

    # Re-split based on March 15, 2025 cutoff
    train_df, test_df = create_splits(all_papers_df, split_date=args.split_date)

    # Save locally if requested
    if args.save_local:
        os.makedirs("results", exist_ok=True)

        print("\n" + "=" * 80)
        print("Saving Local Pickle Files")
        print("=" * 80)

        train_df.to_pickle("results/data_train.pkl")
        print("Saved results/data_train.pkl")

        test_df.to_pickle("results/data_test.pkl")
        print("Saved results/data_test.pkl")

    # Push to HuggingFace
    if not args.no_push:
        print("\n" + "=" * 80)
        print("Pushing to HuggingFace")
        print("=" * 80)
        print(f"Output dataset: {args.output_dataset}")

        # Convert DataFrames to Datasets
        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

        dataset_dict.push_to_hub(args.output_dataset, private=False)

        print("Dataset pushed successfully!")
        print(f"View at: https://huggingface.co/datasets/{args.output_dataset}")

    # Final summary
    print("\n" + "=" * 80)
    print("ALL PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Train set (before {args.split_date}): {len(train_df)} papers")
    print(f"Test set (from {args.split_date}): {len(test_df)} papers")
    print(f"Total: {len(train_df) + len(test_df)} papers")
    print("\nNew columns added:")
    print("  top_10_similar: List of similar papers (embeddings and scores)")
    print("  max_similarity: Maximum similarity score")
    print("  avg_similarity: Average similarity score")
    print("\nNote: Each paper compares with similar papers from the past 1 year")
    print("=" * 80)

    # Print similarity statistics for each embedding
    print("\n" + "=" * 80)
    print("Similarity Statistics by Embedding Type")
    print("=" * 80)
    for embedding in embeddings:
        emb_prefix = embedding.replace("_embedding", "")
        max_col = f"{emb_prefix}_max_sim"
        avg_col = f"{emb_prefix}_avg_sim"
        
        # Filter out None values for statistics
        max_vals = all_papers_df[max_col].dropna()
        avg_vals = all_papers_df[avg_col].dropna()
        
        print(f"\n{embedding}:")
        if len(max_vals) > 0:
            print(f"  {max_col}: min={max_vals.min():.4f}, max={max_vals.max():.4f}, mean={max_vals.mean():.4f}")
        else:
            print(f"  {max_col}: No valid values")
            
        if len(avg_vals) > 0:
            print(f"  {avg_col}: min={avg_vals.min():.4f}, max={avg_vals.max():.4f}, mean={avg_vals.mean():.4f}")
        else:
            print(f"  {avg_col}: No valid values")
    print("=" * 80)

if __name__ == "__main__":
    main()