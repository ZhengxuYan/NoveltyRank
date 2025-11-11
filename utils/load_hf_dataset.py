import json
import pandas as pd
from datasets import load_dataset
import argparse


def convert_json_to_dict(df):
    """
    Convert JSON string columns back to dictionaries/lists

    Args:
        df: pandas DataFrame

    Returns:
        pandas DataFrame with JSON strings converted back to dicts
    """
    print("\n" + "=" * 60)
    print("Converting JSON strings back to dictionaries...")
    print("=" * 60)

    df_copy = df.copy()

    # Handle top_10_similar column
    if "top_10_similar" in df_copy.columns:
        print("  Converting top_10_similar from JSON string to dict...")
        df_copy["top_10_similar"] = df_copy["top_10_similar"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        # Replace empty dicts with None for consistency
        df_copy["top_10_similar"] = df_copy["top_10_similar"].apply(
            lambda x: None if isinstance(x, dict) and len(x) == 0 else x
        )
        print("  ✓ Converted top_10_similar")

    print("✓ Conversion complete")
    return df_copy


def load_and_convert_dataset(repo_name, convert_embeddings_to_numpy=True):
    """
    Load dataset from HuggingFace and convert formats

    Args:
        repo_name: HuggingFace repository name
        convert_embeddings_to_numpy: Whether to convert embedding lists to numpy arrays

    Returns:
        tuple: (train_df, test_df) pandas DataFrames
    """
    print("=" * 60)
    print(f"Loading dataset from: {repo_name}")
    print("=" * 60)

    # Load dataset
    dataset = load_dataset(repo_name)

    print(f"✓ Loaded dataset")
    print(f"  Train split: {len(dataset['train'])} examples")
    print(f"  Test split: {len(dataset['test'])} examples")

    # Convert to pandas
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    # Convert JSON strings back to dicts
    train_df = convert_json_to_dict(train_df)
    test_df = convert_json_to_dict(test_df)

    # Optionally convert embeddings to numpy arrays
    if convert_embeddings_to_numpy:
        import numpy as np

        print("\n" + "=" * 60)
        print("Converting embeddings to numpy arrays...")
        print("=" * 60)

        for col in ["Classification_embedding", "Proximity_embedding"]:
            if col in train_df.columns:
                print(f"  Converting {col}...")
                train_df[col] = train_df[col].apply(
                    lambda x: np.array(x) if isinstance(x, list) else x
                )
                test_df[col] = test_df[col].apply(
                    lambda x: np.array(x) if isinstance(x, list) else x
                )

        print("✓ Conversion complete")

    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description="Load HuggingFace dataset and convert formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load dataset and convert formats
  python load_hf_dataset.py --repo JasonYan777/novelty-dataset
  
  # Load and save as pickle files
  python load_hf_dataset.py \\
    --repo JasonYan777/novelty-dataset \\
    --save-pickle \\
    --output-dir results
        """,
    )

    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repository name (e.g., username/dataset-name)",
    )
    parser.add_argument(
        "--save-pickle",
        action="store_true",
        help="Save as pickle files after conversion",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for pickle files (default: current directory)",
    )
    parser.add_argument(
        "--no-numpy",
        action="store_true",
        help="Skip converting embeddings to numpy arrays",
    )

    args = parser.parse_args()

    # Load and convert dataset
    train_df, test_df = load_and_convert_dataset(
        repo_name=args.repo, convert_embeddings_to_numpy=not args.no_numpy
    )

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nTrain set: {len(train_df)} rows, {len(train_df.columns)} columns")
    print(f"Test set: {len(test_df)} rows, {len(test_df.columns)} columns")
    print(f"\nColumns: {list(train_df.columns)}")

    # Save as pickle if requested
    if args.save_pickle:
        import os

        os.makedirs(args.output_dir, exist_ok=True)

        train_path = os.path.join(args.output_dir, "data_train.pkl")
        test_path = os.path.join(args.output_dir, "data_test.pkl")

        print("\n" + "=" * 60)
        print("Saving as pickle files...")
        print("=" * 60)

        train_df.to_pickle(train_path)
        print(f"✓ Saved train set to: {train_path}")

        test_df.to_pickle(test_path)
        print(f"✓ Saved test set to: {test_path}")

    print("\n✓ All done!")

    return train_df, test_df


if __name__ == "__main__":
    main()
