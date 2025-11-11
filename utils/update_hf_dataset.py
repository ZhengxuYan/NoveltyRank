import pickle
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import login
import os


def load_pickle_data(train_path, test_path):
    print("=" * 60)
    print("Loading pickle files...")
    print("=" * 60)

    with open(train_path, "rb") as f:
        train_df = pickle.load(f)
    print(f"✓ Loaded train data: {len(train_df)} samples")
    print(f"  Columns: {list(train_df.columns)}")

    with open(test_path, "rb") as f:
        test_df = pickle.load(f)
    print(f"✓ Loaded test data: {len(test_df)} samples")
    print(f"  Columns: {list(test_df.columns)}")

    return train_df, test_df


def convert_numpy_to_list(df):
    print("\n" + "=" * 60)
    print("Converting numpy arrays to lists...")
    print("=" * 60)

    df_copy = df.copy()

    for col in df_copy.columns:
        # Check if column contains numpy arrays
        if df_copy[col].dtype == "object":
            first_non_null = (
                df_copy[col].dropna().iloc[0] if df_copy[col].notna().any() else None
            )

            if first_non_null is not None:
                # Convert numpy arrays to lists
                if hasattr(first_non_null, "tolist"):
                    print(f"  Converting {col} from numpy array to list...")
                    df_copy[col] = df_copy[col].apply(
                        lambda x: x.tolist() if hasattr(x, "tolist") else x
                    )

    print("✓ Conversion complete")
    return df_copy


def convert_dict_to_json_string(df):
    import json

    print("\n" + "=" * 60)
    print("Converting dictionaries to JSON strings...")
    print("=" * 60)

    df_copy = df.copy()

    # Handle top_10_similar column specifically
    if "top_10_similar" in df_copy.columns:
        print("  Converting top_10_similar from dict to JSON string...")
        df_copy["top_10_similar"] = df_copy["top_10_similar"].apply(
            lambda x: json.dumps(x)
            if isinstance(x, dict)
            else (json.dumps({}) if pd.isna(x) else x)
        )
        print("  ✓ Converted top_10_similar")

    print("✓ Conversion complete")
    return df_copy


def upload_to_huggingface(
    train_df, test_df, repo_name, hf_token=None, private=True, convert_arrays=True
):
    # Authenticate
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if hf_token:
        print("\n" + "=" * 60)
        print("Authenticating with Hugging Face...")
        print("=" * 60)
        login(token=hf_token)
        print("✓ Authentication successful!")
    else:
        print("\nNo HF token provided. You may be prompted to login.")

    # Convert numpy arrays to lists if requested
    if convert_arrays:
        train_df = convert_numpy_to_list(train_df)
        test_df = convert_numpy_to_list(test_df)

    # Convert dictionary columns to JSON strings (required for variable-key dicts)
    train_df = convert_dict_to_json_string(train_df)
    test_df = convert_dict_to_json_string(test_df)

    # Create dataset dictionary
    print("\n" + "=" * 60)
    print("Creating HuggingFace dataset...")
    print("=" * 60)

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )

    print(f"✓ Train split: {len(dataset_dict['train'])} examples")
    print(f"✓ Test split: {len(dataset_dict['test'])} examples")
    print(f"✓ Features: {list(dataset_dict['train'].features.keys())}")

    # Upload to HuggingFace
    print("\n" + "=" * 60)
    print(f"Uploading to HuggingFace Hub: {repo_name}")
    print(f"Privacy: {'Private' if private else 'Public'}")
    print("=" * 60)

    try:
        dataset_dict.push_to_hub(repo_name, private=private, token=hf_token)

        print("\n" + "=" * 60)
        print("✓ Dataset uploaded successfully!")
        print("=" * 60)
        print(f"View your dataset at: https://huggingface.co/datasets/{repo_name}")

    except Exception as e:
        print(f"\n✗ Error uploading dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your Hugging Face token is valid")
        print("  2. Ensure you have write permissions")
        print("  3. Verify the repo name format (username/dataset-name)")
        raise


def print_dataset_summary(train_df, test_df):
    """Print summary statistics of the dataset"""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print("\nTRAIN SET:")
    print(f"  Size: {len(train_df)}")
    print(f"  Columns: {len(train_df.columns)}")
    print(f"  Label distribution:")
    print(f"    Positive (1): {(train_df['label'] == 1).sum()}")
    print(f"    Negative (0): {(train_df['label'] == 0).sum()}")
    if "source" in train_df.columns:
        print(f"  Source distribution:")
        for source, count in train_df["source"].value_counts().items():
            print(f"    {source}: {count}")

    print("\nTEST SET:")
    print(f"  Size: {len(test_df)}")
    print(f"  Columns: {len(test_df.columns)}")
    print(f"  Label distribution:")
    print(f"    Positive (1): {(test_df['label'] == 1).sum()}")
    print(f"    Negative (0): {(test_df['label'] == 0).sum()}")
    if "source" in test_df.columns:
        print(f"  Source distribution:")
        for source, count in test_df["source"].value_counts().items():
            print(f"    {source}: {count}")

    print("\nCOLUMNS:")
    for col in train_df.columns:
        print(f"  - {col}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Update HuggingFace dataset with new columns from pickle files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update dataset with embeddings and similarity features
  python update_hf_dataset.py \\
    --train-pickle results/data_train.pkl \\
    --test-pickle results/data_test.pkl \\
    --repo JasonYan777/novelty-dataset
  
  # Use custom HF token and make public
  python update_hf_dataset.py \\
    --train-pickle results/data_train.pkl \\
    --test-pickle results/data_test.pkl \\
    --repo JasonYan777/novelty-dataset-v2 \\
    --hf-token YOUR_TOKEN \\
    --public
        """,
    )

    # Input files
    parser.add_argument(
        "--train-pickle", type=str, required=True, help="Path to training pickle file"
    )
    parser.add_argument(
        "--test-pickle", type=str, required=True, help="Path to test pickle file"
    )

    # HuggingFace configuration
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repository name (e.g., username/dataset-name)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the dataset public (default: private)",
    )
    parser.add_argument(
        "--no-convert-arrays",
        action="store_true",
        help="Skip converting numpy arrays to lists (may cause upload issues)",
    )

    args = parser.parse_args()

    # Load data
    train_df, test_df = load_pickle_data(args.train_pickle, args.test_pickle)

    # Print summary
    print_dataset_summary(train_df, test_df)

    # Upload to HuggingFace
    upload_to_huggingface(
        train_df=train_df,
        test_df=test_df,
        repo_name=args.repo,
        hf_token=args.hf_token,
        private=not args.public,
        convert_arrays=not args.no_convert_arrays,
    )

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
