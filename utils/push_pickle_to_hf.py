import argparse
import os
import sys
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import login


def load_pickle_with_compatibility(pickle_file):
    """Load pickle file with numpy version compatibility handling"""
    import pickle

    # First, try standard pandas read_pickle
    try:
        df = pd.read_pickle(pickle_file)
        return df
    except (ModuleNotFoundError, AttributeError) as e:
        if "numpy" in str(e).lower():
            print("\n⚠️  Numpy version mismatch detected. Trying compatibility mode...")
            # Try loading with pickle directly
            with open(pickle_file, "rb") as f:
                df = pickle.load(f)
            return df
        else:
            raise


def push_pickle_to_hf(train_file, test_file, repo_name, private=False, hf_token=None):
    """
    Push train and test pickle files to Hugging Face Hub as a DatasetDict

    Args:
        train_file: Path to train pickle file
        test_file: Path to test pickle file
        repo_name: Hugging Face repository name (e.g., 'username/dataset-name')
        private: Whether to make the dataset private
        hf_token: Hugging Face API token
    """
    print("=" * 80)
    print("PUSH TRAIN/TEST DATASETS TO HUGGING FACE")
    print("=" * 80)

    # Check if files exist
    train_path = Path(train_file)
    test_path = Path(test_file)

    if not train_path.exists():
        print(f"\n❌ Error: Train file not found: {train_file}")
        return False

    if not test_path.exists():
        print(f"\n❌ Error: Test file not found: {test_file}")
        return False

    # Get file sizes
    train_size_mb = train_path.stat().st_size / (1024 * 1024)
    test_size_mb = test_path.stat().st_size / (1024 * 1024)
    print(f"\nTrain file: {train_path.name} ({train_size_mb:.2f} MB)")
    print(f"Test file: {test_path.name} ({test_size_mb:.2f} MB)")

    # Authenticate
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if hf_token:
        print("\nAuthenticating with Hugging Face...")
        login(token=hf_token)
        print("✓ Authentication successful!")
    else:
        print("\nNo HF token provided. You may be prompted to login.")
        print("Set HF_TOKEN environment variable or use --hf-token argument.")

    # Load pickle files
    print("\n" + "=" * 80)
    print("Loading pickle files...")
    print("=" * 80)

    try:
        # Load train file
        print("\nLoading train dataset...")
        train_df = load_pickle_with_compatibility(train_file)
        print(f"✓ Loaded {len(train_df)} train examples")
        print(f"✓ Columns: {len(train_df.columns)}")

        # Load test file
        print("\nLoading test dataset...")
        test_df = load_pickle_with_compatibility(test_file)
        print(f"✓ Loaded {len(test_df)} test examples")

        # Show column names (from train)
        print("\nColumn names:")
        for col in train_df.columns:
            print(f"  - {col}")

        # Show memory usage
        train_memory_mb = train_df.memory_usage(deep=True).sum() / (1024 * 1024)
        test_memory_mb = test_df.memory_usage(deep=True).sum() / (1024 * 1024)
        print("\nMemory usage:")
        print(f"  Train: {train_memory_mb:.2f} MB")
        print(f"  Test: {test_memory_mb:.2f} MB")
        print(f"  Total: {train_memory_mb + test_memory_mb:.2f} MB")

    except Exception as e:
        print(f"❌ Error loading pickle files: {e}")
        print("\n📝 Troubleshooting:")
        print("  This error usually means the pickle was created with a different")
        print("  version of numpy or pandas than what's currently installed.")
        print("\n  Solutions:")
        print("  1. Update numpy: pip install --upgrade numpy")
        print("  2. Or downgrade numpy: pip install numpy==1.26.4")
        print(
            "  3. Check numpy version: python -c 'import numpy; print(numpy.__version__)'"
        )
        return False

    # Convert to Hugging Face DatasetDict
    print("\n" + "=" * 80)
    print("Converting to Hugging Face DatasetDict...")
    print("=" * 80)

    try:
        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

        print("✓ Created DatasetDict with:")
        print(f"  - train: {len(train_dataset)} examples")
        print(f"  - test: {len(test_dataset)} examples")
        print(f"✓ Features: {list(train_dataset.features.keys())}")

    except Exception as e:
        print(f"❌ Error converting to Dataset: {e}")
        return False

    # Push to Hub
    print("\n" + "=" * 80)
    print("Pushing to Hugging Face Hub...")
    print("=" * 80)
    print(f"Repository: {repo_name}")
    print(f"Privacy: {'Private' if private else 'Public'}")
    print("\nThis may take a few minutes depending on dataset size...")

    try:
        dataset_dict.push_to_hub(repo_name, private=private, token=hf_token)

        print("\n" + "=" * 80)
        print("✅ SUCCESS! Dataset uploaded to Hugging Face Hub")
        print("=" * 80)
        print(f"\nDataset URL: https://huggingface.co/datasets/{repo_name}")
        print("\nYou can now load your dataset with:")
        print("\n  from datasets import load_dataset")
        print(f"  ds = load_dataset('{repo_name}')")
        print("  train_df = ds['train'].to_pandas()")
        print("  test_df = ds['test'].to_pandas()")

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR during upload")
        print("=" * 80)
        print(f"\n{str(e)}")
        print("\nTroubleshooting:")
        print("  1. Check your Hugging Face token is valid")
        print("  2. Ensure you have write permissions")
        print("  3. Verify the repo name format (username/dataset-name)")
        print("  4. Check your internet connection")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Push train and test pickle datasets to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python push_pickle_to_hf.py \\
    --train-file results/data_train.pkl \\
    --test-file results/data_test.pkl \\
    --repo username/my-dataset

  # Make it private
  python push_pickle_to_hf.py \\
    --train-file results/data_train.pkl \\
    --test-file results/data_test.pkl \\
    --repo username/my-dataset \\
    --private

  # With HF token
  python push_pickle_to_hf.py \\
    --train-file results/data_train.pkl \\
    --test-file results/data_test.pkl \\
    --repo username/my-dataset \\
    --hf-token your_token_here
        """,
    )

    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to train pickle file",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test pickle file",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Hugging Face repository name (e.g., username/dataset-name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private (default: public)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN environment variable)",
    )

    args = parser.parse_args()

    # Push to Hugging Face
    success = push_pickle_to_hf(
        train_file=args.train_file,
        test_file=args.test_file,
        repo_name=args.repo,
        private=args.private,
        hf_token=args.hf_token,
    )

    if success:
        print("\n✨ All done!")
        sys.exit(0)
    else:
        print("\n💥 Upload failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
