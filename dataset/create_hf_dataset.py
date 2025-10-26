"""
Script to create and upload labeled datasets to Hugging Face Hub

This script takes positive and negative datapoints from files, adds labels,
and uploads the combined dataset to Hugging Face.

Supported file formats: CSV, JSON, Parquet
"""

import os
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import login
from pathlib import Path


class DatasetCreator:
    """Create and upload labeled datasets to Hugging Face"""

    def __init__(self, hf_token=None):
        """
        Initialize the dataset creator

        Args:
            hf_token: Hugging Face API token (if None, will check environment or prompt)
        """
        self.hf_token = hf_token
        self._authenticate()

    def _authenticate(self):
        """Authenticate with Hugging Face"""
        # Try to get token from various sources
        if self.hf_token is None:
            self.hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
                "HUGGINGFACE_TOKEN"
            )

        if self.hf_token:
            print("Authenticating with Hugging Face...")
            login(token=self.hf_token)
            print("✓ Authentication successful!")
        else:
            print("No HF token provided. You may be prompted to login.")
            print("Set HF_TOKEN environment variable or use --hf-token argument.")

    def load_file(self, file_path):
        """
        Load data from a file (CSV, JSON, or Parquet)

        Args:
            file_path: Path to the file

        Returns:
            pandas.DataFrame: Loaded data
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"Loading {file_path.name}...")

        # Determine file type and load accordingly
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == ".json":
            df = pd.read_json(file_path)
        elif file_path.suffix.lower() in [".parquet", ".pq"]:
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def parse_file_source_pairs(self, file_specs, default_source):
        """
        Parse file specifications that can be either 'path' or 'path:source'

        Args:
            file_specs: List of file specifications
            default_source: Default source name if not specified

        Returns:
            List of tuples: [(file_path, source_name), ...]
        """
        parsed = []
        for spec in file_specs:
            if ":" in spec:
                # Split on first colon only
                parts = spec.split(":", 1)
                file_path = parts[0]
                source = parts[1]
            else:
                file_path = spec
                source = default_source
            parsed.append((file_path, source))
        return parsed

    def create_labeled_dataset(
        self,
        positive_files,
        negative_files,
        label_column="label",
        source_column="source",
        default_positive_source="Positive",
        default_negative_source="Random",
    ):
        """
        Create a labeled dataset from positive and negative files

        Args:
            positive_files: List of file paths or 'path:source' pairs for positive examples
            negative_files: List of file paths or 'path:source' pairs for negative examples
            label_column: Name of the label column (default: 'label')
            source_column: Name of the source column (default: 'source')
            default_positive_source: Default source for positive examples if not specified
            default_negative_source: Default source for negative examples if not specified

        Returns:
            pandas.DataFrame: Combined labeled dataset
        """
        all_data = []

        # Load positive examples
        if positive_files:
            print("\n" + "=" * 60)
            print("Loading POSITIVE examples (label=1)")
            print("=" * 60)

            file_source_pairs = self.parse_file_source_pairs(
                positive_files, default_positive_source
            )

            for file_path, source in file_source_pairs:
                df = self.load_file(file_path)
                df[label_column] = 1
                df[source_column] = source
                print(f"  → Source: {source}")
                all_data.append(df)

            print(f"Total positive examples: {sum(len(df) for df in all_data)}")

        # Load negative examples
        if negative_files:
            print("\n" + "=" * 60)
            print("Loading NEGATIVE examples (label=0)")
            print("=" * 60)

            neg_start_idx = len(all_data)
            file_source_pairs = self.parse_file_source_pairs(
                negative_files, default_negative_source
            )

            for file_path, source in file_source_pairs:
                df = self.load_file(file_path)
                df[label_column] = 0
                df[source_column] = source
                print(f"  → Source: {source}")
                all_data.append(df)

            neg_count = sum(len(df) for df in all_data[neg_start_idx:])
            print(f"Total negative examples: {neg_count}")

        if not all_data:
            raise ValueError(
                "No data loaded! Provide at least one positive or negative file."
            )

        # Combine all data
        print("\n" + "=" * 60)
        print("Combining datasets...")
        combined_df = pd.concat(all_data, ignore_index=True)

        # Shuffle the dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"✓ Created dataset with {len(combined_df)} total examples")
        print(f"  Positive: {(combined_df[label_column] == 1).sum()}")
        print(f"  Negative: {(combined_df[label_column] == 0).sum()}")

        # Show source distribution
        print("\n  Source distribution:")
        for source, count in combined_df[source_column].value_counts().items():
            print(f"    {source}: {count}")

        return combined_df

    def create_train_test_split(self, df, test_size=0.2, label_column="label"):
        """
        Split dataset into train and test sets

        Args:
            df: pandas DataFrame
            test_size: Fraction of data for test set (default: 0.2)
            label_column: Name of the label column

        Returns:
            tuple: (train_df, test_df)
        """
        from sklearn.model_selection import train_test_split

        print(f"\nSplitting dataset (train={1-test_size:.0%}, test={test_size:.0%})...")

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=df[label_column],  # Maintain label distribution
        )

        print(f"  Train set: {len(train_df)} examples")
        print(f"  Test set: {len(test_df)} examples")

        return train_df, test_df

    def upload_to_huggingface(
        self, df, repo_name, private=True, split_data=False, test_size=0.2
    ):
        """
        Upload dataset to Hugging Face Hub

        Args:
            df: pandas DataFrame with the dataset
            repo_name: Name of the repository on Hugging Face (e.g., 'username/dataset-name')
            private: Whether to make the dataset private (default: True)
            split_data: Whether to split into train/test sets (default: False)
            test_size: Fraction of data for test set if split_data=True (default: 0.2)
        """
        print("\n" + "=" * 60)
        print("Creating Hugging Face dataset...")
        print("=" * 60)

        if split_data:
            train_df, test_df = self.create_train_test_split(df, test_size=test_size)

            dataset_dict = DatasetDict(
                {
                    "train": Dataset.from_pandas(train_df, preserve_index=False),
                    "test": Dataset.from_pandas(test_df, preserve_index=False),
                }
            )
        else:
            dataset_dict = Dataset.from_pandas(df, preserve_index=False)

        print(f"\nUploading to Hugging Face Hub: {repo_name}")
        print(f"Privacy: {'Private' if private else 'Public'}")

        try:
            dataset_dict.push_to_hub(repo_name, private=private, token=self.hf_token)

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

    def save_locally(self, df, output_path, format="csv"):
        """
        Save dataset locally

        Args:
            df: pandas DataFrame
            output_path: Path to save the file
            format: Output format ('csv', 'json', 'parquet')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving dataset locally to {output_path}...")

        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", lines=True)
        elif format == "parquet":
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"✓ Saved {len(df)} examples to {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Create and upload labeled datasets to Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - each file with its own source label
  python create_hf_dataset.py \\
    --positive iclr_2024.csv:ICLR-2024 neurips_2024.csv:NeurIPS-2024 \\
    --negative random1.csv:Random random2.csv:ArXiv \\
    --repo username/my-dataset

  # Mix explicit and default sources
  python create_hf_dataset.py \\
    --positive iclr_2024.csv:ICLR-2024 accepted.csv \\
    --negative random.csv \\
    --default-positive-source "Conference" \\
    --default-negative-source "Random" \\
    --repo username/my-dataset

  # Split into train/test and save locally
  python create_hf_dataset.py \\
    --positive positive.csv:ICLR-2024 \\
    --negative negative.csv:Random \\
    --repo username/my-dataset \\
    --split \\
    --test-size 0.2 \\
    --save-local output.csv

  # Use environment variable for token
  export HF_TOKEN=your_token_here
  python create_hf_dataset.py --positive pos.csv:Source1 --negative neg.csv:Source2 --repo username/dataset
        """,
    )

    # Input files
    parser.add_argument(
        "--positive",
        nargs="+",
        required=True,
        help='Positive examples (label=1). Format: "path" or "path:source". Example: "data.csv:ICLR-2024"',
    )
    parser.add_argument(
        "--negative",
        nargs="+",
        required=True,
        help='Negative examples (label=0). Format: "path" or "path:source". Example: "data.csv:Random"',
    )

    # Output configuration
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Hugging Face repository name (e.g., username/dataset-name)",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Name of the label column (default: label)",
    )
    parser.add_argument(
        "--source-column",
        type=str,
        default="source",
        help="Name of the source column (default: source)",
    )
    parser.add_argument(
        "--default-positive-source",
        type=str,
        default="Positive",
        help="Default source name for positive files without explicit source (default: Positive)",
    )
    parser.add_argument(
        "--default-negative-source",
        type=str,
        default="Random",
        help="Default source name for negative files without explicit source (default: Random)",
    )

    # Authentication
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN environment variable)",
    )

    # Privacy and splitting
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the dataset public (default: private)",
    )
    parser.add_argument(
        "--split", action="store_true", help="Split dataset into train/test sets"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test set if --split is used (default: 0.2)",
    )

    # Local save
    parser.add_argument(
        "--save-local",
        type=str,
        default=None,
        help="Save dataset locally to this path (optional)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "json", "parquet"],
        help="Format for local save (default: csv)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.test_size <= 0 or args.test_size >= 1:
        raise ValueError("test-size must be between 0 and 1")

    # Initialize creator
    creator = DatasetCreator(hf_token=args.hf_token)

    # Create labeled dataset
    df = creator.create_labeled_dataset(
        positive_files=args.positive,
        negative_files=args.negative,
        label_column=args.label_column,
        source_column=args.source_column,
        default_positive_source=args.default_positive_source,
        default_negative_source=args.default_negative_source,
    )

    # Save locally if requested
    if args.save_local:
        creator.save_locally(df, args.save_local, format=args.format)

    # Upload to Hugging Face
    creator.upload_to_huggingface(
        df,
        repo_name=args.repo,
        private=not args.public,
        split_data=args.split,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
