import os
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import login
from pathlib import Path


class DatasetCreator:
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
        self._authenticate()

    def _authenticate(self):
        if self.hf_token is None:
            self.hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
                "HUGGINGFACE_TOKEN"
            )

        if self.hf_token:
            print("Authenticating with Hugging Face...")
            login(token=self.hf_token)
            print("Authentication successful!")
        else:
            print("No HF token provided. You may be prompted to login.")
            print("Set HF_TOKEN environment variable or use --hf-token argument.")

    def load_file(self, file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"Loading {file_path.name}...")

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
        parsed = []
        for spec in file_specs:
            if ":" in spec:
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
        all_data = []

        if positive_files:
            print("\nLoading POSITIVE examples (label=1)")

            file_source_pairs = self.parse_file_source_pairs(
                positive_files, default_positive_source
            )

            for file_path, source in file_source_pairs:
                df = self.load_file(file_path)
                df[label_column] = 1
                
                if source.startswith("@"):
                    column_name = source[1:]
                    if column_name not in df.columns:
                        raise ValueError(f"Column '{column_name}' not found in {file_path}")
                    df[source_column] = df[column_name]
                    print(f"  Source: from column '{column_name}'")
                    unique_sources = df[source_column].value_counts()
                    for src, count in unique_sources.items():
                        print(f"    - {src}: {count}")
                else:
                    df[source_column] = source
                    print(f"  Source: {source}")
                
                all_data.append(df)

            print(f"Total positive examples: {sum(len(df) for df in all_data)}")

        if negative_files:
            print("\nLoading NEGATIVE examples (label=0)")

            neg_start_idx = len(all_data)
            file_source_pairs = self.parse_file_source_pairs(
                negative_files, default_negative_source
            )

            for file_path, source in file_source_pairs:
                df = self.load_file(file_path)
                df[label_column] = 0
                
                if source.startswith("@"):
                    column_name = source[1:]
                    if column_name not in df.columns:
                        raise ValueError(f"Column '{column_name}' not found in {file_path}")
                    df[source_column] = df[column_name]
                    print(f"  Source: from column '{column_name}'")
                    unique_sources = df[source_column].value_counts()
                    for src, count in unique_sources.items():
                        print(f"    - {src}: {count}")
                else:
                    df[source_column] = source
                    print(f"  Source: {source}")
                
                all_data.append(df)

            neg_count = sum(len(df) for df in all_data[neg_start_idx:])
            print(f"Total negative examples: {neg_count}")

        if not all_data:
            raise ValueError(
                "No data loaded! Provide at least one positive or negative file."
            )

        print("\nCombining datasets...")
        combined_df = pd.concat(all_data, ignore_index=True)

        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Created dataset with {len(combined_df)} total examples")
        print(f"  Positive: {(combined_df[label_column] == 1).sum()}")
        print(f"  Negative: {(combined_df[label_column] == 0).sum()}")

        print("\n  Source distribution:")
        for source, count in combined_df[source_column].value_counts().items():
            print(f"    {source}: {count}")

        return combined_df

    def deduplicate_by_column(self, df, column_name, keep="first"):
        """
        Remove duplicates based on a specific column

        Args:
            df: pandas DataFrame
            column_name: Name of the column to check for duplicates
            keep: Which duplicate to keep ('first', 'last', or False to drop all)

        Returns:
            pandas.DataFrame: Deduplicated dataset
        """
        original_count = len(df)

        if column_name not in df.columns:
            print(
                f"\nWarning: Column '{column_name}' not found. Skipping deduplication."
            )
            return df

        print(f"\nDeduplicating by '{column_name}'...")
        df_dedup = df.drop_duplicates(subset=[column_name], keep=keep)
        duplicates_removed = original_count - len(df_dedup)

        print(f"  Removed {duplicates_removed} duplicates")
        print(f"  Remaining: {len(df_dedup)} examples")

        return df_dedup

    def sort_by_date(self, df, date_column):
        """
        Sort dataset by date column

        Args:
            df: pandas DataFrame
            date_column: Name of the date column

        Returns:
            pandas.DataFrame: Sorted dataset
        """
        if date_column not in df.columns:
            print(
                f"\nWarning: Date column '{date_column}' not found. Skipping sorting."
            )
            return df

        print(f"\nSorting by '{date_column}'...")

        # Convert to datetime if not already
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

        # Sort by date
        df_sorted = df.sort_values(by=date_column).reset_index(drop=True)

        print(
            f"  Date range: {df_sorted[date_column].min()} to {df_sorted[date_column].max()}"
        )

        return df_sorted

    def create_train_test_split(self, df, test_size=0.2, label_column="label"):
        """
        Split dataset into train and test sets (random stratified split)

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

    def create_date_based_split(self, df, date_column, test_year=2025):
        """
        Split dataset by publication year (temporal split)

        Args:
            df: pandas DataFrame
            date_column: Name of the date column
            test_year: Year(s) to use for test set (int or list of ints)

        Returns:
            tuple: (train_df, test_df)
        """
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in dataset")

        print("\nCreating date-based split...")

        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

        # Extract year
        df["_year"] = df[date_column].dt.year

        # Handle single year or list of years
        if isinstance(test_year, int):
            test_years = [test_year]
        else:
            test_years = test_year

        # Split by year
        test_df = df[df["_year"].isin(test_years)].copy()
        train_df = df[~df["_year"].isin(test_years)].copy()

        # Remove temporary year column
        train_df = train_df.drop("_year", axis=1)
        test_df = test_df.drop("_year", axis=1)

        print(
            f"  Train set: {len(train_df)} examples (years < {min(test_years)} or > {max(test_years)})"
        )
        print(f"  Test set: {len(test_df)} examples (year(s): {test_years})")

        if len(test_df) == 0:
            print(f"\n  Warning: No papers found for test year(s) {test_years}")

        # Show year distribution
        train_years = (
            pd.to_datetime(train_df[date_column]).dt.year.value_counts().sort_index()
        )
        test_years_dist = (
            pd.to_datetime(test_df[date_column]).dt.year.value_counts().sort_index()
        )

        print("\n  Train year distribution:")
        for year, count in train_years.items():
            print(f"    {int(year)}: {count}")

        print("\n  Test year distribution:")
        for year, count in test_years_dist.items():
            print(f"    {int(year)}: {count}")

        return train_df, test_df

    def upload_to_huggingface(
        self,
        df,
        repo_name,
        private=True,
        split_data=False,
        test_size=0.2,
        date_based_split=False,
        date_column=None,
        test_year=2025,
    ):
        """
        Upload dataset to Hugging Face Hub

        Args:
            df: pandas DataFrame with the dataset
            repo_name: Name of the repository on Hugging Face (e.g., 'username/dataset-name')
            private: Whether to make the dataset private (default: True)
            split_data: Whether to split into train/test sets (default: False)
            test_size: Fraction of data for test set if split_data=True (default: 0.2)
            date_based_split: Use date-based split instead of random split
            date_column: Name of date column for date-based split
            test_year: Year(s) for test set in date-based split
        """
        print("\n" + "=" * 60)
        print("Creating Hugging Face dataset...")
        print("=" * 60)

        if split_data:
            if date_based_split:
                if date_column is None:
                    raise ValueError(
                        "date_column must be specified for date-based split"
                    )
                train_df, test_df = self.create_date_based_split(
                    df, date_column, test_year
                )
            else:
                train_df, test_df = self.create_train_test_split(
                    df, test_size=test_size
                )

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

  # Use a column from the dataframe as source (prefix with @)
  python create_hf_dataset.py \\
    --positive conference_papers.csv:@"Matched Conferences" \\
    --negative arxiv_random.csv:ArXiv \\
    --repo username/my-dataset

  # With deduplication and date sorting
  python create_hf_dataset.py \\
    --positive iclr_2024.csv:ICLR-2024 \\
    --negative random.csv:Random \\
    --deduplicate "Link/DOI" \\
    --sort-by-date \\
    --repo username/my-dataset

  # Date-based split (2025 papers as test set)
  python create_hf_dataset.py \\
    --positive positive.csv:ICLR-2024 \\
    --negative negative.csv:Random \\
    --deduplicate "Link/DOI" \\
    --sort-by-date \\
    --split \\
    --date-based-split \\
    --test-year 2025 \\
    --repo username/my-dataset

  # Random split with preprocessing
  python create_hf_dataset.py \\
    --positive positive.csv:ICLR-2024 \\
    --negative negative.csv:Random \\
    --deduplicate "Link/DOI" \\
    --sort-by-date \\
    --split \\
    --test-size 0.2 \\
    --save-local output.csv \\
    --repo username/my-dataset
        """,
    )

    # Input files
    parser.add_argument(
        "--positive",
        nargs="+",
        required=True,
        help='Positive examples (label=1). Format: "path", "path:source", or "path:@column_name". Example: "data.csv:ICLR-2024" or "data.csv:@Matched_Conferences"',
    )
    parser.add_argument(
        "--negative",
        nargs="+",
        required=True,
        help='Negative examples (label=0). Format: "path", "path:source", or "path:@column_name". Example: "data.csv:Random" or "data.csv:@Source"',
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

    # Data preprocessing
    parser.add_argument(
        "--deduplicate",
        type=str,
        default=None,
        help='Column name to deduplicate by (e.g., "Link/DOI")',
    )
    parser.add_argument(
        "--sort-by-date",
        action="store_true",
        help="Sort dataset by publication date",
    )
    parser.add_argument(
        "--date-column",
        type=str,
        default="Publication Date",
        help="Name of the date column (default: 'Publication Date')",
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
    parser.add_argument(
        "--date-based-split",
        action="store_true",
        help="Use date-based split (test set = specified year(s))",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        nargs="+",
        default=[2025],
        help="Year(s) for test set in date-based split (default: 2025)",
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

    if args.date_based_split and not args.split:
        raise ValueError("--date-based-split requires --split to be set")

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

    # Deduplicate if requested
    if args.deduplicate:
        df = creator.deduplicate_by_column(df, args.deduplicate)

    # Sort by date if requested
    if args.sort_by_date:
        df = creator.sort_by_date(df, args.date_column)

    # Save locally if requested
    if args.save_local:
        creator.save_locally(df, args.save_local, format=args.format)

    # Prepare test year argument
    test_year = args.test_year[0] if len(args.test_year) == 1 else args.test_year

    # Upload to Hugging Face
    creator.upload_to_huggingface(
        df,
        repo_name=args.repo,
        private=not args.public,
        split_data=args.split,
        test_size=args.test_size,
        date_based_split=args.date_based_split,
        date_column=args.date_column,
        test_year=test_year,
    )


if __name__ == "__main__":
    main()
