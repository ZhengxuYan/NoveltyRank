import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from create_hf_dataset import DatasetCreator


def main():
    parser = argparse.ArgumentParser(
        description="Create novelty ranking dataset from conference and arXiv papers"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="username/novelty-ranking-dataset",
        help="Hugging Face repository name (default: username/novelty-ranking-dataset)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        nargs="+",
        default=[2025],
        help="Year(s) for test set (default: 2025)",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Don't split into train/test sets",
    )
    parser.add_argument(
        "--random-split",
        action="store_true",
        help="Use random split instead of date-based split",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size for random split (default: 0.2)",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading to Hugging Face (only save locally)",
    )

    args = parser.parse_args()

    # File paths
    script_dir = Path(__file__).parent
    positive_file = (
        script_dir.parent
        / "conference_scraper"
        / "results"
        / "arxiv_conference_papers_aaai_acl_cvpr_emnlp_iccv_iclr_icml_neurips_20251109_163955.csv"
    )
    negative_file = (
        script_dir.parent
        / "arxiv_scraper"
        / "arxiv_scraper"
        / "results"
        / "arxiv_results_v2.csv"
    )
    output_file = script_dir.parent / "results" / "novelty_dataset.csv"

    # Check if files exist
    if not positive_file.exists():
        print(f"Error: Positive file not found: {positive_file}")
        sys.exit(1)

    if not negative_file.exists():
        print(f"Error: Negative file not found: {negative_file}")
        sys.exit(1)

    print("=" * 60)
    print("Creating Novelty Ranking Dataset")
    print("=" * 60)
    print(f"Positive examples: Conference papers")
    print(f"  File: {positive_file.name}")
    print(f"  Source: From 'Matched Conferences' column")
    print()
    print(f"Negative examples: Random arXiv papers")
    print(f"  File: {negative_file.name}")
    print(f"  Source: ArXiv")
    print()
    print(f"Output: {args.repo}")
    print(f"Local save: {output_file}")
    print("=" * 60)
    print()

    # Initialize creator
    creator = DatasetCreator(hf_token=args.hf_token)

    # Create labeled dataset
    df = creator.create_labeled_dataset(
        positive_files=[f"{positive_file}:@Matched Conferences"],
        negative_files=[f"{negative_file}:ArXiv"],
        label_column="label",
        source_column="source",
    )

    # Deduplicate by arXiv ID
    df = creator.deduplicate_by_column(df, "arXiv ID")

    # Sort by publication date
    df = creator.sort_by_date(df, "Publication Date")

    # Save locally
    creator.save_locally(df, output_file, format="csv")

    # Upload to Hugging Face (unless skip_upload is set)
    if not args.skip_upload:
        test_year = args.test_year[0] if len(args.test_year) == 1 else args.test_year

        creator.upload_to_huggingface(
            df,
            repo_name=args.repo,
            private=True,
            split_data=not args.no_split,
            test_size=args.test_size,
            date_based_split=not args.random_split,
            date_column="Publication Date",
            test_year=test_year,
        )
    else:
        print("\nSkipping upload to Hugging Face (--skip-upload flag set)")

    print()
    print("=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
