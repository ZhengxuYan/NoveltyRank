"""
Filter arXiv papers by conference acceptance

This script filters papers from arxiv_results_accepted.csv that were accepted
at top ML/AI/CV/NLP conferences.

Supported conferences:
- NeurIPS (Neural Information Processing Systems)
- ICML (International Conference on Machine Learning)
- ICLR (International Conference on Learning Representations)
- AAAI (Association for the Advancement of Artificial Intelligence)
- CVPR (Computer Vision and Pattern Recognition)
- ICCV (International Conference on Computer Vision)
- ACL (Association for Computational Linguistics)
- EMNLP (Empirical Methods in Natural Language Processing)

Usage:
    # Filter all conferences
    python filter_arxiv_by_conference.py

    # Filter specific conferences
    python filter_arxiv_by_conference.py --conferences neurips icml iclr

    # Specify custom input/output paths
    python filter_arxiv_by_conference.py --input custom.csv --output filtered.csv
"""

import pandas as pd
import re
from datetime import datetime
import argparse
import os


class ConferenceFilter:
    """Filter arXiv papers by conference acceptance"""

    # Define conference name variations to search for
    CONFERENCE_PATTERNS = {
        "neurips": [
            r"neurips",
            r"neural information processing systems",
            r"\bnips\b",  # word boundary to avoid matching "snippets"
        ],
        "icml": [
            r"icml",
            r"international conference on machine learning",
        ],
        "iclr": [
            r"iclr",
            r"international conference on learning representations",
        ],
        "aaai": [
            r"aaai",
            r"association for the advancement of artificial intelligence",
        ],
        "cvpr": [
            r"cvpr",
            r"computer vision and pattern recognition",
            r"ieee.*conference on computer vision",
        ],
        "iccv": [
            r"iccv",
            r"international conference on computer vision",
        ],
        "acl": [
            r"\bacl\b",  # word boundary to avoid false positives
            r"association for computational linguistics",
            r"annual meeting.*association for computational linguistics",
        ],
        "emnlp": [
            r"emnlp",
            r"empirical methods in natural language processing",
        ],
    }

    def __init__(self, input_csv, output_csv=None, conferences=None):
        """
        Initialize the filter

        Args:
            input_csv: Path to input arXiv CSV
            output_csv: Path for output CSV (auto-generated if None)
            conferences: List of conference names to filter (all if None)
        """
        self.input_csv = input_csv
        self.conferences = (
            [c.lower() for c in conferences]
            if conferences
            else list(self.CONFERENCE_PATTERNS.keys())
        )

        if output_csv is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            conferences_str = "_".join(sorted(self.conferences))
            self.output_csv = (
                f"results/arxiv_conference_papers_{conferences_str}_{timestamp}.csv"
            )
        else:
            self.output_csv = output_csv

        # Statistics
        self.total_papers = 0
        self.filtered_papers = 0
        self.conference_counts = {conf: 0 for conf in self.conferences}

    def matches_conference(self, text, conference):
        """
        Check if text mentions a specific conference

        Args:
            text: Text to search (comment or journal reference)
            conference: Conference name to look for

        Returns:
            bool: True if conference is mentioned
        """
        if pd.isna(text) or not text:
            return False

        text_lower = text.lower()
        patterns = self.CONFERENCE_PATTERNS.get(conference, [])

        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def classify_paper(self, row):
        """
        Classify which conference(s) a paper belongs to

        Args:
            row: DataFrame row with paper data

        Returns:
            list: List of conference names the paper was accepted at
        """
        comment = row.get("Comment", "")
        journal_ref = row.get("Journal Reference", "")

        # Combine both fields for searching
        search_text = f"{comment} {journal_ref}"

        matched_conferences = []
        for conf in self.conferences:
            if self.matches_conference(search_text, conf):
                matched_conferences.append(conf)

        return matched_conferences

    def filter_papers(self):
        """Filter papers by conference acceptance"""
        print(f"Loading papers from {self.input_csv}...")
        df = pd.read_csv(self.input_csv)
        self.total_papers = len(df)
        print(f"Total papers: {self.total_papers}")

        print(
            f"\nFiltering for conferences: {', '.join([c.upper() for c in self.conferences])}"
        )
        print("=" * 80)

        # Add a column to store matched conferences
        matched_conferences_list = []
        filtered_rows = []

        for idx, row in df.iterrows():
            if (idx + 1) % 5000 == 0:
                print(f"Processed {idx + 1}/{self.total_papers} papers...")

            conferences = self.classify_paper(row)

            if conferences:
                matched_conferences_list.append(
                    "; ".join([c.upper() for c in conferences])
                )
                filtered_rows.append(idx)

                # Update counts
                for conf in conferences:
                    self.conference_counts[conf] += 1
                self.filtered_papers += 1

        # Create filtered dataframe
        filtered_df = df.loc[filtered_rows].copy()
        filtered_df["Matched Conferences"] = matched_conferences_list

        # Create output directory
        os.makedirs("results", exist_ok=True)

        # Save to CSV
        filtered_df.to_csv(self.output_csv, index=False)

        # Print statistics
        print("\n" + "=" * 80)
        print("FILTERING COMPLETE")
        print("=" * 80)
        print(f"Total papers processed: {self.total_papers:,}")
        print(
            f"Papers from target conferences: {self.filtered_papers:,} ({self.filtered_papers/self.total_papers*100:.2f}%)"
        )
        print("\nPapers by conference:")
        for conf in sorted(self.conferences):
            count = self.conference_counts[conf]
            if count > 0:
                print(f"  {conf.upper()}: {count:,}")
        print(f"\nResults saved to: {self.output_csv}")
        print()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Filter arXiv papers by conference acceptance"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../arxiv_scraper/arxiv_scraper/results/arxiv_results_accepted.csv",
        help="Path to input arXiv CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (auto-generated if not provided)",
    )
    parser.add_argument(
        "--conferences",
        type=str,
        nargs="*",
        default=None,
        choices=["neurips", "icml", "iclr", "aaai", "cvpr", "iccv", "acl", "emnlp"],
        help="Conferences to filter (all if not specified)",
    )

    args = parser.parse_args()

    # Initialize filter
    filter_obj = ConferenceFilter(
        input_csv=args.input,
        output_csv=args.output,
        conferences=args.conferences,
    )

    # Filter papers
    filter_obj.filter_papers()


if __name__ == "__main__":
    main()
