"""
Find ICLR accepted papers on arXiv

This script:
1. Reads iclr26v1_papers.csv
2. Filters for accepted papers by year(s)
3. Searches each paper on arXiv by title
4. Outputs results in the same format as arxiv_scraper output

Usage examples:
    # Single year (default: 2024)
    python find_iclr_on_arxiv.py --year 2024

    # Multiple years
    python find_iclr_on_arxiv.py --year 2023 2024

    # All years
    python find_iclr_on_arxiv.py --all-years

    # All papers (not just accepted)
    python find_iclr_on_arxiv.py --year 2024 --all-papers
"""

import pandas as pd
import requests
import time
import csv
import xml.etree.ElementTree as ET
import urllib.parse
from datetime import datetime
import os


class ICLRArxivMatcher:
    """Match ICLR papers with arXiv versions"""

    def __init__(self, iclr_csv_path, output_csv_path=None, year=None):
        """
        Initialize the matcher

        Args:
            iclr_csv_path: Path to ICLR papers CSV
            output_csv_path: Path for output CSV (auto-generated if None)
            year: Year for output filename (only used if output_csv_path is None)
        """
        self.iclr_csv_path = iclr_csv_path

        if output_csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            year_str = str(year) if year else "all"
            self.output_csv_path = f"results/iclr_{year_str}_on_arxiv_{timestamp}.csv"
        else:
            self.output_csv_path = output_csv_path

        # arXiv API settings
        self.arxiv_api_url = "https://export.arxiv.org/api/query"
        self.request_delay = 3  # arXiv recommends 3 seconds between requests

        # Statistics
        self.papers_checked = 0
        self.papers_found = 0
        self.papers_not_found = 0

    def load_iclr_papers(self, year=None, accept_only=True):
        """
        Load and filter ICLR papers

        Args:
            year: Year to filter (int or list of ints). If None, loads all years.
            accept_only: If True, only load accepted papers
        """
        print(f"Loading ICLR papers from {self.iclr_csv_path}...")
        df = pd.read_csv(self.iclr_csv_path)
        print(f"Total papers in dataset: {len(df)}")

        # Filter by year
        if year is not None:
            if isinstance(year, list):
                df = df[df["year"].isin(year)]
                print(f"Found {len(df)} papers from years {year}")
            else:
                df = df[df["year"] == year]
                print(f"Found {len(df)} papers from {year}")
        else:
            print("Loading papers from all years")

        # Filter by decision (accepted papers only)
        if accept_only:
            df = df[df["decision"].str.contains("Accept", na=False, case=False)]
            print(f"Found {len(df)} accepted papers")

        return df

    def search_arxiv_by_title(self, title, fuzzy=True):
        """
        Search arXiv by paper title

        Args:
            title: Paper title to search for
            fuzzy: If True and exact match fails, try fuzzy matching

        Returns:
            dict: Paper metadata if found, None otherwise
        """
        # Clean up title for search
        title_clean = title.strip()

        # Try exact match first
        search_query = f'ti:"{title_clean}"'
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": 5,  # Get top 5 results to check for fuzzy matches
        }

        url = f"{self.arxiv_api_url}?{urllib.parse.urlencode(params)}"

        try:
            response = requests.get(url)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.content)

            # Define namespaces
            namespaces = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            # Get entries
            entries = root.findall("atom:entry", namespaces)

            if len(entries) == 0:
                # If no exact match and fuzzy is enabled, try without quotes
                if fuzzy:
                    # Extract key words from title (remove common words)
                    title_words = title_clean.lower().split()
                    key_words = [w for w in title_words if len(w) > 3][
                        :5
                    ]  # First 5 words > 3 chars

                    if len(key_words) >= 2:
                        # Search with key words
                        search_query = "ti:" + "+AND+ti:".join(key_words)
                        params = {
                            "search_query": search_query,
                            "start": 0,
                            "max_results": 10,
                        }

                        url = f"{self.arxiv_api_url}?{urllib.parse.urlencode(params)}"
                        response = requests.get(url)
                        response.raise_for_status()
                        root = ET.fromstring(response.content)
                        entries = root.findall("atom:entry", namespaces)

                if len(entries) == 0:
                    return None

            # Find best matching entry by title similarity
            best_entry = None
            best_score = 0

            for entry in entries:
                title_elem = entry.find("atom:title", namespaces)
                if title_elem is not None:
                    arxiv_title = title_elem.text.strip().replace("\n", " ").lower()
                    iclr_title = title_clean.lower()

                    # Simple similarity: count matching words
                    arxiv_words = set(arxiv_title.split())
                    iclr_words = set(iclr_title.split())

                    if len(iclr_words) > 0:
                        similarity = len(arxiv_words & iclr_words) / len(iclr_words)

                        if similarity > best_score:
                            best_score = similarity
                            best_entry = entry

            # Only return if similarity is high enough (>70%)
            if best_entry is None or best_score < 0.7:
                return None

            entry = best_entry

            # Extract metadata
            paper = {}

            # arXiv URL/ID
            id_elem = entry.find("atom:id", namespaces)
            paper["arxiv_url"] = id_elem.text if id_elem is not None else ""

            # Title
            title_elem = entry.find("atom:title", namespaces)
            paper["title"] = (
                title_elem.text.strip().replace("\n", " ")
                if title_elem is not None
                else ""
            )

            # Published date
            published_elem = entry.find("atom:published", namespaces)
            if published_elem is not None:
                published_date = datetime.fromisoformat(
                    published_elem.text.replace("Z", "+00:00")
                )
                paper["publication_date"] = published_date.date().isoformat()
            else:
                paper["publication_date"] = ""

            # Authors
            author_elems = entry.findall("atom:author", namespaces)
            authors = []
            for author in author_elems:
                name_elem = author.find("atom:name", namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text)
            paper["authors"] = "; ".join(authors)

            # Abstract
            summary_elem = entry.find("atom:summary", namespaces)
            paper["abstract"] = (
                summary_elem.text.strip().replace("\n", " ")
                if summary_elem is not None
                else ""
            )

            # Categories
            category_elems = entry.findall("atom:category", namespaces)
            categories = []
            for category in category_elems:
                term = category.get("term")
                if term:
                    categories.append(term)
            paper["categories"] = "; ".join(categories)

            return paper

        except Exception as e:
            print(f"Error searching for '{title}': {e}")
            return None

    def match_papers(self, year=None, accept_only=True):
        """
        Match ICLR papers with arXiv versions

        Args:
            year: ICLR year to process (int, list of ints, or None for all years)
            accept_only: Only process accepted papers
        """
        # Load ICLR papers
        iclr_df = self.load_iclr_papers(year=year, accept_only=accept_only)

        print(f"\nSearching for {len(iclr_df)} papers on arXiv...")
        print(
            f"This will take approximately {len(iclr_df) * self.request_delay / 60:.1f} minutes"
        )
        print("=" * 80)

        # Create output directory
        os.makedirs("results", exist_ok=True)

        # Open output CSV file
        with open(self.output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header (same format as arxiv.py)
            writer.writerow(
                [
                    "Link/DOI",
                    "Publication Date",
                    "Title",
                    "Authors",
                    "Abstract",
                    "Categories",
                ]
            )

            # Process each paper
            for idx, row in iclr_df.iterrows():
                self.papers_checked += 1

                # Progress indicator
                if self.papers_checked % 100 == 0:
                    print(
                        f"Processed {self.papers_checked}/{len(iclr_df)} papers "
                        f"(Found: {self.papers_found}, Not found: {self.papers_not_found})"
                    )

                title = row["title"]

                # Search arXiv
                arxiv_paper = self.search_arxiv_by_title(title)

                if arxiv_paper:
                    # Write to CSV
                    writer.writerow(
                        [
                            arxiv_paper["arxiv_url"],
                            arxiv_paper["publication_date"],
                            arxiv_paper["title"],
                            arxiv_paper["authors"],
                            arxiv_paper["abstract"],
                            arxiv_paper["categories"],
                        ]
                    )

                    self.papers_found += 1
                else:
                    self.papers_not_found += 1

                # Rate limiting - wait 3 seconds between requests
                time.sleep(self.request_delay)

        print("\n" + "=" * 80)
        print("MATCHING COMPLETE")
        print("=" * 80)
        print(f"Total papers checked: {self.papers_checked}")
        print(
            f"Papers found on arXiv: {self.papers_found} ({self.papers_found/self.papers_checked*100:.1f}%)"
        )
        print(
            f"Papers NOT found: {self.papers_not_found} ({self.papers_not_found/self.papers_checked*100:.1f}%)"
        )
        print(f"\nResults saved to: {self.output_csv_path}")
        print()


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Find ICLR papers on arXiv and output in arXiv scraper format"
    )
    parser.add_argument(
        "--iclr-csv",
        type=str,
        default="results/iclr26v1_papers.csv",
        help="Path to ICLR papers CSV",
    )
    parser.add_argument(
        "--year",
        type=int,
        nargs="*",
        default=[2024],
        help="ICLR year(s) to process (space-separated). Omit to process all years.",
    )
    parser.add_argument(
        "--all-years",
        action="store_true",
        help="Process all years (overrides --year)",
    )
    parser.add_argument(
        "--all-papers",
        action="store_true",
        help="Process all papers (not just accepted ones)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (auto-generated if not provided)",
    )

    args = parser.parse_args()

    # Determine year filter
    if args.all_years:
        year_filter = None
        year_for_filename = None
    elif len(args.year) == 0:
        year_filter = None
        year_for_filename = None
    elif len(args.year) == 1:
        year_filter = args.year[0]
        year_for_filename = args.year[0]
    else:
        year_filter = args.year
        year_for_filename = "-".join(map(str, args.year))

    # Initialize matcher
    matcher = ICLRArxivMatcher(
        iclr_csv_path=args.iclr_csv,
        output_csv_path=args.output,
        year=year_for_filename,
    )

    # Match papers
    accept_only = not args.all_papers
    matcher.match_papers(year=year_filter, accept_only=accept_only)


if __name__ == "__main__":
    main()
