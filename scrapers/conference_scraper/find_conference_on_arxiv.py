import pandas as pd
import requests
import time
import csv
import xml.etree.ElementTree as ET
import urllib.parse
from datetime import datetime
import os


class ConferenceArxivMatcher:
    def __init__(
        self, conference_csv_path, conference_name, output_csv_path=None, year=None
    ):
        self.conference_csv_path = conference_csv_path
        self.conference_name = conference_name.lower()

        if output_csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            year_str = str(year) if year else "all"
            self.output_csv_path = (
                f"results/{self.conference_name}_{year_str}_on_arxiv_{timestamp}.csv"
            )
        else:
            self.output_csv_path = output_csv_path

        self.arxiv_api_url = "https://export.arxiv.org/api/query"
        self.request_delay = 3

        self.papers_checked = 0
        self.papers_found = 0
        self.papers_not_found = 0

    def load_conference_papers(self, year=None, accept_only=True):
        print(
            f"Loading {self.conference_name.upper()} papers from {self.conference_csv_path}..."
        )
        df = pd.read_csv(self.conference_csv_path)
        print(f"Total papers in dataset: {len(df)}")

        if year is not None:
            if isinstance(year, list):
                df = df[df["year"].isin(year)]
                print(f"Found {len(df)} papers from years {year}")
            else:
                df = df[df["year"] == year]
                print(f"Found {len(df)} papers from {year}")
        else:
            print("Loading papers from all years")

        if accept_only:
            df = df[df["decision"].str.contains("Accept", na=False, case=False)]
            print(f"Found {len(df)} accepted papers")

        return df

    def search_arxiv_by_title(self, title, fuzzy=True):
        title_clean = title.strip()

        search_query = f'ti:"{title_clean}"'
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": 5,
        }

        url = f"{self.arxiv_api_url}?{urllib.parse.urlencode(params)}"

        try:
            response = requests.get(url)
            response.raise_for_status()

            root = ET.fromstring(response.content)

            namespaces = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            entries = root.findall("atom:entry", namespaces)

            if len(entries) == 0:
                if fuzzy:
                    title_words = title_clean.lower().split()
                    key_words = [w for w in title_words if len(w) > 3][:5]

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

            # arXiv URL and ID
            id_elem = entry.find("atom:id", namespaces)
            paper["arxiv_url"] = id_elem.text if id_elem is not None else ""
            # Extract arXiv ID (e.g., "2501.12345" from "http://arxiv.org/abs/2501.12345")
            paper["arxiv_id"] = (
                paper["arxiv_url"].split("/abs/")[-1] if paper["arxiv_url"] else ""
            )

            # Get PDF URL from links
            pdf_url = ""
            link_elems = entry.findall("atom:link", namespaces)
            for link in link_elems:
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break
            paper["pdf_url"] = pdf_url

            # Get DOI
            doi_elem = entry.find("arxiv:doi", namespaces)
            paper["doi"] = doi_elem.text if doi_elem is not None else ""

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

            # Updated date
            updated_elem = entry.find("atom:updated", namespaces)
            if updated_elem is not None:
                updated_date = datetime.fromisoformat(
                    updated_elem.text.replace("Z", "+00:00")
                )
                paper["updated_date"] = updated_date.date().isoformat()
            else:
                paper["updated_date"] = ""

            # Authors and affiliations
            author_elems = entry.findall("atom:author", namespaces)
            authors = []
            affiliations = []
            for author in author_elems:
                name_elem = author.find("atom:name", namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text)

                # Get affiliation if available
                affiliation_elem = author.find("arxiv:affiliation", namespaces)
                if affiliation_elem is not None:
                    affiliations.append(f"{name_elem.text}: {affiliation_elem.text}")

            paper["authors"] = "; ".join(authors)
            paper["affiliations"] = "; ".join(affiliations) if affiliations else ""

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

            # Primary category
            primary_category_elem = entry.find("arxiv:primary_category", namespaces)
            paper["primary_category"] = (
                primary_category_elem.get("term")
                if primary_category_elem is not None
                else ""
            )

            # Comment (often contains page count, conference info, etc.)
            comment_elem = entry.find("arxiv:comment", namespaces)
            paper["comment"] = (
                comment_elem.text.strip().replace("\n", " ")
                if comment_elem is not None
                else ""
            )

            # Journal reference
            journal_ref_elem = entry.find("arxiv:journal_ref", namespaces)
            paper["journal_ref"] = (
                journal_ref_elem.text.strip().replace("\n", " ")
                if journal_ref_elem is not None
                else ""
            )

            return paper

        except Exception as e:
            print(f"Error searching for '{title}': {e}")
            return None

    def match_papers(self, year=None, accept_only=True):
        """
        Match conference papers with arXiv versions

        Args:
            year: Conference year to process (int, list of ints, or None for all years)
            accept_only: Only process accepted papers
        """
        # Load conference papers
        conference_df = self.load_conference_papers(year=year, accept_only=accept_only)

        print(f"\nSearching for {len(conference_df)} papers on arXiv...")
        print(
            f"This will take approximately {len(conference_df) * self.request_delay / 60:.1f} minutes"
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
                    "arXiv ID",
                    "arXiv URL",
                    "PDF URL",
                    "DOI",
                    "Publication Date",
                    "Updated Date",
                    "Title",
                    "Authors",
                    "Author Affiliations",
                    "Abstract",
                    "Categories",
                    "Primary Category",
                    "Comment",
                    "Journal Reference",
                ]
            )

            # Process each paper
            for idx, row in conference_df.iterrows():
                self.papers_checked += 1

                # Progress indicator
                if self.papers_checked % 100 == 0:
                    print(
                        f"Processed {self.papers_checked}/{len(conference_df)} papers "
                        f"(Found: {self.papers_found}, Not found: {self.papers_not_found})"
                    )

                title = row["title"]

                # Search arXiv
                arxiv_paper = self.search_arxiv_by_title(title)

                if arxiv_paper:
                    # Write to CSV
                    writer.writerow(
                        [
                            arxiv_paper["arxiv_id"],
                            arxiv_paper["arxiv_url"],
                            arxiv_paper["pdf_url"],
                            arxiv_paper["doi"],
                            arxiv_paper["publication_date"],
                            arxiv_paper["updated_date"],
                            arxiv_paper["title"],
                            arxiv_paper["authors"],
                            arxiv_paper["affiliations"],
                            arxiv_paper["abstract"],
                            arxiv_paper["categories"],
                            arxiv_paper["primary_category"],
                            arxiv_paper["comment"],
                            arxiv_paper["journal_ref"],
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
        description="Find conference papers on arXiv and output in arXiv scraper format"
    )
    parser.add_argument(
        "--conference",
        type=str,
        required=True,
        choices=["iclr", "neurips", "icml"],
        help="Conference name (iclr, neurips, or icml)",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to conference papers CSV (auto-detected if not provided)",
    )
    parser.add_argument(
        "--year",
        type=int,
        nargs="*",
        default=[2024],
        help="Conference year(s) to process (space-separated). Omit to process all years.",
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

    # Auto-detect input CSV if not provided
    if args.input_csv is None:
        csv_map = {
            "iclr": "results/iclr26v1_papers.csv",
            "neurips": "results/neurips_papers_2024_2025_20251109_081310_fixed.csv",
            "icml": "results/icml_papers_2024_2025_20251109_153723.csv",
        }
        input_csv = csv_map.get(args.conference)
        if input_csv is None or not os.path.exists(input_csv):
            print(
                f"Error: Could not auto-detect CSV for {args.conference}. Please specify --input-csv"
            )
            return
    else:
        input_csv = args.input_csv

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
    matcher = ConferenceArxivMatcher(
        conference_csv_path=input_csv,
        conference_name=args.conference,
        output_csv_path=args.output,
        year=year_for_filename,
    )

    # Match papers
    accept_only = not args.all_papers
    matcher.match_papers(year=year_filter, accept_only=accept_only)


if __name__ == "__main__":
    main()
