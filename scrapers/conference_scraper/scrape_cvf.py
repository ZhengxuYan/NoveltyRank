import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from datetime import datetime
import re
from urllib.parse import urljoin


class CVFScraper:
    def __init__(self, conference="CVPR", start_year=2024, end_year=2024):
        self.conference = conference.upper()
        self.start_year = start_year
        self.end_year = end_year
        self.papers = []

        self.base_url = "https://openaccess.thecvf.com"

        self.request_delay = 1

        if self.conference not in ["CVPR", "ICCV", "ECCV"]:
            raise ValueError(
                f"Conference must be CVPR, ICCV, or ECCV, got {self.conference}"
            )

    def make_request(self, url, retries=3):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                time.sleep(self.request_delay)
                return response
            except requests.exceptions.RequestException as e:
                print(f"\nError fetching {url} (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    return None

        return None

    def get_conference_url(self, year):
        return f"{self.base_url}/{self.conference}{year}?day=all"

    def extract_paper_links(self, soup, year):
        paper_links = []

        links = soup.find_all("a", href=True)

        for link in links:
            href = link["href"]

            if (
                f"/content/{self.conference}{year}/html/" in href
                and "_paper.html" in href
            ):
                full_url = urljoin(self.base_url, href)
                if full_url not in paper_links:
                    paper_links.append(full_url)

        return paper_links

    def extract_paper_data(self, paper_url, year):
        try:
            response = self.make_request(paper_url)
            if response is None:
                return None

            soup = BeautifulSoup(response.content, "html.parser")

            title = ""
            title_elem = soup.find("div", id="papertitle")
            if title_elem:
                title = title_elem.get_text(strip=True)

            authors = ""
            authors_div = soup.find("div", id="authors")
            if authors_div:
                i_tag = authors_div.find("i")
                if i_tag:
                    authors = i_tag.get_text(strip=True)

            abstract = ""
            abstract_elem = soup.find("div", id="abstract")
            if abstract_elem:
                abstract = abstract_elem.get_text(strip=True)

            pdf_link = ""
            pdf_elem = soup.find("a", href=re.compile(r"/content/.*/papers/.*\.pdf$"))
            if pdf_elem:
                pdf_link = urljoin(self.base_url, pdf_elem["href"])

            paper_id = paper_url.split("/")[-1].replace("_paper.html", "")

            paper = {
                "year": year,
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "pdf_url": pdf_link,
                "cvf_url": paper_url,
            }

            return paper

        except Exception as e:
            print(f"\nError extracting paper from {paper_url}: {e}")
            return None

    def scrape_year(self, year):
        print(f"\nScraping {self.conference} {year}:")

        conference_url = self.get_conference_url(year)
        print(f"Conference URL: {conference_url}")

        response = self.make_request(conference_url)
        if response is None:
            print(f"Failed to fetch conference page for {year}")
            return []

        if response.status_code == 404:
            print(f"Conference page not found for {year}")
            return []

        soup = BeautifulSoup(response.content, "html.parser")

        paper_links = self.extract_paper_links(soup, year)
        print(f"Found {len(paper_links)} paper links")

        if len(paper_links) == 0:
            print(
                f"No papers found. The page structure may have changed or {year} is not available yet."
            )
            return []

        year_papers = []
        for i, paper_url in enumerate(paper_links):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(paper_links)} papers...")

            paper = self.extract_paper_data(paper_url, year)
            if paper:
                year_papers.append(paper)

        print(f"Successfully extracted {len(year_papers)} papers")
        return year_papers

    def scrape_all(self):
        print(
            f"Scraping {self.conference} papers from {self.start_year} to {self.end_year}"
        )
        print("=" * 60)

        all_papers = []

        for year in range(self.start_year, self.end_year + 1):
            year_papers = self.scrape_year(year)
            all_papers.extend(year_papers)

        self.papers = all_papers
        print("\n" + "=" * 60)
        print(f"Total papers scraped: {len(all_papers)}")

        return all_papers

    def filter_papers(self, min_abstract_length=100):
        """Filter out papers with very short abstracts"""
        print(
            f"\nFiltering papers with abstract length < {min_abstract_length} characters..."
        )

        filtered_papers = []
        removed_count = 0

        for paper in self.papers:
            if len(paper["abstract"]) >= min_abstract_length:
                filtered_papers.append(paper)
            else:
                removed_count += 1

        print(f"Removed {removed_count} papers with short abstracts")
        self.papers = filtered_papers

        return filtered_papers

    def to_dataframe(self):
        if not self.papers:
            print("No papers to convert!")
            return None

        df = pd.DataFrame(self.papers)

        df = df.sort_values(by=["year", "title"]).reset_index(drop=True)

        columns = ["year", "id", "title", "abstract", "authors", "pdf_url", "cvf_url"]
        df = df[columns]

        return df

    def save_to_csv(self, output_file="papers.csv"):
        df = self.to_dataframe()
        if df is None:
            return None

        text_columns = ["title", "abstract", "authors"]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: x.replace("\n", " ").replace("\r", " ")
                    if isinstance(x, str)
                    else x
                )

        df.to_csv(
            output_file, index=False, encoding="utf-8", escapechar="\\", quoting=1
        )
        print(f"\nSaved {len(df)} papers to {output_file}")

        return df

    def save_to_parquet(self, output_file="papers.parquet"):
        df = self.to_dataframe()
        if df is None:
            return None

        df.to_parquet(output_file)
        print(f"\nSaved {len(df)} papers to {output_file}")

        return df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape papers from CVF Open Access (CVPR, ICCV, ECCV)"
    )
    parser.add_argument(
        "--conference",
        type=str,
        default="CVPR",
        help="Conference name (CVPR, ICCV, ECCV)",
    )
    parser.add_argument("--start-year", type=int, default=2024, help="Start year")
    parser.add_argument("--end-year", type=int, default=2024, help="End year")
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "parquet"],
        help="Output format",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (auto-generated if not provided)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Initialize scraper
    print(
        f"Initializing CVF scraper for {args.conference} ({args.start_year}-{args.end_year})"
    )

    try:
        scraper = CVFScraper(
            conference=args.conference,
            start_year=args.start_year,
            end_year=args.end_year,
        )

        # Scrape all papers
        scraper.scrape_all()

        # Filter out placeholder abstracts
        scraper.filter_papers(min_abstract_length=100)

        # Generate output filename
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            conf_lower = args.conference.lower()
            ext = "csv" if args.format == "csv" else "parquet"
            output_file = f"results/{conf_lower}_papers_{args.start_year}_{args.end_year}_{timestamp}.{ext}"
        else:
            output_file = args.output

        # Save to file
        if args.format == "csv":
            df = scraper.save_to_csv(output_file)
        else:
            df = scraper.save_to_parquet(output_file)

        # Print summary statistics
        if df is not None:
            print("\n" + "=" * 60)
            print("Summary Statistics:")
            print("=" * 60)
            print(df.groupby("year")["id"].count().rename("papers_per_year"))
            print("\n")
        else:
            print("\n" + "=" * 60)
            print("ERROR: No papers found!")
            print("=" * 60)
            print(
                f"The {args.conference} {args.start_year}-{args.end_year} may not be available yet."
            )
            print(
                "Check https://openaccess.thecvf.com/ for available conferences and years."
            )
            print("=" * 60)
            print("\n")

    except ValueError as e:
        print(f"\nError: {e}")
        print("Supported conferences: CVPR, ICCV, ECCV")


if __name__ == "__main__":
    main()
