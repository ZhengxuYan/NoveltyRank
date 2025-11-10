"""
OpenAlex Author Information Fetcher

Retrieves author metadata including h-index, citation count, affiliations, etc.
Using the OpenAlex API - much faster rate limits than Semantic Scholar!

Rate Limits:
- Standard: 100,000 requests per day
- Polite Pool (with email): 10x faster, no throttling
"""

import requests
import time
from typing import Dict, List, Optional


class OpenAlexAuthor:
    """Fetch author information from OpenAlex API"""

    def __init__(self, email: str = "your-email@example.com"):
        """
        Initialize the OpenAlex author fetcher

        Args:
            email (str): Your email for the "polite pool" (10x faster, recommended)
        """
        self.base_url = "https://api.openalex.org"
        self.email = email
        self.headers = {"User-Agent": f"mailto:{email}"}

        # Rate limiting: OpenAlex allows max 10 requests/second
        # We make 3 requests per author, so need 0.35s delay to stay under 10 req/sec
        self.request_delay = 0.35  # ~3 requests/second = safe

    def search_author_by_name(self, author_name: str, limit: int = 10) -> List[Dict]:
        """
        Search for authors by name

        Args:
            author_name (str): Author name to search for
            limit (int): Maximum number of results to return

        Returns:
            List[Dict]: List of author results with basic info
        """
        url = f"{self.base_url}/authors"
        params = {"search": author_name, "per_page": limit}

        try:
            response = requests.get(
                url, params=params, headers=self.headers, timeout=10
            )
            response.raise_for_status()
            time.sleep(self.request_delay)

            data = response.json()
            results = data.get("results", [])

            # Convert to simpler format
            authors = []
            for author in results:
                authors.append(
                    {
                        "id": author.get("id"),
                        "openalex_id": author.get("id", "").replace(
                            "https://openalex.org/", ""
                        ),
                        "name": author.get("display_name"),
                        "orcid": author.get("orcid"),
                        "works_count": author.get("works_count", 0),
                        "cited_by_count": author.get("cited_by_count", 0),
                        "h_index": author.get("summary_stats", {}).get("h_index", 0),
                        "i10_index": author.get("summary_stats", {}).get(
                            "i10_index", 0
                        ),
                        "affiliations": self._extract_affiliations(author),
                    }
                )

            return authors

        except Exception as e:
            print(f"Error searching for author '{author_name}': {e}")
            return []

    def get_author_by_id(self, author_id: str) -> Optional[Dict]:
        """
        Get detailed author information by OpenAlex author ID

        Args:
            author_id (str): OpenAlex author ID (e.g., "A2208157607" or full URL)

        Returns:
            Dict: Detailed author information
        """
        # Clean the ID
        if author_id.startswith("https://"):
            author_id = author_id.replace("https://openalex.org/", "")

        url = f"{self.base_url}/authors/{author_id}"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            time.sleep(self.request_delay)

            author = response.json()

            return {
                "id": author.get("id"),
                "openalex_id": author.get("id", "").replace(
                    "https://openalex.org/", ""
                ),
                "name": author.get("display_name"),
                "orcid": author.get("orcid"),
                "works_count": author.get("works_count", 0),
                "cited_by_count": author.get("cited_by_count", 0),
                "h_index": author.get("summary_stats", {}).get("h_index", 0),
                "i10_index": author.get("summary_stats", {}).get("i10_index", 0),
                "two_year_mean_citedness": author.get("summary_stats", {}).get(
                    "2yr_mean_citedness", 0
                ),
                "affiliations": self._extract_affiliations(author),
                "counts_by_year": author.get("counts_by_year", []),
                "works_api_url": author.get("works_api_url"),
            }

        except Exception as e:
            print(f"Error fetching author ID '{author_id}': {e}")
            return None

    def get_author_works(self, author_id: str, limit: int = 100) -> List[Dict]:
        """
        Get works (papers) by an author

        Args:
            author_id (str): OpenAlex author ID
            limit (int): Maximum number of works to return

        Returns:
            List[Dict]: List of works
        """
        # Clean the ID
        if author_id.startswith("https://"):
            author_id = author_id.replace("https://openalex.org/", "")

        url = f"{self.base_url}/works"
        params = {
            "filter": f"author.id:{author_id}",
            "per_page": min(limit, 200),  # Max 200 per page
            "sort": "publication_year:desc",
        }

        try:
            response = requests.get(
                url, params=params, headers=self.headers, timeout=10
            )
            response.raise_for_status()
            time.sleep(self.request_delay)

            data = response.json()
            works = []

            for work in data.get("results", []):
                works.append(
                    {
                        "id": work.get("id"),
                        "title": work.get("title"),
                        "publication_year": work.get("publication_year"),
                        "cited_by_count": work.get("cited_by_count", 0),
                        "type": work.get("type"),
                    }
                )

            return works

        except Exception as e:
            print(f"Error fetching works for author '{author_id}': {e}")
            return []

    def find_best_match(
        self, author_name: str, affiliation_hint: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Find the best matching author for a given name

        Args:
            author_name (str): Author name to search for
            affiliation_hint (str, optional): Expected affiliation to help disambiguate

        Returns:
            Dict: Best matching author info
        """
        results = self.search_author_by_name(author_name, limit=10)

        if not results:
            return None

        # If no affiliation hint, return the first result (usually best match)
        if not affiliation_hint:
            return results[0]

        # Try to match by affiliation
        affiliation_lower = affiliation_hint.lower()
        for author in results:
            affiliations = author.get("affiliations", [])
            for aff in affiliations:
                if affiliation_lower in aff.lower():
                    return author

        # No affiliation match, return first result
        return results[0]

    def _extract_affiliations(self, author: Dict) -> List[str]:
        """Extract affiliation names from author data"""
        affiliations = []

        # Current affiliation
        last_known = author.get("last_known_institution")
        if last_known and last_known.get("display_name"):
            affiliations.append(last_known["display_name"])

        # Alternative: from affiliations list
        if not affiliations:
            for aff in author.get("affiliations", [])[:3]:  # Top 3
                inst = aff.get("institution")
                if inst and inst.get("display_name"):
                    affiliations.append(inst["display_name"])

        return affiliations


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch author info from OpenAlex")
    parser.add_argument("author_name", type=str, help="Author name to search for")
    parser.add_argument(
        "--email",
        type=str,
        default="research@example.com",
        help="Your email for polite pool (faster access)",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of search results"
    )
    parser.add_argument("--works", action="store_true", help="Include works list")

    args = parser.parse_args()

    # Initialize fetcher
    fetcher = OpenAlexAuthor(email=args.email)

    # Search for author
    print(f"\nSearching OpenAlex for '{args.author_name}'...\n")
    results = fetcher.search_author_by_name(args.author_name, limit=args.limit)

    if not results:
        print("No authors found!")
        return

    print(f"Found {len(results)} author(s):\n")

    # Display all results
    for i, author in enumerate(results, 1):
        print(f"{i}. {author['name']}")
        print(f"   OpenAlex ID: {author['openalex_id']}")
        print(f"   h-index: {author['h_index']}")
        print(f"   Citations: {author['cited_by_count']:,}")
        print(f"   Works: {author['works_count']}")
        if author.get("affiliations"):
            print(f"   Affiliation: {', '.join(author['affiliations'])}")
        print()

    # Get works for first result
    if results and args.works:
        print("=" * 60)
        print("WORKS FOR TOP MATCH")
        print("=" * 60)

        author_id = results[0]["openalex_id"]
        works = fetcher.get_author_works(author_id, limit=20)

        for i, work in enumerate(works, 1):
            print(f"\n{i}. {work.get('title', 'N/A')}")
            print(f"   Year: {work.get('publication_year', 'N/A')}")
            print(f"   Citations: {work.get('cited_by_count', 0)}")


if __name__ == "__main__":
    main()
