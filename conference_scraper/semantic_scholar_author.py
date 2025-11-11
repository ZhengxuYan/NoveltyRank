"""
Semantic Scholar Author Information Fetcher

Retrieves author metadata including h-index, citation count, affiliations, etc.
Using the Semantic Scholar Academic Graph API.
"""

import requests
import time
import pandas as pd
from typing import Dict, List, Optional


class SemanticScholarAuthor:
    """Fetch author information from Semantic Scholar API"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Semantic Scholar author fetcher

        Args:
            api_key (str, optional): API key for higher rate limits
                Without key: 100 requests/5 minutes
                With key: 5,000 requests/5 minutes
        """
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key

        # Rate limiting
        self.request_delay = 0.1  # seconds between requests
        self.rate_limit_delay = 60  # seconds to wait when rate limited

    def search_author_by_name(self, author_name: str, limit: int = 10) -> List[Dict]:
        """
        Search for authors by name

        Args:
            author_name (str): Author name to search for
            limit (int): Maximum number of results to return

        Returns:
            List[Dict]: List of author results with basic info
        """
        url = f"{self.base_url}/author/search"
        params = {
            "query": author_name,
            "limit": limit,
            "fields": "authorId,name,affiliations,paperCount,citationCount,hIndex",
        }

        try:
            response = requests.get(url, params=params, headers=self.headers)

            if response.status_code == 429:  # Rate limited
                print(f"Rate limited! Waiting {self.rate_limit_delay} seconds...")
                time.sleep(self.rate_limit_delay)
                response = requests.get(url, params=params, headers=self.headers)

            response.raise_for_status()
            time.sleep(self.request_delay)

            data = response.json()
            return data.get("data", [])

        except Exception as e:
            print(f"Error searching for author '{author_name}': {e}")
            return []

    def get_author_by_id(
        self, author_id: str, include_papers: bool = False
    ) -> Optional[Dict]:
        """
        Get detailed author information by Semantic Scholar author ID

        Args:
            author_id (str): Semantic Scholar author ID
            include_papers (bool): Whether to include paper list

        Returns:
            Dict: Detailed author information
        """
        url = f"{self.base_url}/author/{author_id}"

        fields = [
            "authorId",
            "name",
            "aliases",
            "affiliations",
            "homepage",
            "paperCount",
            "citationCount",
            "hIndex",
        ]

        if include_papers:
            fields.extend(
                [
                    "papers",
                    "papers.title",
                    "papers.year",
                    "papers.citationCount",
                    "papers.venue",
                    "papers.publicationTypes",
                ]
            )

        params = {"fields": ",".join(fields)}

        try:
            response = requests.get(url, params=params, headers=self.headers)

            if response.status_code == 429:
                print(f"Rate limited! Waiting {self.rate_limit_delay} seconds...")
                time.sleep(self.rate_limit_delay)
                response = requests.get(url, params=params, headers=self.headers)

            response.raise_for_status()
            time.sleep(self.request_delay)

            return response.json()

        except Exception as e:
            print(f"Error fetching author ID '{author_id}': {e}")
            return None

    def get_author_papers(self, author_id: str, limit: int = 100) -> List[Dict]:
        """
        Get all papers by an author

        Args:
            author_id (str): Semantic Scholar author ID
            limit (int): Maximum number of papers to return

        Returns:
            List[Dict]: List of papers
        """
        url = f"{self.base_url}/author/{author_id}/papers"
        params = {
            "fields": "paperId,title,year,citationCount,venue,publicationTypes,abstract",
            "limit": limit,
        }

        try:
            response = requests.get(url, params=params, headers=self.headers)

            if response.status_code == 429:
                print(f"Rate limited! Waiting {self.rate_limit_delay} seconds...")
                time.sleep(self.rate_limit_delay)
                response = requests.get(url, params=params, headers=self.headers)

            response.raise_for_status()
            time.sleep(self.request_delay)

            data = response.json()
            return data.get("data", [])

        except Exception as e:
            print(f"Error fetching papers for author '{author_id}': {e}")
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
            if author.get("affiliations"):
                for aff in author["affiliations"]:
                    if affiliation_lower in aff.lower():
                        return author

        # No affiliation match, return first result
        return results[0]
