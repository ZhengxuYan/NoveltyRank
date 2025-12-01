import requests
import time
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class OpenAlexFetcher:
    def __init__(self, email="your-email@example.com", request_delay=0.35):
        self.base_url = "https://api.openalex.org"
        self.email = email
        self.request_delay = request_delay
        self.headers = {
            "User-Agent": f"mailto:{email}"
        }
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(self.headers)

    def _wait_for_rate_limit(self):
        # No-op for parallel execution
        pass

    def search_author_by_name(self, name, limit=1):
        self._wait_for_rate_limit()
        url = f"{self.base_url}/authors"
        params = {
            "search": name,
            "per_page": limit
        }
        try:
            # Small random sleep to desynchronize threads
            time.sleep(random.uniform(0.1, 0.5))
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            print(f"Error searching author '{name}': {e}")
            return []

    def get_author_by_id(self, author_id):
        self._wait_for_rate_limit()
        # author_id might be a full URL or just the ID
        if "openalex.org/" in author_id:
            author_id = author_id.split("/")[-1]
            
        url = f"{self.base_url}/authors/{author_id}"
        try:
            # Small random sleep to desynchronize threads
            time.sleep(random.uniform(0.1, 0.5))
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting author '{author_id}': {e}")
            return None

    def get_author_works(self, author_id, limit=200):
        self._wait_for_rate_limit()
        if "openalex.org/" in author_id:
            author_id = author_id.split("/")[-1]
            
        url = f"{self.base_url}/works"
        params = {
            "filter": f"author.id:{author_id}",
            "per_page": limit,
            "sort": "publication_date:desc"
        }
        try:
            # Small random sleep to desynchronize threads
            time.sleep(random.uniform(0.1, 0.5))
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            print(f"Error getting works for author '{author_id}': {e}")
            return []

def parse_author_names(author_string):
    if pd.isna(author_string) or author_string == "":
        return []
    if isinstance(author_string, list):
        return author_string

    if ";" in author_string:
        separator = ";"
    else:
        separator = ","

    return [name.strip() for name in author_string.split(separator) if name.strip()]

def calculate_author_career_metrics(works, counts_by_year):
    if not counts_by_year:
        return {}

    years_with_pubs = [
        item["year"] for item in counts_by_year if item.get("works_count", 0) > 0
    ]

    if not years_with_pubs:
        return {}

    current_year = datetime.now().year
    first_year = min(years_with_pubs)
    last_year = max(years_with_pubs)
    years_active = last_year - first_year + 1

    recent_paper_count = sum(
        item.get("works_count", 0)
        for item in counts_by_year
        if item.get("year", 0) >= current_year - 3
    )

    if works:
        citations = [w.get("cited_by_count", 0) for w in works]
        total_citations = sum(citations)
        avg_citations_per_paper = total_citations / len(works) if works else 0
        max_paper_citations = max(citations) if citations else 0
    else:
        total_citations = 0
        avg_citations_per_paper = 0
        max_paper_citations = 0

    return {
        "years_active": years_active,
        "first_pub_year": first_year,
        "last_pub_year": last_year,
        "career_stage": current_year - first_year,
        "recent_productivity": recent_paper_count,
        "avg_citations_per_paper": avg_citations_per_paper,
        "max_paper_citations": max_paper_citations,
        "total_citations": total_citations,
        "total_papers": len(works),
    }

def fetch_comprehensive_author_info(author_name, fetcher, debug=False):
    if debug:
        print(f"  -> Searching for author...", flush=True)
    results = fetcher.search_author_by_name(author_name, limit=1)

    if not results:
        if debug:
            print(f"  -> Not found", flush=True)
        return None

    author = results[0]
    author_id = author.get("id") # OpenAlex returns 'id' in search results

    if debug:
        print(f"  -> Getting detailed info for {author_id}...", flush=True)
    detailed = fetcher.get_author_by_id(author_id)

    if not detailed:
        return author

    if debug:
        print(f"  -> Getting works...", flush=True)
    works = fetcher.get_author_works(author_id, limit=200)

    career_metrics = calculate_author_career_metrics(
        works, detailed.get("counts_by_year", [])
    )

    comprehensive_info = {
        "openalex_id": detailed.get("id"),
        "name": detailed.get("display_name"),
        "orcid": detailed.get("orcid"),
        "affiliations": ", ".join([a.get("institution", {}).get("display_name", "") for a in detailed.get("affiliations", []) if a.get("institution")]),
        "h_index": detailed.get("summary_stats", {}).get("h_index", 0),
        "i10_index": detailed.get("summary_stats", {}).get("i10_index", 0),
        "citation_count": detailed.get("cited_by_count", 0),
        "paper_count": detailed.get("works_count", 0),
        "two_year_citedness": detailed.get("summary_stats", {}).get("2yr_mean_citedness", 0),
        "years_active": career_metrics.get("years_active"),
        "first_pub_year": career_metrics.get("first_pub_year"),
        "last_pub_year": career_metrics.get("last_pub_year"),
        "career_stage": career_metrics.get("career_stage"),
        "recent_productivity": career_metrics.get("recent_productivity"),
        "avg_citations_per_paper": career_metrics.get("avg_citations_per_paper"),
        "max_paper_citations": career_metrics.get("max_paper_citations"),
    }

    return comprehensive_info

def enrich_papers_with_openalex(df, cache_file="openalex_cache.pkl", email="your-email@example.com"):
    print(f"Enriching {len(df)} papers with OpenAlex author data...")
    
    # Ensure authors column exists
    author_col = None
    for col in df.columns:
        if col.lower() == "authors":
            author_col = col
            break
            
    if author_col is None:
        print("Warning: No 'authors' column found. Skipping enrichment.")
        return df

    # Load cache
    author_info = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                author_info = pickle.load(f)
            print(f"Loaded {len(author_info)} cached authors.")
        except Exception as e:
            print(f"Error loading cache: {e}")

    # Identify unique authors
    all_authors = set()
    for author_string in df[author_col]:
        authors = parse_author_names(author_string)
        all_authors.update(authors)
        
    remaining_authors = [a for a in all_authors if a not in author_info]
    print(f"Found {len(all_authors)} unique authors, {len(remaining_authors)} to fetch.")
    
    if remaining_authors:
        fetcher = OpenAlexFetcher(email=email)
        
        # Use ThreadPoolExecutor for parallel fetching
        # OpenAlex polite pool allows ~10 req/s. 
        # Reduced to 5 workers and added retries to be safe.
        max_workers = 5
        
        print(f"Fetching {len(remaining_authors)} authors with {max_workers} threads...")
        
        lock = threading.Lock()
        pbar = tqdm(total=len(remaining_authors), desc="Fetching Authors")
        
        def fetch_and_cache(author_name):
            try:
                info = fetch_comprehensive_author_info(author_name, fetcher)
                with lock:
                    if info:
                        author_info[author_name] = info
                    else:
                        author_info[author_name] = {
                            "openalex_id": None,
                            "name": author_name,
                            "h_index": None
                        }
                    pbar.update(1)
                    
                    # Periodic save (thread-safeish)
                    if pbar.n % 100 == 0:
                        with open(cache_file, "wb") as f:
                            pickle.dump(author_info, f)
                            
            except Exception as e:
                print(f"Error fetching {author_name}: {e}")
                with lock:
                    pbar.update(1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_and_cache, name) for name in remaining_authors]
            
            # Wait for all to complete
            for future in as_completed(futures):
                pass
                
        pbar.close()
        
        # Final save
        with open(cache_file, "wb") as f:
            pickle.dump(author_info, f)

    # Calculate features per paper
    def calculate_features(author_string):
        authors = parse_author_names(author_string)
        if not authors:
            return {
                "num_authors": 0,
                "max_h_index": None,
                "avg_h_index": None,
                "has_top_author": False
            }
            
        h_indices = []
        citations = []
        career_stages = []
        
        for author in authors:
            if author in author_info:
                info = author_info[author]
                if info.get("h_index") is not None:
                    h_indices.append(info["h_index"])
                if info.get("citation_count") is not None:
                    citations.append(info["citation_count"])
                if info.get("career_stage") is not None:
                    career_stages.append(info["career_stage"])
                    
        return {
            "num_authors": len(authors),
            "max_h_index": max(h_indices) if h_indices else None,
            "avg_h_index": np.mean(h_indices) if h_indices else None,
            "min_h_index": min(h_indices) if h_indices else None,
            "first_author_h_index": author_info.get(authors[0], {}).get("h_index") if authors else None,
            "last_author_h_index": author_info.get(authors[-1], {}).get("h_index") if authors else None,
            "max_citations": max(citations) if citations else None,
            "avg_citations": np.mean(citations) if citations else None,
            "total_author_citations": sum(citations) if citations else None,
            "max_career_stage": max(career_stages) if career_stages else None,
            "avg_career_stage": np.mean(career_stages) if career_stages else None,
            "has_early_career_author": any(s <= 5 for s in career_stages) if career_stages else None,
            "has_senior_author": any(s >= 20 for s in career_stages) if career_stages else None,
            "author_diversity_score": np.std(h_indices) if len(h_indices) > 1 else 0,
            "has_top_author": any(h >= 50 for h in h_indices) if h_indices else False,
        }

    print("Calculating paper features...")
    features_list = df[author_col].apply(calculate_features)
    features_df = pd.DataFrame(features_list.tolist())
    
    # Merge
    # Reset indices to ensure alignment
    df = df.reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    
    df_enriched = pd.concat([df, features_df], axis=1)
    return df_enriched
