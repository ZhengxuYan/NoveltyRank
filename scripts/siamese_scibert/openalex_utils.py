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

    def search_work(self, title):
        self._wait_for_rate_limit()
        url = f"{self.base_url}/works"
        params = {
            "search": title,
            "per_page": 1
        }
        try:
            # Small random sleep to desynchronize threads
            time.sleep(random.uniform(0.1, 0.5))
            response = self.session.get(url, params=params)
            response.raise_for_status()
            results = response.json().get("results", [])
            return results[0] if results else None
        except Exception as e:
            print(f"Error searching work '{title}': {e}")
            return None

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

    # Validate name similarity to avoid bad matches (e.g. "Peter Du" -> "Peter Fonagy")
    def validate_author_name(query, result):
        def normalize(s):
            return "".join(c.lower() for c in s if c.isalnum() or c.isspace()).split()
        
        q_parts = normalize(query)
        r_parts = normalize(result)
        
        if not q_parts or not r_parts:
            return False
            
        # Check if query parts are present in result
        # Allow initials (e.g. "j" matches "john")
        match_count = 0
        for q in q_parts:
            for r in r_parts:
                if r.startswith(q):
                    match_count += 1
                    break
        
        # Require at least 50% of query parts to match
        # And specifically for short names (<= 2 parts), require all parts to match
        if len(q_parts) <= 2:
            return match_count == len(q_parts)
        return match_count >= len(q_parts) * 0.8

    author = results[0]
    if not validate_author_name(author_name, author.get("display_name", "")):
        if debug:
            print(f"  -> Name mismatch: '{author_name}' vs '{author.get('display_name')}'", flush=True)
        # Try other results if available?
        # For now, just return None to be safe
        return None
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

    # Process affiliations
    affiliations_str = ""
    
    # Priority 1: Use last_known_institutions from OpenAlex (curated current affiliations)
    last_known = detailed.get("last_known_institutions", [])
    if last_known:
        names = [inst.get("display_name") for inst in last_known if inst.get("display_name")]
        if names:
            affiliations_str = "; ".join(names)

    # Priority 2: Fallback to calculating from affiliations list
    if not affiliations_str:
        affiliations_list = []
        for aff in detailed.get("affiliations", []):
            institution = aff.get("institution")
            if institution and institution.get("display_name"):
                years = aff.get("years", [])
                if years:
                    max_year = max(years)
                    affiliations_list.append({
                        "name": institution.get("display_name"),
                        "max_year": max_year
                    })
        
        if affiliations_list:
            # Sort by max_year descending
            affiliations_list.sort(key=lambda x: x["max_year"], reverse=True)
            
            # Get affiliations from the most recent active years (within 1 year of the latest)
            latest_year = affiliations_list[0]["max_year"]
            recent_affs = []
            seen_names = set()
            
            for aff in affiliations_list:
                if aff["max_year"] >= latest_year - 1:
                    name = aff["name"]
                    if name not in seen_names:
                        recent_affs.append(name)
                        seen_names.add(name)
                else:
                    break
            
            # Take up to 3 most recent unique affiliations
            affiliations_str = "; ".join(recent_affs[:3])

    comprehensive_info = {
        "openalex_id": detailed.get("id"),
        "name": detailed.get("display_name"),
        "orcid": detailed.get("orcid"),
        "affiliations": affiliations_str,
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
    title_col = None
    for col in df.columns:
        if col.lower() == "authors":
            author_col = col
        if col.lower() == "title":
            title_col = col
            
    if author_col is None:
        print("Warning: No 'authors' column found. Skipping enrichment.")
        return df

    # Load cache
    author_info = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                author_info = pickle.load(f)
            print(f"Loaded {len(author_info)} cached entries.")
        except Exception as e:
            print(f"Error loading cache: {e}")

    fetcher = OpenAlexFetcher(email=email)
    
    # Step 1: Identify papers and their authors
    # We will try to find the OpenAlex Work for each paper to get precise Author IDs
    print("Resolving authors via OpenAlex Works...")
    
    paper_authors_map = {} # index -> [{"id": "...", "name": "..."}]
    
    # Load works cache
    works_cache_file = "openalex_works_cache.pkl"
    works_cache = {}
    if os.path.exists(works_cache_file):
        try:
            with open(works_cache_file, "rb") as f:
                works_cache = pickle.load(f)
            print(f"Loaded {len(works_cache)} cached works.")
        except Exception as e:
            print(f"Error loading works cache: {e}")

    # We need to fetch works in parallel
    max_workers = 5
    lock = threading.Lock()
    
    # Identify which papers need work lookup
    indices_to_process = list(df.index)
    
    pbar_works = tqdm(total=len(indices_to_process), desc="Resolving Works")
    
    import difflib

    def process_paper_work(idx):
        try:
            row = df.loc[idx]
            title = row[title_col] if title_col else None
            if not title:
                pbar_works.update(1)
                return
            
            # Check cache first
            if title in works_cache:
                with lock:
                    paper_authors_map[idx] = works_cache[title]
                pbar_works.update(1)
                return

            # Search for work
            work = fetcher.search_work(title)
            
            resolved_authors = []
            if work:
                # Validate title match
                work_title = work.get("display_name", "")
                if work_title:
                    # Normalize titles for comparison
                    def normalize(s):
                        return "".join(c.lower() for c in s if c.isalnum())
                    
                    t1 = normalize(title)
                    t2 = normalize(work_title)
                    
                    # Calculate similarity
                    ratio = difflib.SequenceMatcher(None, t1, t2).ratio()
                    
                    # Threshold: 0.8 seems reasonable for minor variations
                    if ratio >= 0.8:
                        authorships = work.get("authorships", [])
                        for auth in authorships:
                            author_obj = auth.get("author", {})
                            aid = author_obj.get("id")
                            aname = author_obj.get("display_name")
                            if aid:
                                resolved_authors.append({"id": aid, "name": aname})
            
            # Update maps and cache
            with lock:
                if resolved_authors:
                    paper_authors_map[idx] = resolved_authors
                
                # Cache the result (even if empty, to avoid re-searching failed titles)
                # But maybe we only cache if we found something? 
                # If we cache empty, we might miss it if it appears later in OpenAlex.
                # Let's cache it.
                works_cache[title] = resolved_authors
                
                pbar_works.update(1)
                if pbar_works.n % 50 == 0:
                    with open(works_cache_file, "wb") as f:
                        pickle.dump(works_cache, f)

        except Exception as e:
            # print(f"Error processing work {idx}: {e}")
            with lock:
                pbar_works.update(1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_paper_work, idx) for idx in indices_to_process]
        for future in as_completed(futures):
            pass
            
    pbar_works.close()
    
    # Final save of works cache
    with open(works_cache_file, "wb") as f:
        pickle.dump(works_cache, f)
    
    # Step 2: Identify missing author info (IDs or Names)
    to_fetch_ids = set()
    to_fetch_names = set()
    
    for idx in df.index:
        # If we have resolved authors (IDs), use them
        if idx in paper_authors_map:
            for auth in paper_authors_map[idx]:
                aid = auth["id"]
                if aid not in author_info:
                    to_fetch_ids.add(aid)
        else:
            # Fallback to names
            author_string = df.loc[idx, author_col]
            names = parse_author_names(author_string)
            for name in names:
                if name not in author_info:
                    to_fetch_names.add(name)
                    
    print(f"Need to fetch {len(to_fetch_ids)} authors by ID and {len(to_fetch_names)} by Name.")
    
    # Step 3: Fetch missing info
    total_fetch = len(to_fetch_ids) + len(to_fetch_names)
    if total_fetch > 0:
        pbar_fetch = tqdm(total=total_fetch, desc="Fetching Authors")
        
        def fetch_and_cache_id(aid):
            try:
                # Check cache again
                if aid in author_info: 
                    with lock: pbar_fetch.update(1)
                    return

                info = fetcher.get_author_by_id(aid)
                if info:
                    # Process info similar to fetch_comprehensive_author_info
                    # But we need to extract the fields we want
                    # We can reuse calculate_author_career_metrics logic
                    
                    works = fetcher.get_author_works(aid, limit=200)
                    career_metrics = calculate_author_career_metrics(works, info.get("counts_by_year", []))
                    
                    # Process affiliations
                    affiliations_str = ""
                    last_known = info.get("last_known_institutions", [])
                    if last_known:
                        names = [inst.get("display_name") for inst in last_known if inst.get("display_name")]
                        if names:
                            affiliations_str = "; ".join(names)

                    if not affiliations_str:
                        # Fallback logic
                        affiliations_list = []
                        for aff in info.get("affiliations", []):
                            institution = aff.get("institution")
                            if institution and institution.get("display_name"):
                                years = aff.get("years", [])
                                if years:
                                    max_year = max(years)
                                    affiliations_list.append({
                                        "name": institution.get("display_name"),
                                        "max_year": max_year
                                    })
                        if affiliations_list:
                            affiliations_list.sort(key=lambda x: x["max_year"], reverse=True)
                            latest_year = affiliations_list[0]["max_year"]
                            recent_affs = []
                            seen_names = set()
                            for aff in affiliations_list:
                                if aff["max_year"] >= latest_year - 1:
                                    name = aff["name"]
                                    if name not in seen_names:
                                        recent_affs.append(name)
                                        seen_names.add(name)
                                else:
                                    break
                            affiliations_str = "; ".join(recent_affs[:3])

                    comprehensive_info = {
                        "openalex_id": info.get("id"),
                        "name": info.get("display_name"),
                        "orcid": info.get("orcid"),
                        "affiliations": affiliations_str,
                        "h_index": info.get("summary_stats", {}).get("h_index", 0),
                        "i10_index": info.get("summary_stats", {}).get("i10_index", 0),
                        "citation_count": info.get("cited_by_count", 0),
                        "paper_count": info.get("works_count", 0),
                        "two_year_citedness": info.get("summary_stats", {}).get("2yr_mean_citedness", 0),
                        "years_active": career_metrics.get("years_active"),
                        "first_pub_year": career_metrics.get("first_pub_year"),
                        "last_pub_year": career_metrics.get("last_pub_year"),
                        "career_stage": career_metrics.get("career_stage"),
                        "recent_productivity": career_metrics.get("recent_productivity"),
                        "avg_citations_per_paper": career_metrics.get("avg_citations_per_paper"),
                        "max_paper_citations": career_metrics.get("max_paper_citations"),
                    }
                    
                    with lock:
                        author_info[aid] = comprehensive_info
                        # Also cache by name if not exists? No, might be ambiguous.
                        # But we can cache by the name returned by OpenAlex to speed up future lookups?
                        # No, let's stick to ID for ID-based lookups.
                
                with lock:
                    pbar_fetch.update(1)
                    if pbar_fetch.n % 50 == 0:
                        with open(cache_file, "wb") as f:
                            pickle.dump(author_info, f)
                            
            except Exception as e:
                print(f"Error fetching ID {aid}: {e}")
                with lock: pbar_fetch.update(1)

        def fetch_and_cache_name(name):
            try:
                if name in author_info:
                    with lock: pbar_fetch.update(1)
                    return
                    
                info = fetch_comprehensive_author_info(name, fetcher)
                with lock:
                    if info:
                        author_info[name] = info
                    else:
                        author_info[name] = {
                            "openalex_id": None,
                            "name": name,
                            "h_index": None
                        }
                    pbar_fetch.update(1)
                    if pbar_fetch.n % 50 == 0:
                        with open(cache_file, "wb") as f:
                            pickle.dump(author_info, f)
            except Exception as e:
                print(f"Error fetching name {name}: {e}")
                with lock: pbar_fetch.update(1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_ids = [executor.submit(fetch_and_cache_id, aid) for aid in to_fetch_ids]
            futures_names = [executor.submit(fetch_and_cache_name, name) for name in to_fetch_names]
            
            for future in as_completed(futures_ids + futures_names):
                pass
                
        pbar_fetch.close()
        
        # Final save
        with open(cache_file, "wb") as f:
            pickle.dump(author_info, f)

    # Step 4: Calculate features
    def calculate_features_row(row):
        idx = row.name # Assuming index is preserved
        
        authors_data = []
        
        # Priority: Use resolved IDs
        if idx in paper_authors_map and paper_authors_map[idx]:
            for auth in paper_authors_map[idx]:
                aid = auth["id"]
                if aid in author_info:
                    authors_data.append(author_info[aid])
                else:
                    # Should not happen if fetch worked
                    pass
        else:
            # Fallback: Use names
            author_string = row[author_col]
            names = parse_author_names(author_string)
            for name in names:
                if name in author_info:
                    authors_data.append(author_info[name])
                else:
                    # Basic info
                    authors_data.append({"name": name})
        
        if not authors_data:
             return {
                "num_authors": 0,
                "max_h_index": None,
                "avg_h_index": None,
                "has_top_author": False
            }

        # Aggregate stats
        h_indices = []
        citations = []
        career_stages = []
        
        # Lists for string fields
        author_affs = []
        author_ids = []
        author_names = []
        author_orcids = []
        author_h_indices = []
        author_i10_indices = []
        author_citation_counts = []
        author_paper_counts = []
        author_two_year_citedness = []
        author_years_active = []
        author_first_pub_years = []
        author_last_pub_years = []
        author_career_stages = []
        author_recent_productivity = []
        author_avg_citations_per_paper = []
        author_max_paper_citations = []
        
        def safe_str(val):
            if val is None: return ""
            return str(val)

        for info in authors_data:
            if info.get("h_index") is not None:
                h_indices.append(info["h_index"])
            if info.get("citation_count") is not None:
                citations.append(info["citation_count"])
            if info.get("career_stage") is not None:
                career_stages.append(info["career_stage"])
                
            author_affs.append(safe_str(info.get("affiliations")))
            author_ids.append(safe_str(info.get("openalex_id")))
            author_names.append(safe_str(info.get("name")))
            author_orcids.append(safe_str(info.get("orcid")))
            author_h_indices.append(safe_str(info.get("h_index")))
            author_i10_indices.append(safe_str(info.get("i10_index")))
            author_citation_counts.append(safe_str(info.get("citation_count")))
            author_paper_counts.append(safe_str(info.get("paper_count")))
            author_two_year_citedness.append(safe_str(info.get("two_year_citedness")))
            author_years_active.append(safe_str(info.get("years_active")))
            author_first_pub_years.append(safe_str(info.get("first_pub_year")))
            author_last_pub_years.append(safe_str(info.get("last_pub_year")))
            author_career_stages.append(safe_str(info.get("career_stage")))
            author_recent_productivity.append(safe_str(info.get("recent_productivity")))
            author_avg_citations_per_paper.append(safe_str(info.get("avg_citations_per_paper")))
            author_max_paper_citations.append(safe_str(info.get("max_paper_citations")))

        return {
            "num_authors": len(authors_data),
            "max_h_index": max(h_indices) if h_indices else None,
            "avg_h_index": np.mean(h_indices) if h_indices else None,
            "min_h_index": min(h_indices) if h_indices else None,
            "first_author_h_index": authors_data[0].get("h_index") if authors_data else None,
            "last_author_h_index": authors_data[-1].get("h_index") if authors_data else None,
            "max_citations": max(citations) if citations else None,
            "avg_citations": np.mean(citations) if citations else None,
            "total_author_citations": sum(citations) if citations else None,
            "max_career_stage": max(career_stages) if career_stages else None,
            "avg_career_stage": np.mean(career_stages) if career_stages else None,
            "has_early_career_author": any(s <= 5 for s in career_stages) if career_stages else None,
            "has_senior_author": any(s >= 20 for s in career_stages) if career_stages else None,
            "author_diversity_score": np.std(h_indices) if len(h_indices) > 1 else 0,
            "has_top_author": any(h >= 50 for h in h_indices) if h_indices else False,
            
            "author_affiliations": "; ".join(author_affs),
            "author_ids": "; ".join(author_ids),
            "author_names": "; ".join(author_names),
            "author_orcids": "; ".join(author_orcids),
            "author_h_indices": "; ".join(author_h_indices),
            "author_i10_indices": "; ".join(author_i10_indices),
            "author_citation_counts": "; ".join(author_citation_counts),
            "author_paper_counts": "; ".join(author_paper_counts),
            "author_two_year_citedness": "; ".join(author_two_year_citedness),
            "author_years_active": "; ".join(author_years_active),
            "author_first_pub_years": "; ".join(author_first_pub_years),
            "author_last_pub_years": "; ".join(author_last_pub_years),
            "author_career_stages": "; ".join(author_career_stages),
            "author_recent_productivity": "; ".join(author_recent_productivity),
            "author_avg_citations_per_paper": "; ".join(author_avg_citations_per_paper),
            "author_max_paper_citations": "; ".join(author_max_paper_citations)
        }

    print("Calculating paper features...")
    # Apply row-wise
    features_list = df.apply(calculate_features_row, axis=1)
    features_df = pd.DataFrame(features_list.tolist())
    
    # Merge
    df = df.reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    
    df_enriched = pd.concat([df, features_df], axis=1)
    return df_enriched
