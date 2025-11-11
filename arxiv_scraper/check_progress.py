"""
Check progress of OpenAlex enrichment
"""

import pickle
import os

cache_file = "arxiv_scraper/results/arxiv_results_v2_filtered_openalex_cache.pkl"

if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        author_info = pickle.load(f)

    total_authors = 111005  # From our earlier count
    cached_authors = len(author_info)
    progress = (cached_authors / total_authors) * 100

    print(f"\n📊 Progress Report")
    print(f"=" * 50)
    print(f"Authors processed: {cached_authors:,} / {total_authors:,}")
    print(f"Progress: {progress:.1f}%")
    print(
        f"Authors with data: {sum(1 for v in author_info.values() if v.get('openalex_id'))}"
    )

    # Estimate time remaining
    if cached_authors > 0:
        rate = 3  # authors per second (3 API calls each, delay=0.35)
        remaining = total_authors - cached_authors
        time_remaining = remaining / rate / 60  # minutes
        print(f"\nEstimated time remaining: ~{time_remaining:.1f} minutes")

    print(f"=" * 50)
else:
    print("Cache file not found yet. Process may still be starting...")
