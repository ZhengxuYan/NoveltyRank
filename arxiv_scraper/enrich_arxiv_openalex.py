"""
Enrich ArXiv dataset with OpenAlex author information
MUCH FASTER than Semantic Scholar - no rate limit issues!

OpenAlex Rate Limits:
- 100,000 requests per day
- Max 10 requests per second
- With email (polite pool): More consistent response times
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "conference_scraper"))

import pandas as pd
import numpy as np
import time
import pickle
from datetime import datetime
from openalex_author import OpenAlexAuthor


def parse_author_names(author_string):
    """Parse semicolon or comma-separated author names"""
    if pd.isna(author_string) or author_string == "":
        return []

    # ArXiv uses semicolons
    if ";" in author_string:
        separator = ";"
    else:
        separator = ","

    return [name.strip() for name in author_string.split(separator) if name.strip()]


def calculate_author_career_metrics(works, counts_by_year):
    """
    Calculate career-stage metrics from author's works and yearly stats

    Args:
        works: List of author's works
        counts_by_year: List of dicts with year/works_count/cited_by_count
    """
    if not counts_by_year:
        return {}

    # Extract years with publications
    years_with_pubs = [
        item["year"] for item in counts_by_year if item.get("works_count", 0) > 0
    ]

    if not years_with_pubs:
        return {}

    current_year = datetime.now().year
    first_year = min(years_with_pubs)
    last_year = max(years_with_pubs)
    years_active = last_year - first_year + 1

    # Recent productivity (last 3 years)
    recent_years = [y for y in years_with_pubs if y >= current_year - 3]
    recent_paper_count = sum(
        item.get("works_count", 0)
        for item in counts_by_year
        if item.get("year", 0) >= current_year - 3
    )

    # Citation metrics from works
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
        "career_stage": current_year - first_year,  # Years since first paper
        "recent_productivity": recent_paper_count,  # Papers in last 3 years
        "avg_citations_per_paper": avg_citations_per_paper,
        "max_paper_citations": max_paper_citations,
        "total_citations": total_citations,
        "total_papers": len(works),
    }


def fetch_comprehensive_author_info(author_name, fetcher, debug=False):
    """
    Fetch comprehensive author information for novelty ranking using OpenAlex

    Returns dict with:
    - Basic info: h-index, citations, works
    - Career metrics: years active, career stage, productivity
    - Quality metrics: avg citations per paper
    """
    # Search for author
    if debug:
        print(f"  -> Searching for author...", flush=True)
    results = fetcher.search_author_by_name(author_name, limit=1)

    if not results:
        if debug:
            print(f"  -> Not found", flush=True)
        return None

    author = results[0]
    author_id = author.get("openalex_id")

    # Get detailed info
    if debug:
        print(f"  -> Getting detailed info for {author_id}...", flush=True)
    detailed = fetcher.get_author_by_id(author_id)

    if not detailed:
        return author  # Return basic info

    # Get works for career metrics
    if debug:
        print(f"  -> Getting works...", flush=True)
    works = fetcher.get_author_works(author_id, limit=200)

    # Calculate career metrics
    career_metrics = calculate_author_career_metrics(
        works, detailed.get("counts_by_year", [])
    )

    # Combine all information
    comprehensive_info = {
        # Basic OpenAlex info
        "openalex_id": detailed.get("openalex_id"),
        "name": detailed.get("name"),
        "orcid": detailed.get("orcid"),
        "affiliations": ", ".join(detailed.get("affiliations", [])),
        # Core metrics
        "h_index": detailed.get("h_index", 0),
        "i10_index": detailed.get("i10_index", 0),
        "citation_count": detailed.get("cited_by_count", 0),
        "paper_count": detailed.get("works_count", 0),
        "two_year_citedness": detailed.get("two_year_mean_citedness", 0),
        # Career stage metrics (valuable for novelty!)
        "years_active": career_metrics.get("years_active"),
        "first_pub_year": career_metrics.get("first_pub_year"),
        "last_pub_year": career_metrics.get("last_pub_year"),
        "career_stage": career_metrics.get("career_stage"),  # Years since first paper
        # Productivity metrics
        "recent_productivity": career_metrics.get(
            "recent_productivity"
        ),  # Last 3 years
        # Quality metrics
        "avg_citations_per_paper": career_metrics.get("avg_citations_per_paper"),
        "max_paper_citations": career_metrics.get("max_paper_citations"),
    }

    return comprehensive_info


def enrich_arxiv_with_authors(
    input_file,
    output_file=None,
    cache_file=None,
    email="your-email@example.com",
    sample_size=None,
    delay=0.35,
    resume=True,
    filter_years=None,
):
    """
    Enrich ArXiv dataset with comprehensive author information from OpenAlex

    Args:
        input_file: Path to arxiv CSV
        output_file: Path to save enriched CSV
        cache_file: Path to cache author info (to resume if interrupted)
        email: Your email for OpenAlex polite pool (faster access)
        sample_size: Process only N papers (for testing)
        delay: Delay between API requests (can be very small with OpenAlex!)
        resume: Resume from cache if available
        filter_years: List of years to filter (e.g., [2024, 2025])
    """
    print("\n" + "=" * 80)
    print("ARXIV AUTHOR ENRICHMENT USING OPENALEX")
    print("=" * 80)
    print("OpenAlex is MUCH faster - 100,000 requests/day!")

    # Load data
    print(f"\nLoading papers from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} papers")

    # Filter by year if requested
    if filter_years:
        # Find the date column (could be 'Published', 'Date', 'published', etc.)
        date_col = None
        for col in df.columns:
            if col.lower() in ["published", "date", "publication_date", "year"]:
                date_col = col
                break

        if date_col:
            print(f"\nFiltering papers by year using column '{date_col}'...")
            original_count = len(df)

            # Convert to datetime if it's not a year column
            if date_col.lower() != "year":
                df["_year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
                year_col = "_year"
            else:
                year_col = date_col

            # Filter by years
            df = df[df[year_col].isin(filter_years)]
            print(
                f"Filtered from {original_count} to {len(df)} papers (years: {filter_years})"
            )

            # Drop temporary year column if created
            if year_col == "_year":
                df = df.drop(columns=["_year"])
        else:
            print(
                f"\n⚠️  Warning: No date column found! Available columns: {list(df.columns)}"
            )
            print(f"   Proceeding without year filter...")

    print(f"Processing {len(df)} papers")

    # Find author column
    author_col = None
    for col in df.columns:
        if col.lower() == "authors":
            author_col = col
            break

    if author_col is None:
        raise ValueError("No 'authors' column found!")

    print(f"Using column '{author_col}' for author information")

    # Sample if requested
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} papers for testing...")
        df = df.head(sample_size)

    # Extract unique authors
    print("\nExtracting unique authors...")
    all_authors = set()
    for author_string in df[author_col]:
        authors = parse_author_names(author_string)
        all_authors.update(authors)

    all_authors = sorted(list(all_authors))
    print(f"Found {len(all_authors)} unique authors")

    # Set up cache
    if cache_file is None:
        cache_file = input_file.replace(".csv", "_openalex_cache.pkl")

    # Load cache if resuming
    author_info = {}
    if resume and os.path.exists(cache_file):
        print(f"\nLoading cached author data from {cache_file}...")
        with open(cache_file, "rb") as f:
            author_info = pickle.load(f)
        print(f"Loaded {len(author_info)} cached authors")

    # Fetch author information
    fetcher = OpenAlexAuthor(email=email)
    fetcher.request_delay = delay

    remaining_authors = [a for a in all_authors if a not in author_info]

    if remaining_authors:
        print(f"\nFetching info for {len(remaining_authors)} authors from OpenAlex...")
        print(
            f"Estimated time: ~{len(remaining_authors) * delay / 60:.1f} minutes (~{len(remaining_authors) * delay / 3600:.1f} hours)"
        )
        print(f"Rate: ~{1/delay:.0f} requests/second")
        print(f"\n(Progress: . = 10 authors, * = 100 authors, # = 1000 authors)\n")
        print("Starting fetch loop...", flush=True)

        for i, author_name in enumerate(remaining_authors):
            # Debug: Show first few authors being processed
            if i < 5:
                print(f"Processing author {i+1}: {author_name}", flush=True)

            # Progress indicator - more frequent so you know it's working!
            if (i + 1) % 1000 == 0:
                print(f"# [{i+1}/{len(remaining_authors)}]", end="", flush=True)
                # Save cache every 1000 authors
                with open(cache_file, "wb") as f:
                    pickle.dump(author_info, f)
            elif (i + 1) % 100 == 0:
                print("*", end="", flush=True)
            elif (i + 1) % 10 == 0:
                print(".", end="", flush=True)

            # Fetch comprehensive info
            try:
                # Debug mode for first 5 authors
                info = fetch_comprehensive_author_info(
                    author_name, fetcher, debug=(i < 5)
                )

                if info:
                    author_info[author_name] = info
                    if i < 5:
                        print(
                            f"  -> Success! h-index={info.get('h_index')}", flush=True
                        )
                else:
                    # Author not found - store placeholder
                    author_info[author_name] = {
                        "openalex_id": None,
                        "name": author_name,
                        "h_index": None,
                        "citation_count": None,
                        "paper_count": None,
                        "years_active": None,
                        "career_stage": None,
                        "recent_productivity": None,
                        "avg_citations_per_paper": None,
                    }
                    if i < 5:
                        print(f"  -> Not found", flush=True)
            except Exception as e:
                print(
                    f"\n⚠️  Error fetching info for '{author_name}': {e}\n", flush=True
                )
                # Store placeholder on error
                author_info[author_name] = {
                    "openalex_id": None,
                    "name": author_name,
                    "h_index": None,
                    "citation_count": None,
                    "paper_count": None,
                    "years_active": None,
                    "career_stage": None,
                    "recent_productivity": None,
                    "avg_citations_per_paper": None,
                }

        print("\n")

        # Save final cache
        print(f"Saving author cache to {cache_file}...")
        with open(cache_file, "wb") as f:
            pickle.dump(author_info, f)

    # Calculate statistics
    found_count = sum(
        1 for v in author_info.values() if v.get("openalex_id") is not None
    )
    print(
        f"\nSuccessfully fetched info for {found_count}/{len(all_authors)} authors ({found_count/len(all_authors)*100:.1f}%)"
    )

    # Enrich papers with author metrics
    print("\nCalculating paper-level author features for novelty ranking...")

    def calculate_paper_author_features(author_string):
        """Calculate comprehensive author features for a paper"""
        authors = parse_author_names(author_string)

        if not authors:
            return {
                col: None
                for col in [
                    "num_authors",
                    "max_h_index",
                    "avg_h_index",
                    "min_h_index",
                    "first_author_h_index",
                    "last_author_h_index",
                    "max_citations",
                    "avg_citations",
                    "total_author_citations",
                    "max_career_stage",
                    "avg_career_stage",
                    "min_career_stage",
                    "has_early_career_author",
                    "has_senior_author",
                    "max_recent_productivity",
                    "avg_recent_productivity",
                    "max_avg_paper_citations",
                    "avg_avg_paper_citations",
                    "author_diversity_score",
                    "has_top_author",  # h-index > 50
                ]
            }

        # Extract metrics
        h_indices = []
        citations = []
        career_stages = []
        recent_productivities = []
        avg_paper_citations = []

        for author in authors:
            if author in author_info:
                info = author_info[author]
                if info.get("h_index") is not None:
                    h_indices.append(info["h_index"])
                if info.get("citation_count") is not None:
                    citations.append(info["citation_count"])
                if info.get("career_stage") is not None:
                    career_stages.append(info["career_stage"])
                if info.get("recent_productivity") is not None:
                    recent_productivities.append(info["recent_productivity"])
                if info.get("avg_citations_per_paper") is not None:
                    avg_paper_citations.append(info["avg_citations_per_paper"])

        # Calculate features
        features = {
            "num_authors": len(authors),
            # H-index features
            "max_h_index": max(h_indices) if h_indices else None,
            "avg_h_index": np.mean(h_indices) if h_indices else None,
            "min_h_index": min(h_indices) if h_indices else None,
            "first_author_h_index": author_info.get(authors[0], {}).get("h_index"),
            "last_author_h_index": author_info.get(authors[-1], {}).get("h_index"),
            # Citation features
            "max_citations": max(citations) if citations else None,
            "avg_citations": np.mean(citations) if citations else None,
            "total_author_citations": sum(citations) if citations else None,
            # Career stage features (important for novelty!)
            "max_career_stage": max(career_stages) if career_stages else None,
            "avg_career_stage": np.mean(career_stages) if career_stages else None,
            "min_career_stage": min(career_stages) if career_stages else None,
            "has_early_career_author": any(s <= 5 for s in career_stages)
            if career_stages
            else None,
            "has_senior_author": any(s >= 20 for s in career_stages)
            if career_stages
            else None,
            # Productivity features
            "max_recent_productivity": max(recent_productivities)
            if recent_productivities
            else None,
            "avg_recent_productivity": np.mean(recent_productivities)
            if recent_productivities
            else None,
            # Quality features
            "max_avg_paper_citations": max(avg_paper_citations)
            if avg_paper_citations
            else None,
            "avg_avg_paper_citations": np.mean(avg_paper_citations)
            if avg_paper_citations
            else None,
            # Diversity features
            "author_diversity_score": np.std(h_indices) if len(h_indices) > 1 else 0,
            # Top author indicator
            "has_top_author": any(h >= 50 for h in h_indices) if h_indices else False,
        }

        return features

    # Apply feature calculation
    features_list = df[author_col].apply(calculate_paper_author_features)
    features_df = pd.DataFrame(features_list.tolist())

    # Combine with original dataframe
    df_enriched = pd.concat([df, features_df], axis=1)

    # Generate output filename
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = input_file.replace(".csv", f"_openalex_enriched_{timestamp}.csv")

    # Save enriched data
    print(f"\nSaving enriched data to {output_file}...")
    df_enriched.to_csv(output_file, index=False)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("ENRICHMENT SUMMARY")
    print("=" * 80)
    print(f"Total papers: {len(df_enriched)}")
    print(f"Total unique authors: {len(all_authors)}")
    print(
        f"Authors found in OpenAlex: {found_count} ({found_count/len(all_authors)*100:.1f}%)"
    )

    print(f"\nAuthor features added (for novelty ranking):")
    print(f"  Basic: num_authors, h-indices, citations")
    print(f"  Career: career_stage, early_career, senior_author")
    print(f"  Quality: avg_citations_per_paper")
    print(f"  Productivity: recent_productivity")
    print(f"  Diversity: author_diversity_score")
    print(f"  Flags: has_top_author, has_early_career_author")

    # Show statistics
    print(
        f"\nPapers with author metrics: {features_df['max_h_index'].notna().sum()}/{len(df_enriched)} ({features_df['max_h_index'].notna().sum()/len(df_enriched)*100:.1f}%)"
    )

    if features_df["max_h_index"].notna().any():
        print(f"\nH-index statistics:")
        print(f"  Max h-index mean: {features_df['max_h_index'].mean():.2f}")
        print(f"  Max h-index median: {features_df['max_h_index'].median():.2f}")
        print(f"  Max h-index max: {features_df['max_h_index'].max():.0f}")

        print(f"\nCareer stage statistics:")
        print(
            f"  Avg career stage mean: {features_df['avg_career_stage'].mean():.2f} years"
        )
        print(
            f"  Papers with early-career authors: {features_df['has_early_career_author'].sum()}"
        )
        print(f"  Papers with senior authors: {features_df['has_senior_author'].sum()}")
        print(
            f"  Papers with top authors (h>50): {features_df['has_top_author'].sum()}"
        )

    print("=" * 80)

    return df_enriched, author_info


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Enrich ArXiv papers with OpenAlex author info (FAST!)"
    )
    parser.add_argument("input_file", type=str, help="Path to ArXiv CSV file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (auto-generated if not provided)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Cache file for author info (default: input_file_openalex_cache.pkl)",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="your-email@example.com",
        help="Your email for OpenAlex polite pool (10x faster!)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only first N papers (for testing)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Delay between requests in seconds (default: 0.35, makes 3 API calls per author)",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't resume from cache, start fresh"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Filter papers by publication year (e.g., --years 2024 2025)",
    )

    args = parser.parse_args()

    # Run enrichment
    enrich_arxiv_with_authors(
        input_file=args.input_file,
        output_file=args.output,
        cache_file=args.cache,
        email=args.email,
        sample_size=args.sample,
        delay=args.delay,
        resume=not args.no_resume,
        filter_years=args.years,
    )


if __name__ == "__main__":
    main()
