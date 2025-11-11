import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "conference_scraper"))

import pandas as pd
import numpy as np
import time
import pickle
from datetime import datetime
from semantic_scholar_author import SemanticScholarAuthor


def parse_author_names(author_string):
    if pd.isna(author_string) or author_string == "":
        return []

    if ";" in author_string:
        separator = ";"
    else:
        separator = ","

    return [name.strip() for name in author_string.split(separator) if name.strip()]


def calculate_author_career_metrics(papers_info):
    if not papers_info or "papers" not in papers_info:
        return {}

    papers = papers_info.get("papers", [])
    if not papers:
        return {}

    years = [p.get("year") for p in papers if p.get("year")]
    if not years:
        return {}

    citations = [p.get("citationCount", 0) for p in papers]

    current_year = datetime.now().year
    first_year = min(years)
    last_year = max(years)
    years_active = last_year - first_year + 1

    recent_papers = [
        p for p in papers if p.get("year") and p.get("year") >= current_year - 3
    ]
    recent_paper_count = len(recent_papers)

    total_citations = sum(citations)
    avg_citations_per_paper = total_citations / len(papers) if papers else 0
    max_paper_citations = max(citations) if citations else 0

    return {
        "years_active": years_active,
        "first_pub_year": first_year,
        "last_pub_year": last_year,
        "career_stage": current_year - first_year,
        "recent_productivity": recent_paper_count,
        "avg_citations_per_paper": avg_citations_per_paper,
        "max_paper_citations": max_paper_citations,
        "total_citations": total_citations,
        "total_papers": len(papers),
    }


def fetch_comprehensive_author_info(author_name, fetcher):
    results = fetcher.search_author_by_name(author_name, limit=1)

    if not results:
        return None

    author = results[0]
    author_id = author.get("authorId")

    papers = fetcher.get_author_papers(author_id, limit=100)

    detailed = author.copy()
    detailed["papers"] = papers

    career_metrics = calculate_author_career_metrics(detailed)

    comprehensive_info = {
        "semantic_scholar_id": detailed.get("authorId"),
        "name": detailed.get("name"),
        "affiliations": ", ".join(detailed.get("affiliations", [])),
        "h_index": detailed.get("hIndex"),
        "citation_count": detailed.get("citationCount"),
        "paper_count": detailed.get("paperCount"),
        "years_active": career_metrics.get("years_active"),
        "first_pub_year": career_metrics.get("first_pub_year"),
        "last_pub_year": career_metrics.get("last_pub_year"),
        "career_stage": career_metrics.get("career_stage"),
        "recent_productivity": career_metrics.get("recent_productivity"),
        "avg_citations_per_paper": career_metrics.get("avg_citations_per_paper"),
        "max_paper_citations": career_metrics.get("max_paper_citations"),
    }

    return comprehensive_info


def enrich_arxiv_with_authors(
    input_file,
    output_file=None,
    cache_file=None,
    api_key=None,
    sample_size=None,
    delay=1.0,
    resume=True,
):
    print(f"Loading papers from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} papers")

    author_col = None
    for col in df.columns:
        if col.lower() == "authors":
            author_col = col
            break

    if author_col is None:
        raise ValueError("No 'authors' column found!")

    print(f"Using column '{author_col}' for author information")

    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} papers for testing...")
        df = df.head(sample_size)

    print("Extracting unique authors...")
    all_authors = set()
    for author_string in df[author_col]:
        authors = parse_author_names(author_string)
        all_authors.update(authors)

    all_authors = sorted(list(all_authors))
    print(f"Found {len(all_authors)} unique authors")

    if cache_file is None:
        cache_file = input_file.replace(".csv", "_author_cache.pkl")

    author_info = {}
    if resume and os.path.exists(cache_file):
        print(f"Loading cached author data from {cache_file}...")
        with open(cache_file, "rb") as f:
            author_info = pickle.load(f)
        print(f"Loaded {len(author_info)} cached authors")

    fetcher = SemanticScholarAuthor(api_key=api_key)
    fetcher.request_delay = delay

    remaining_authors = [a for a in all_authors if a not in author_info]

    if remaining_authors:
        print(f"Fetching info for {len(remaining_authors)} authors...")
        print(
            f"Estimated time: ~{len(remaining_authors) * delay / 60:.1f} minutes (~{len(remaining_authors) * delay / 3600:.1f} hours)"
        )

        if api_key is None:
            print(f"WARNING: No API key provided!")
            print(
                f"Free tier limit: 100 requests per 5 minutes (3 seconds per request)"
            )
            print(f"Current delay: {delay} seconds")
            if delay < 3.0:
                print(f"DELAY TOO SHORT! You will get rate limited.")
                print(f"Recommendation: Use --delay 3.5 or get an API key")
        else:
            print(
                f"With API key: 5,000 requests per 5 minutes (0.06 seconds per request)"
            )

        print(f"(Progress: . = 10 authors, * = 100 authors)")

        for i, author_name in enumerate(remaining_authors):
            if (i + 1) % 100 == 0:
                print("*", end="", flush=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(author_info, f)
            elif (i + 1) % 10 == 0:
                print(".", end="", flush=True)

            info = fetch_comprehensive_author_info(author_name, fetcher)

            if info:
                author_info[author_name] = info
            else:
                author_info[author_name] = {
                    "semantic_scholar_id": None,
                    "name": author_name,
                    "h_index": None,
                    "citation_count": None,
                    "paper_count": None,
                    "years_active": None,
                    "career_stage": None,
                    "recent_productivity": None,
                    "avg_citations_per_paper": None,
                }

        print()

        print(f"Saving author cache to {cache_file}...")
        with open(cache_file, "wb") as f:
            pickle.dump(author_info, f)

    found_count = sum(
        1 for v in author_info.values() if v.get("semantic_scholar_id") is not None
    )
    print(
        f"Successfully fetched info for {found_count}/{len(all_authors)} authors ({found_count/len(all_authors)*100:.1f}%)"
    )

    print("Calculating paper-level author features...")

    def calculate_paper_author_features(author_string):
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
                    "has_top_author",
                ]
            }

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

        features = {
            "num_authors": len(authors),
            "max_h_index": max(h_indices) if h_indices else None,
            "avg_h_index": np.mean(h_indices) if h_indices else None,
            "min_h_index": min(h_indices) if h_indices else None,
            "first_author_h_index": author_info.get(authors[0], {}).get("h_index"),
            "last_author_h_index": author_info.get(authors[-1], {}).get("h_index"),
            "max_citations": max(citations) if citations else None,
            "avg_citations": np.mean(citations) if citations else None,
            "total_author_citations": sum(citations) if citations else None,
            "max_career_stage": max(career_stages) if career_stages else None,
            "avg_career_stage": np.mean(career_stages) if career_stages else None,
            "min_career_stage": min(career_stages) if career_stages else None,
            "has_early_career_author": any(s <= 5 for s in career_stages)
            if career_stages
            else None,
            "has_senior_author": any(s >= 20 for s in career_stages)
            if career_stages
            else None,
            "max_recent_productivity": max(recent_productivities)
            if recent_productivities
            else None,
            "avg_recent_productivity": np.mean(recent_productivities)
            if recent_productivities
            else None,
            "max_avg_paper_citations": max(avg_paper_citations)
            if avg_paper_citations
            else None,
            "avg_avg_paper_citations": np.mean(avg_paper_citations)
            if avg_paper_citations
            else None,
            "author_diversity_score": np.std(h_indices) if len(h_indices) > 1 else 0,
            "has_top_author": any(h >= 50 for h in h_indices) if h_indices else False,
        }

        return features

    features_list = df[author_col].apply(calculate_paper_author_features)
    features_df = pd.DataFrame(features_list.tolist())

    df_enriched = pd.concat([df, features_df], axis=1)

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = input_file.replace(".csv", f"_enriched_{timestamp}.csv")

    print(f"Saving enriched data to {output_file}...")
    df_enriched.to_csv(output_file, index=False)

    print("\nEnrichment Summary")
    print(f"Total papers: {len(df_enriched)}")
    print(f"Total unique authors: {len(all_authors)}")
    print(
        f"Authors found in Semantic Scholar: {found_count} ({found_count/len(all_authors)*100:.1f}%)"
    )

    print(
        f"Papers with author metrics: {features_df['max_h_index'].notna().sum()}/{len(df_enriched)} ({features_df['max_h_index'].notna().sum()/len(df_enriched)*100:.1f}%)"
    )

    if features_df["max_h_index"].notna().any():
        print(f"H-index statistics:")
        print(f"  Max h-index mean: {features_df['max_h_index'].mean():.2f}")
        print(f"  Max h-index median: {features_df['max_h_index'].median():.2f}")
        print(f"  Max h-index max: {features_df['max_h_index'].max():.0f}")

        print(f"Career stage statistics:")
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

    return df_enriched, author_info


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Enrich ArXiv papers with Semantic Scholar author info for novelty ranking"
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
        help="Cache file for author info (default: input_file_author_cache.pkl)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Semantic Scholar API key (for higher rate limits)",
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
        default=3.5,
        help="Delay between requests in seconds (default: 3.5 for free tier, 0.1 with API key)",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't resume from cache, start fresh"
    )

    args = parser.parse_args()

    # Run enrichment
    enrich_arxiv_with_authors(
        input_file=args.input_file,
        output_file=args.output,
        cache_file=args.cache,
        api_key=args.api_key,
        sample_size=args.sample,
        delay=args.delay,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
