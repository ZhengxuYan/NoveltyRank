"""
Enrich OpenReview paper data with Semantic Scholar author information

Reads papers from CSV/Parquet and adds author metrics (h-index, citations, etc.)
"""

import pandas as pd
import argparse
import time
from datetime import datetime
from semantic_scholar_author import SemanticScholarAuthor


def parse_author_names(author_string):
    """Parse comma or semicolon-separated author names"""
    if pd.isna(author_string) or author_string == "":
        return []

    # Try semicolon first (ArXiv format), then comma (OpenReview format)
    if ";" in author_string:
        separator = ";"
    else:
        separator = ","

    return [name.strip() for name in author_string.split(separator)]


def enrich_papers_with_author_info(
    input_file, output_file=None, api_key=None, sample_size=None, delay=0.5
):
    """
    Enrich papers dataframe with author information from Semantic Scholar

    Args:
        input_file (str): Path to input CSV/Parquet file
        output_file (str): Path to output file (auto-generated if None)
        api_key (str): Semantic Scholar API key for higher rate limits
        sample_size (int): Process only first N papers (for testing)
        delay (float): Delay between author queries in seconds
    """
    print(f"\nLoading papers from {input_file}...")

    # Load data
    if input_file.endswith(".parquet"):
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file)

    print(f"Loaded {len(df)} papers")

    # Sample if requested
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} papers for testing...")
        df = df.head(sample_size)

    # Find author column (case-insensitive)
    author_col = None
    for col in df.columns:
        if col.lower() == "authors":
            author_col = col
            break

    if author_col is None:
        raise ValueError("No 'authors' column found in the data!")

    print(f"Using column '{author_col}' for author information")

    # Initialize Semantic Scholar fetcher
    fetcher = SemanticScholarAuthor(api_key=api_key)
    fetcher.request_delay = delay

    # Extract all unique authors
    print("\nExtracting unique authors...")
    all_authors = set()
    for author_string in df[author_col]:
        authors = parse_author_names(author_string)
        all_authors.update(authors)

    all_authors = sorted(list(all_authors))
    print(f"Found {len(all_authors)} unique authors")

    # Fetch author information
    print("\nFetching author information from Semantic Scholar...")
    print(f"(This will take ~{len(all_authors) * delay / 60:.1f} minutes)")

    author_info = {}
    for i, author_name in enumerate(all_authors):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(all_authors)} authors...")

        # Search for author
        author_data = fetcher.find_best_match(author_name)

        if author_data:
            author_info[author_name] = {
                "semantic_scholar_id": author_data.get("authorId"),
                "h_index": author_data.get("hIndex"),
                "citation_count": author_data.get("citationCount"),
                "paper_count": author_data.get("paperCount"),
                "affiliations": ", ".join(author_data.get("affiliations", [])),
            }
        else:
            # Author not found
            author_info[author_name] = {
                "semantic_scholar_id": None,
                "h_index": None,
                "citation_count": None,
                "paper_count": None,
                "affiliations": None,
            }

    print(
        f"\nSuccessfully fetched info for {sum(1 for v in author_info.values() if v['semantic_scholar_id'] is not None)}/{len(all_authors)} authors"
    )

    # Add author metrics to each paper
    print("\nEnriching papers with author metrics...")

    def calculate_paper_author_metrics(author_string):
        """Calculate aggregate author metrics for a paper"""
        authors = parse_author_names(author_string)

        if not authors:
            return {
                "num_authors": 0,
                "max_h_index": None,
                "avg_h_index": None,
                "max_citations": None,
                "avg_citations": None,
                "first_author_h_index": None,
                "last_author_h_index": None,
            }

        h_indices = []
        citations = []

        for author in authors:
            if author in author_info and author_info[author]["h_index"] is not None:
                h_indices.append(author_info[author]["h_index"])
                citations.append(author_info[author]["citation_count"])

        first_author_h = author_info.get(authors[0], {}).get("h_index")
        last_author_h = author_info.get(authors[-1], {}).get("h_index")

        return {
            "num_authors": len(authors),
            "max_h_index": max(h_indices) if h_indices else None,
            "avg_h_index": sum(h_indices) / len(h_indices) if h_indices else None,
            "max_citations": max(citations) if citations else None,
            "avg_citations": sum(citations) / len(citations) if citations else None,
            "first_author_h_index": first_author_h,
            "last_author_h_index": last_author_h,
        }

    # Calculate metrics for each paper
    metrics = df[author_col].apply(calculate_paper_author_metrics)
    metrics_df = pd.DataFrame(metrics.tolist())

    # Combine with original dataframe
    df_enriched = pd.concat([df, metrics_df], axis=1)

    # Generate output filename
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = input_file.rsplit(".", 1)[0]
        ext = input_file.rsplit(".", 1)[1]
        output_file = f"{base_name}_enriched_{timestamp}.{ext}"

    # Save enriched data
    print(f"\nSaving enriched data to {output_file}...")
    if output_file.endswith(".parquet"):
        df_enriched.to_parquet(output_file, index=False)
    else:
        df_enriched.to_csv(output_file, index=False)

    # Print statistics
    print("\n" + "=" * 60)
    print("ENRICHMENT SUMMARY")
    print("=" * 60)
    print(f"Total papers: {len(df_enriched)}")
    print(f"Total unique authors: {len(all_authors)}")
    print(
        f"Authors found in Semantic Scholar: {sum(1 for v in author_info.values() if v['semantic_scholar_id'] is not None)}"
    )
    print(f"\nAuthor metrics added:")
    print(f"  - num_authors")
    print(f"  - max_h_index")
    print(f"  - avg_h_index")
    print(f"  - max_citations")
    print(f"  - avg_citations")
    print(f"  - first_author_h_index")
    print(f"  - last_author_h_index")

    # Show some statistics
    print(
        f"\nPapers with author metrics: {df_enriched['max_h_index'].notna().sum()}/{len(df_enriched)}"
    )

    if df_enriched["max_h_index"].notna().any():
        print(f"\nMax h-index statistics:")
        print(f"  Mean: {df_enriched['max_h_index'].mean():.2f}")
        print(f"  Median: {df_enriched['max_h_index'].median():.2f}")
        print(f"  Max: {df_enriched['max_h_index'].max():.0f}")

    print("=" * 60)

    return df_enriched, author_info


def main():
    parser = argparse.ArgumentParser(
        description="Enrich OpenReview papers with Semantic Scholar author info"
    )
    parser.add_argument(
        "input_file", type=str, help="Path to input CSV/Parquet file with papers"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (auto-generated if not provided)",
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
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)",
    )

    args = parser.parse_args()

    # Run enrichment
    enrich_papers_with_author_info(
        input_file=args.input_file,
        output_file=args.output,
        api_key=args.api_key,
        sample_size=args.sample,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
