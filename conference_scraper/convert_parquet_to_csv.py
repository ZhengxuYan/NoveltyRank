"""
Convert the ICLR parquet file to CSV format
"""

import pandas as pd

# Read parquet
print("Reading parquet file...")
df = pd.read_parquet("results/iclr26v1.parquet")

print(f"Loaded {len(df)} papers")
print(f"Years covered: {df['year'].min()} - {df['year'].max()}")

# Convert scores from array to string for CSV
print("Converting scores array to string...")
df["scores"] = df["scores"].apply(
    lambda x: ", ".join(map(str, x)) if len(x) > 0 else ""
)

# Convert keywords from list to string
print("Converting keywords list to string...")
df["keywords"] = df["keywords"].apply(
    lambda x: ", ".join(x) if isinstance(x, list) else x
)

# Add OpenReview URL
print("Adding OpenReview URLs...")
df["openreview_url"] = df["id"].apply(lambda x: f"https://openreview.net/forum?id={x}")

# Reorder columns to include author_ids
columns = [
    "year",
    "id",
    "title",
    "authors",
    "author_ids",
    "abstract",
    "keywords",
    "decision",
    "scores",
    "labels",
    "openreview_url",
]
df = df[columns]

# Save to CSV
output_file = "results/iclr26v1_papers.csv"
print(f"Saving to {output_file}...")
df.to_csv(output_file, index=False, encoding="utf-8", escapechar="\\", quoting=1)

print(f"\n✅ Successfully converted {len(df)} papers to CSV!")
print(f"\nYears covered: {df['year'].min()} - {df['year'].max()}")
print(f"\nPapers per year:")
print(df.groupby("year").size())
print(f"\nDecision breakdown:")
print(df["decision"].value_counts())
print(f"\nPapers with decisions: {(df['decision'] != '').sum()}")
print(f"Papers without decisions: {(df['decision'] == '').sum()}")
print(
    f"\nAverage score (for papers with scores): {df[df['scores'] != '']['scores'].apply(lambda x: sum(map(float, x.split(', '))) / len(x.split(', '))).mean():.2f}"
)
