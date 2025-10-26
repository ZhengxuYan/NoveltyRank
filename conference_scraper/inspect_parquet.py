"""
Quick script to inspect the ICLR parquet file format
"""

import pandas as pd

# Read the parquet file
df = pd.read_parquet("results/iclr26v1.parquet")

print("=" * 80)
print("PARQUET FILE INSPECTION")
print("=" * 80)

# Basic info
print(f"\nTotal rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\n" + "=" * 80)
print("COLUMN NAMES AND TYPES")
print("=" * 80)
print(df.dtypes)

print("\n" + "=" * 80)
print("FIRST 5 ROWS")
print("=" * 80)
print(df.head())

print("\n" + "=" * 80)
print("COLUMN DETAILS")
print("=" * 80)
for col in df.columns:
    print(f"\n{col}:")
    print(f"  Type: {df[col].dtype}")
    print(f"  Non-null count: {df[col].notna().sum()}")
    print(f"  Null count: {df[col].isna().sum()}")
    if df[col].dtype == "object":
        print(
            f"  Sample value: {df[col].iloc[0][:100] if isinstance(df[col].iloc[0], str) else df[col].iloc[0]}"
        )
    else:
        print(f"  Sample value: {df[col].iloc[0]}")

print("\n" + "=" * 80)
print("DECISION BREAKDOWN")
print("=" * 80)
if "decision" in df.columns:
    print(df["decision"].value_counts())
else:
    print("No 'decision' column found")

print("\n" + "=" * 80)
print("YEAR BREAKDOWN")
print("=" * 80)
if "year" in df.columns:
    print(df.groupby("year").size())
else:
    print("No 'year' column found")

print("\n" + "=" * 80)
print("SAMPLE PAPER (full details)")
print("=" * 80)
print("\nFirst paper:")
for col in df.columns:
    print(f"\n{col}:")
    print(f"  {df[col].iloc[0]}")
