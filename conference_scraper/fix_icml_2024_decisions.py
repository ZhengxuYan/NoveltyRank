"""
Fix ICML 2024 decisions by fetching venue information from OpenReview
ICML 2024 stores decisions in the 'venue' field (e.g., "ICML 2024 Poster")
"""

import pandas as pd
import requests
import time
from datetime import datetime

# Read the existing CSV
input_file = "../results/icml_papers_2024_2025_20251109_153723.csv"
df = pd.read_csv(input_file)

print(f"Total papers loaded: {len(df)}")
print(f"\nYears distribution:")
print(df['year'].value_counts().sort_index())

# Filter for 2024 papers with empty decisions
df_2024 = df[df['year'] == 2024].copy()
print(f"\n2024 papers to fix: {len(df_2024)}")

# Function to fetch venue information for a paper
def fetch_venue(paper_id):
    """Fetch venue information from OpenReview API"""
    try:
        url = f"https://api2.openreview.net/notes?id={paper_id}"
        response = requests.get(url)
        data = response.json()
        
        notes = data.get('notes', [])
        if notes:
            content = notes[0].get('content', {})
            venue = content.get('venue', {})
            if isinstance(venue, dict):
                venue_value = venue.get('value', '')
            else:
                venue_value = venue
            
            # Extract decision from venue
            # e.g., "ICML 2024 Poster" -> "Accept (poster)"
            # e.g., "ICML 2024 Oral" -> "Accept (oral)"
            if 'Poster' in venue_value or 'poster' in venue_value:
                return 'Accept (poster)'
            elif 'Oral' in venue_value or 'oral' in venue_value:
                return 'Accept (oral)'
            elif 'Spotlight' in venue_value or 'spotlight' in venue_value:
                return 'Accept (spotlight)'
            elif venue_value and '2024' in venue_value:
                # If it has 2024 in venue, it's likely accepted
                return 'Accept'
            else:
                return ''
        
        time.sleep(0.5)  # Rate limiting
        return ''
    
    except Exception as e:
        print(f"\nError fetching venue for {paper_id}: {e}")
        return ''

# Update decisions for 2024 papers
print("\nFetching venue information for 2024 papers...")
print("Progress: ", end="", flush=True)

for idx, row in df_2024.iterrows():
    if (idx + 1) % 100 == 0:
        print("*", end="", flush=True)
    elif (idx + 1) % 10 == 0:
        print(".", end="", flush=True)
    
    paper_id = row['id']
    venue_decision = fetch_venue(paper_id)
    
    # Update the decision in the original dataframe
    df.loc[idx, 'decision'] = venue_decision

print("\n\nDone fetching venue information!")

# Show updated statistics
print(f"\n{'='*60}")
print("Updated statistics:")
print(f"\nDecisions distribution:")
print(df['decision'].value_counts())

print(f"\nDecisions by year:")
print(df.groupby('year')['decision'].value_counts())

# Save updated file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"../results/icml_papers_2024_2025_{timestamp}_with_decisions.csv"

df.to_csv(output_file, index=False, encoding='utf-8', escapechar='\\', quoting=1)

print(f"\n{'='*60}")
print(f"Updated file saved to:")
print(f"  {output_file}")
print(f"\n{'='*60}")

# Now split by year and filter for accepted papers
df_accepted = df[df['decision'].str.contains('Accept', case=False, na=False)]

print(f"\nAfter filtering for accepted papers: {len(df_accepted)}")
print(f"\nYears distribution (accepted only):")
print(df_accepted['year'].value_counts().sort_index())

# Split by year
df_2024_accepted = df_accepted[df_accepted['year'] == 2024]
df_2025_accepted = df_accepted[df_accepted['year'] == 2025]

print(f"\n{'='*60}")
print(f"2024 accepted papers: {len(df_2024_accepted)}")
print(f"2025 accepted papers: {len(df_2025_accepted)}")

# Save split files
output_2024 = f"../results/icml_papers_2024_accepted_{timestamp}.csv"
output_2025 = f"../results/icml_papers_2025_accepted_{timestamp}.csv"

df_2024_accepted.to_csv(output_2024, index=False, encoding='utf-8', escapechar='\\', quoting=1)
df_2025_accepted.to_csv(output_2025, index=False, encoding='utf-8', escapechar='\\', quoting=1)

print(f"\nSplit files saved:")
print(f"  - {output_2024}")
print(f"  - {output_2025}")
print(f"\n{'='*60}")

