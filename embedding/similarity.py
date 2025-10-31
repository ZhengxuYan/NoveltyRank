"""
Compute cosine similarity between paper embeddings and store top similar papers and corresponding similarity scores.

The output final train and test datasets include additional columns:
- 'classification_embedding': The feature embeddings used for novelty classification.
- 'proximity_embedding': The proximity embeddings used for similarity calculations.
- 'top_10_similar': Dictionary of top 10 similar papers with their Link/DOI and similarity scores.
- 'max_similarity': Maximum similarity score among the top similar papers.
- 'avg_similarity': Average similarity score among the top similar papers.

"""

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from tqdm import tqdm

# Load the embeddings
with open('output/embeddings_train.pkl', 'rb') as f:
    embeddings_train = pickle.load(f)

with open('output/embeddings_test.pkl', 'rb') as f:
    embeddings_test = pickle.load(f)

# Combine train and test embeddings
df = pd.concat([embeddings_train, embeddings_test], ignore_index=True)

# Convert 'Publication Date' to datetime if it's not already
if 'Publication Date' in df.columns:
    df['Publication Date'] = pd.to_datetime(df['Publication Date'])

# Initialize columns for similarity results
df['top_10_similar'] = None
df['max_similarity'] = None
df['avg_similarity'] = None

# Define papers that need similarity calculation 
# Papers with source == ICLR-2024, ICLR-2025, or Publication Date >= 2024-01-01 & label=0
target_mask = (
    (df['source'] == 'ICLR-2024') |
    (df['source'] == 'ICLR-2025') |
    ((df['Publication Date'] >= pd.Timestamp('2024-01-01')) & (df['label'] == 0))
)

target_indices = df[target_mask].index

print(f"\nProcessing {len(target_indices)} papers for similarity calculation...")
print("="*60)

# Process each target paper
for idx in tqdm(target_indices, desc="Computing similarities"):
    paper = df.loc[idx]

    # Determine comparison papers based on source and publication date
    if paper['source'] == 'ICLR-2024':
        # Compare with ICLR-2023 and papers published before 2023-09-28 (ICLR 2024 submission deadline) with label=0
        comparison_mask = (
            (df['source'] == 'ICLR-2023') |
            ((df['Publication Date'] < pd.Timestamp('2023-09-28')) & (df['label'] == 0))
        )
    elif paper['source'] == 'ICLR-2025':
        # Compare with ICLR-2024 and papers published before 2024-10-01 (ICLR 2025 submission deadline) with label=0
        comparison_mask = (
            (df['source'] == 'ICLR-2024') |
            ((df['Publication Date'] < pd.Timestamp('2024-10-01')) & (df['label'] == 0))
        )
    else:
        # For papers with label=0 and Publication Date >= 2024-01-01, compare with papers published before them
        comparison_mask = df['Publication Date'] < paper['Publication Date']

    # Exclude the paper itself from comparison
    comparison_mask = comparison_mask & (df.index != idx)
    comparison_indices = df[comparison_mask].index

    if len(comparison_indices) == 0:
        continue
    
    # Calculate cosine similarity
    paper_embedding = df.loc[idx, 'Proximity_embedding'].reshape(1, -1)
    comparison_embeddings = np.vstack(df.loc[comparison_indices, 'Proximity_embedding'].values)

    similarities = cosine_similarity(paper_embedding, comparison_embeddings)[0]
    
    # Get top 10 most similar papers
    top_10_idx = np.argsort(similarities)[-10:][::-1]
    top_10_papers = comparison_indices[top_10_idx]
    top_10_scores = similarities[top_10_idx]
    
    # Create dictionary of Link/DOI and similarity scores
    top_10_dict = {
        df.loc[top_10_papers[i], 'Link/DOI']: float(top_10_scores[i])
        for i in range(len(top_10_papers))
    }
    
    # Store results
    df.at[idx, 'top_10_similar'] = top_10_dict
    df.at[idx, 'max_similarity'] = float(np.max(top_10_scores)) if len(top_10_scores) > 0 else None
    df.at[idx, 'avg_similarity'] = float(np.mean(top_10_scores)) if len(top_10_scores) > 0 else None

# Split back to train and test
# Test papers: source == 'ICLR-2025' or (Publication Date >= 2025-01-01 & label == 0)
test_mask = (
    (df['source'] == 'ICLR-2025') |
    ((df['Publication Date'] >= pd.Timestamp('2025-01-01')) & (df['label'] == 0))
)

data_test = df[test_mask].reset_index(drop=True)
data_train = df[~test_mask].reset_index(drop=True)

# Save the results
print("\n" + "="*60)
print("Saving results...")
print("="*60)

with open('data_train.pkl', 'wb') as f:
    pickle.dump(data_train, f)
print(f"✓ Saved train set to data_train.pkl")

with open('data_test.pkl', 'wb') as f:
    pickle.dump(data_test, f)
print(f"✓ Saved test set to data_test.pkl")

print("\n" + "="*60)
print("Summary:")
print("="*60)
print(f"Total papers processed: {len(target_indices)}")
print(f"Train set size: {len(data_train)}")
print(f"Test set size: {len(data_test)}")
print("="*60)