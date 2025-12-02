
import os
import argparse
import datetime
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import arxiv
# import faiss # Removed due to segfault
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModel
from adapters import AutoAdapterModel
from openalex_utils import enrich_papers_with_openalex

# ======================== Configuration ========================

class Config:
    # Model
    SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
    MAX_LEN = 512
    
    # Features
    USE_CLASSIFICATION_EMB = True
    USE_PROXIMITY_EMB = True
    USE_SIMILARITY_FEATURES = True
    
    # Paths
    MODEL_PATH = "models/siamese_scibert_multimodal/best_siamese_model_v3.pth"

# ======================== Model Definition ========================

class SiameseSciBERT(nn.Module):
    def __init__(self, config):
        super(SiameseSciBERT, self).__init__()
        self.config = config
        
        self.scibert = AutoModel.from_pretrained(config.SCIBERT_MODEL)
        self.hidden_size = self.scibert.config.hidden_size
        
        total_dim = self.hidden_size
        if config.USE_CLASSIFICATION_EMB: total_dim += 768
        if config.USE_PROXIMITY_EMB: total_dim += 768
        if config.USE_SIMILARITY_FEATURES: total_dim += 2
        
        # Scoring Head (Outputs a single scalar)
        self.score_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Scalar output
        )
        
    def forward_one(self, input_ids, attention_mask, classification_emb=None, proximity_emb=None, similarity_features=None):
        outputs = self.scibert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        features = [cls_output]
        if self.config.USE_CLASSIFICATION_EMB and classification_emb is not None:
            features.append(classification_emb)
        if self.config.USE_PROXIMITY_EMB and proximity_emb is not None:
            features.append(proximity_emb)
        if self.config.USE_SIMILARITY_FEATURES and similarity_features is not None:
            features.append(similarity_features)
            
        combined = torch.cat(features, dim=1)
        score = self.score_head(combined)
        return score

# ======================== Helper Functions ========================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

import subprocess

def fetch_recent_papers(days=30):
    """Fetch recent papers using Scrapy spider (daily batch)."""
    print(f"Fetching papers from the last {days} days using Scrapy (daily batch)...")
    
    # Calculate dates
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    scraper_dir = os.path.join(project_root, "scrapers", "arxiv_scraper")
    # Use daily_batch.csv which is overwritten each time
    results_file = os.path.join(scraper_dir, "results", "daily_batch.csv")
    
    # Remove existing results file to ensure fresh data
    if os.path.exists(results_file):
        os.remove(results_file)
        print(f"Removed existing results file: {results_file}")
        
    # Construct command
    # scrapy crawl arxiv_daily -a start_date=YYYY-MM-DD -a end_date=YYYY-MM-DD
    cmd = [
        "scrapy", "crawl", "arxiv_daily",
        "-a", f"start_date={start_date}",
        "-a", f"end_date={end_date}"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=scraper_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Scrapy spider: {e}")
        return pd.DataFrame()
        
    # Read results
    if not os.path.exists(results_file):
        print("No results file generated.")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(results_file)
        print(f"Loaded {len(df)} papers from CSV.")
        
        # Map columns to expected format
        # CSV: Title, Abstract, Categories, Primary Category, Publication Date, arXiv ID, arXiv URL, Is Accepted, Acceptance Details
        
        df = df.rename(columns={
            "Title": "title",
            "Abstract": "abstract",
            "Categories": "categories",
            "Primary Category": "primary_category",
            "Publication Date": "published",
            "arXiv ID": "arxiv_id",
            "arXiv URL": "url",
            "Is Accepted": "is_accepted",
            "Acceptance Details": "acceptance_details",
            "Authors": "authors",
            "DOI": "doi",
            "Updated Date": "updated_date",
            "Author Affiliations": "affiliations",
            "Comment": "comment",
            "Journal Reference": "journal_ref"
        })
        
        # Ensure required columns exist
        required_cols = ["title", "abstract", "categories", "primary_category", "published", "arxiv_id", "url"]
        for col in required_cols:
            if col not in df.columns:
                print(f"Missing column: {col}")
                return pd.DataFrame()
        
        # Keep new columns if they exist
        cols_to_keep = required_cols
        optional_cols = ["is_accepted", "acceptance_details", "authors", "doi", "updated_date", "affiliations", "comment", "journal_ref"]
        for col in optional_cols:
            if col in df.columns:
                cols_to_keep.append(col)
            
        df = df[cols_to_keep]
        
        # Deduplicate by arxiv_id (papers can appear in multiple categories)
        initial_len = len(df)
        df = df.drop_duplicates(subset=["arxiv_id"])
        print(f"Deduplicated fetched papers: {initial_len} -> {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"Error reading results CSV: {e}")
        return pd.DataFrame()

def generate_specter_embeddings(df, device):
    """Generate SPECTER2 embeddings (classification and proximity)."""
    print("Generating SPECTER2 embeddings...")
    
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
    model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
    model.load_adapter("allenai/specter2_classification", source="hf", load_as="classification", set_active=True)
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    model.to(device)
    model.eval()
    
    classification_embs = []
    proximity_embs = []
    
    batch_size = 8
    
    for i in tqdm(range(0, len(df), batch_size), desc="Embedding"):
        batch = df.iloc[i:i+batch_size]
        texts = [p["title"] + tokenizer.sep_token + p["abstract"] for _, p in batch.iterrows()]
        
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        
        with torch.no_grad():
            # Classification embedding
            model.set_active_adapters("classification")
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            classification_embs.extend(cls_emb)
            
            # Proximity embedding
            model.set_active_adapters("proximity")
            outputs = model(**inputs)
            prox_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            proximity_embs.extend(prox_emb)
            
    df["classification_embedding"] = classification_embs
    df["proximity_embedding"] = proximity_embs
    return df

def compute_similarity_features(target_df, reference_df):
    """Compute max_similarity and avg_similarity against reference dataset (365-day window)."""
    print("Computing similarity features (365-day window)...")
    
    # Filter reference df for valid embeddings
    reference_df = reference_df[reference_df["proximity_embedding"].notna()]
    valid_mask = reference_df["proximity_embedding"].apply(lambda x: len(x) == 768 if isinstance(x, (list, np.ndarray)) else False)
    reference_df = reference_df[valid_mask].copy()
    
    if len(reference_df) == 0:
        print("Error: No valid reference embeddings found.")
        return target_df

    # Parse dates
    print("Parsing dates...")
    try:
        target_df["published_dt"] = pd.to_datetime(target_df["published"], errors='coerce')
        reference_df["published_dt"] = pd.to_datetime(reference_df["Publication Date"], errors='coerce')
        
        # Drop rows with invalid dates in reference (optional, but good for safety)
        reference_df = reference_df.dropna(subset=["published_dt"])
        
        # Sort reference by date for potential optimization (though we'll filter by mask)
        reference_df = reference_df.sort_values("published_dt")
        
        # Pre-convert reference embeddings to a single numpy array for indexing
        # But since we need to filter dynamically, we might need to slice or index into this
        # Actually, keeping them in DF and converting subset to numpy is safer/easier
        
    except Exception as e:
        print(f"Error parsing dates: {e}")
        return target_df

    max_sims = []
    avg_sims = []
    top_10_sims_list = []
    
    print(f"Processing {len(target_df)} target papers...")
    
    for idx, row in tqdm(target_df.iterrows(), total=len(target_df), desc="Similarity"):
        target_date = row["published_dt"]
        if pd.isna(target_date):
            max_sims.append(0.0)
            avg_sims.append(0.0)
            top_10_sims_list.append([])
            continue
            
        start_date = target_date - datetime.timedelta(days=365)
        
        # Filter reference papers within [start_date, target_date]
        window_mask = (reference_df["published_dt"] >= start_date) & (reference_df["published_dt"] <= target_date)
        window_refs = reference_df[window_mask].copy()
        
        # Exclude the paper itself if present
        target_id = str(row["arxiv_id"])
        # Normalize target ID (remove version if present)
        if "v" in target_id:
             target_id = target_id.split("v")[0]
             
        # Filter out self
        # Reference IDs might also have versions
        window_refs["normalized_id"] = window_refs["arXiv ID"].astype(str).apply(lambda x: x.split("v")[0] if "v" in x else x)
        window_refs = window_refs[window_refs["normalized_id"] != target_id]
        
        if len(window_refs) == 0:
            max_sims.append(0.0)
            avg_sims.append(0.0)
            top_10_sims_list.append([])
            continue
            
        # Compute similarity
        target_emb = np.array(row["proximity_embedding"], dtype='float32').reshape(1, -1)
        ref_embs = np.stack(window_refs["proximity_embedding"].values).astype('float32')
        
        # Cosine similarity: (1, 768) x (N, 768).T -> (1, N)
        sims = cosine_similarity(target_emb, ref_embs)[0]
        
        # Top 10
        if len(sims) > 0:
            # Sort descending
            top_k_indices = np.argsort(sims)[-10:][::-1]
            top_k_sims = sims[top_k_indices]
            
            max_sims.append(float(top_k_sims[0]))
            avg_sims.append(float(np.mean(top_k_sims)))
            
            # Construct top_10_similar list
            current_top_10 = []
            for k_idx, sim_score in zip(top_k_indices, top_k_sims):
                 ref_row = window_refs.iloc[k_idx]
                 current_top_10.append({
                     "arxiv_id": ref_row["arXiv ID"],
                     "similarity_score": float(sim_score),
                     "embedding": ref_row["proximity_embedding"],
                     "publication_date": str(ref_row["published_dt"].date())
                 })
            top_10_sims_list.append(current_top_10)
        else:
            max_sims.append(0.0)
            avg_sims.append(0.0)
            top_10_sims_list.append([])
            
    target_df["max_similarity"] = max_sims
    target_df["avg_similarity"] = avg_sims
    target_df["top_10_similar"] = top_10_sims_list
    
    # Cleanup temp column
    if "published_dt" in target_df.columns:
        target_df = target_df.drop(columns=["published_dt"])
        
    return target_df

def update_reference_dataset(new_papers_df):
    """Update the reference dataset with new papers."""
    print("\n" + "="*80)
    print("Updating Reference Dataset")
    print("="*80)
    
    try:
        dataset_name = "JasonYan777/novelty-rank-with-similarities-final"
        print(f"Loading {dataset_name}...")
        ds = load_dataset(dataset_name)
        
        # We will append to the 'test' split as these are new papers
        test_ds = ds["test"]
        
        # Remove artifact columns if present
        if "__index_level_0__" in test_ds.column_names:
            print("Removing __index_level_0__ from reference dataset...")
            test_ds = test_ds.remove_columns(["__index_level_0__"])
        
        # Prepare new papers dataframe to match reference schema
        # Reference columns: ['arXiv ID', 'arXiv URL', 'PDF URL', 'DOI', 'Publication Date', 'Updated Date', 'Title', 'Authors', 'Author Affiliations', 'Abstract', 'Categories', 'Primary Category', 'Comment', 'Journal Reference', 'Matched Conferences', 'label', 'source', 'classification_embedding', 'proximity_embedding', 'top_10_similar', 'max_similarity', 'avg_similarity']
        
        df = new_papers_df.copy()
        
        # Map columns
        rename_map = {
            "arxiv_id": "arXiv ID",
            "url": "arXiv URL",
            "title": "Title",
            "authors": "Authors",
            "abstract": "Abstract",
            "categories": "Categories",
            "primary_category": "Primary Category",
            "published": "Publication Date"
        }
        df = df.rename(columns=rename_map)
        
        # Add missing columns
        # Add missing columns
        # PDF URL can be derived
        df["PDF URL"] = df["arXiv ID"].apply(lambda x: f"https://arxiv.org/pdf/{x}.pdf" if x else None)
        
        # Map fields if they exist in the dataframe, otherwise None
        df["DOI"] = df["doi"] if "doi" in df.columns else None
        df["Updated Date"] = df["updated_date"] if "updated_date" in df.columns else df["Publication Date"]
        df["Author Affiliations"] = df["affiliations"] if "affiliations" in df.columns else None
        df["Comment"] = df["comment"] if "comment" in df.columns else None
        df["Journal Reference"] = df["journal_ref"] if "journal_ref" in df.columns else None
        
        df["Matched Conferences"] = None
        df["label"] = None # No label for new papers
        df["source"] = "arxiv_daily_ranking"
        
        # Ensure all columns exist and are in correct order
        ref_features = test_ds.features
        
        # Convert to Dataset to handle type casting
        # But first ensure dataframe has all columns
        for col in ref_features.keys():
            if col not in df.columns:
                df[col] = None
                
        # Select only relevant columns
        df = df[list(ref_features.keys())]
        
        # Convert to Dataset
        new_ds = Dataset.from_pandas(df)
        
        # Cast features to match reference
        new_ds = new_ds.cast(ref_features)
        
        print(f"Appending {len(new_ds)} new papers to test set ({len(test_ds)} existing)...")
        try:
            updated_test_ds = concatenate_datasets([test_ds, new_ds])
        except ValueError as e:
            if "must be identical" in str(e):
                print("Column mismatch detected during concatenation. Attempting to resolve...")
                # Identify mismatch
                ref_cols = set(test_ds.column_names)
                new_cols = set(new_ds.column_names)
                
                # Check for __index_level_0__ specifically in features if not in column_names
                if "__index_level_0__" not in ref_cols and "__index_level_0__" in test_ds.features:
                     print("Found hidden __index_level_0__ in reference features. Removing...")
                     test_ds = test_ds.remove_columns(["__index_level_0__"])
                     ref_cols = set(test_ds.column_names)

                # If test_ds has extra columns
                extra_in_ref = ref_cols - new_cols
                if extra_in_ref:
                    print(f"Removing extra columns from reference dataset: {extra_in_ref}")
                    test_ds = test_ds.remove_columns(list(extra_in_ref))
                
                # If new_ds has extra columns
                extra_in_new = new_cols - ref_cols
                if extra_in_new:
                    print(f"Removing extra columns from new papers: {extra_in_new}")
                    new_ds = new_ds.remove_columns(list(extra_in_new))
                    
                # Retry concatenation
                updated_test_ds = concatenate_datasets([test_ds, new_ds])
            else:
                raise e
        
        # Update dataset dict
        ds["test"] = updated_test_ds
        
        print(f"Pushing updated dataset to {dataset_name}...")
        ds.push_to_hub(dataset_name)
        print("Successfully updated reference dataset!")
        
    except Exception as e:
        print(f"Error updating reference dataset: {e}")


import re

def clean_text(text):
    """
    Remove sentences that contain URLs or code availability phrases to prevent bias.
    """
    if not isinstance(text, str):
        return ""
        
    # Simple sentence splitting by punctuation (.!?) followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    cleaned_sentences = []
    
    # Regex for URLs
    url_pattern = r'http\S+|www\.\S+|github\.com'
    
    # Phrases to trigger removal (case insensitive)
    code_phrases = [
        r'code is available',
        r'code available',
        r'source code',
        r'implementation is available',
        r'implementation available',
        r'project page',
        r'github'
    ]
    
    for sent in sentences:
        # Check for URL
        if re.search(url_pattern, sent, re.IGNORECASE):
            continue
            
        # Check for phrases
        found_phrase = False
        for phrase in code_phrases:
            if re.search(phrase, sent, re.IGNORECASE):
                found_phrase = True
                break
        
        if found_phrase:
            continue
            
        cleaned_sentences.append(sent)
        
    return ' '.join(cleaned_sentences)

def rank_papers(df, model, tokenizer, device, config):
    """Rank papers using the Siamese model."""
    print("Ranking papers...")
    model.eval()
    
    scores = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        # Prepare inputs
        # Clean abstract to remove URLs
        abstract = clean_text(row['abstract'])
        text = f"{row['title']} [SEP] {abstract} [SEP] {row['categories']}"
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].flatten().unsqueeze(0).to(device)
        attention_mask = encoding["attention_mask"].flatten().unsqueeze(0).to(device)
        
        cls_emb = torch.tensor(row["classification_embedding"], dtype=torch.float).unsqueeze(0).to(device)
        prox_emb = torch.tensor(row["proximity_embedding"], dtype=torch.float).unsqueeze(0).to(device)
        sim_feats = torch.tensor([row["max_similarity"], row["avg_similarity"]], dtype=torch.float).unsqueeze(0).to(device)
        
        with torch.no_grad():
            score = model.forward_one(
                input_ids, 
                attention_mask, 
                classification_emb=cls_emb, 
                proximity_emb=prox_emb, 
                similarity_features=sim_feats
            )
            scores.append(score.item())
            
    df["novelty_score"] = scores
    return df.sort_values("novelty_score", ascending=False)

# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser(description="Rank recent arXiv preprints by novelty")
    parser.add_argument("--days", type=int, default=30, help="Number of days to look back")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset instead of incremental update")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers to process (for testing)")
    parser.add_argument("--skip-enrich", action="store_true", help="Skip OpenAlex author enrichment")
    parser.add_argument("--no-push", action="store_true", help="Do not push results to Hugging Face (dry run)")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Fetch Papers (Daily Batch)
    new_papers_df = fetch_recent_papers(days=args.days)
    if len(new_papers_df) == 0:
        print("No papers found.")
        return

    # 2. Load Existing Dataset from Hugging Face
    existing_df = pd.DataFrame()
    unique_new_papers_df = new_papers_df
    
    if not args.overwrite:
        print("Loading existing dataset from Hugging Face...")
        try:
            existing_ds = load_dataset("JasonYan777/novelty-ranked-preprints", split="train")
            existing_df = existing_ds.to_pandas()
            print(f"Loaded {len(existing_df)} existing papers.")
            
            # Deduplicate: Filter out papers already in existing_df
            # Normalize IDs by removing version suffix (e.g., v1)
            import re
            def normalize_id(aid):
                return re.sub(r'v\d+$', '', str(aid))
                
            existing_ids = set(existing_df["arxiv_id"].apply(normalize_id))
            
            # Ensure new papers arxiv_id is string
            new_papers_df["arxiv_id"] = new_papers_df["arxiv_id"].astype(str)
            
            # Filter
            # We need to check normalized new IDs against normalized existing IDs
            # But we want to keep the original ID in the dataframe (or maybe normalized? let's keep original)
            new_papers_df["normalized_id"] = new_papers_df["arxiv_id"].apply(normalize_id)
            
            unique_new_papers_df = new_papers_df[~new_papers_df["normalized_id"].isin(existing_ids)].copy()
            
            # Drop temp column
            unique_new_papers_df = unique_new_papers_df.drop(columns=["normalized_id"])
            
            print(f"Found {len(unique_new_papers_df)} new unique papers.")
            
            if len(unique_new_papers_df) == 0:
                print("No new papers to process. Exiting.")
                return

        except Exception as e:
            print(f"Could not load existing dataset (or it doesn't exist): {e}")
            print("Proceeding with all fetched papers as new.")
            existing_df = pd.DataFrame()
            unique_new_papers_df = new_papers_df
    else:
        print("Overwrite mode enabled. Ignoring existing dataset.")

    # Apply limit if specified
    if args.limit and len(unique_new_papers_df) > args.limit:
        print(f"Limiting to {args.limit} papers.")
        unique_new_papers_df = unique_new_papers_df.head(args.limit)

    # 2.5 Enrich with OpenAlex Author Data
    # We do this for unique_new_papers_df before embeddings/ranking so we have the data
    # Note: This might take some time due to API rate limits
    if not args.skip_enrich:
        print("\n" + "="*80)
        print("Enriching with OpenAlex Author Data")
        print("="*80)
        
        # Define cache file path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cache_path = os.path.join(project_root, "scripts", "siamese_scibert", "openalex_cache.pkl")
        
        unique_new_papers_df = enrich_papers_with_openalex(unique_new_papers_df, cache_file=cache_path)
    else:
        print("\nSkipping OpenAlex author enrichment as requested.")

    # 3. Generate Embeddings for NEW papers only
    unique_new_papers_df = generate_specter_embeddings(unique_new_papers_df, device)
    
    # 4. Load Reference Dataset (Training Data + Test Data)
    print("Loading reference dataset for similarity...")
    dataset_name = "JasonYan777/novelty-rank-with-similarities-final"
    # Load both splits
    ds_dict = load_dataset(dataset_name)
    
    # Concatenate train and test if they exist
    dfs = []
    if "train" in ds_dict:
        dfs.append(ds_dict["train"].to_pandas())
    if "test" in ds_dict:
        dfs.append(ds_dict["test"].to_pandas())
        
    if dfs:
        ref_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(ref_df)} reference papers (train + test).")
    else:
        print("Warning: No reference data found in train or test splits.")
        ref_df = pd.DataFrame()
    
    # Rename columns to match
    ref_df = ref_df.rename(columns={
        "classification_embedding": "classification_embedding", 
        "proximity_embedding": "proximity_embedding"
    })
    
    # 5. Compute Similarity for NEW papers
    unique_new_papers_df = compute_similarity_features(unique_new_papers_df, ref_df)
    
    # 6. Load Model and Rank NEW papers
    print("Loading ranking model...")
    tokenizer = AutoTokenizer.from_pretrained(Config.SCIBERT_MODEL)
    model = SiameseSciBERT(Config)
    
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_PATH}")
    else:
        print(f"Error: Model not found at {Config.MODEL_PATH}")
        return
        
    model.to(device)
    
    ranked_new_df = rank_papers(unique_new_papers_df, model, tokenizer, device, Config)
    
    # 7. Merge and Display
    print("\n" + "="*80)
    print(f"Top 10 Most Novel NEW Papers")
    print("="*80)
    
    for i, (idx, row) in enumerate(ranked_new_df.head(10).iterrows()):
        print(f"{i+1}. {row['title']}")
        print(f"   Score: {row['novelty_score']:.4f} | Categories: {row['categories']}")
        print(f"   URL: {row['url']}")
        print("-" * 80)
        
    # Combine with existing data
    if not existing_df.empty:
        # Ensure columns match
        # If existing_df doesn't have new columns (is_accepted, acceptance_details), add them
        if "is_accepted" not in existing_df.columns:
            existing_df["is_accepted"] = None
        if "acceptance_details" not in existing_df.columns:
            existing_df["acceptance_details"] = None
        if "authors" not in existing_df.columns:
            existing_df["authors"] = None
            
        # Select columns to keep for final dataset
        # Add all OpenAlex columns
        openalex_cols = [
            "num_authors", "max_h_index", "avg_h_index", "min_h_index", 
            "first_author_h_index", "last_author_h_index", "max_citations", 
            "avg_citations", "total_author_citations", "max_career_stage", 
            "avg_career_stage", "has_early_career_author", "has_senior_author", 
            "author_diversity_score", "has_top_author",
            "author_affiliations", "author_ids", "author_names",
            "author_orcids", "author_h_indices", "author_i10_indices",
            "author_citation_counts", "author_paper_counts", "author_two_year_citedness",
            "author_years_active", "author_first_pub_years", "author_last_pub_years",
            "author_career_stages", "author_recent_productivity",
            "author_avg_citations_per_paper", "author_max_paper_citations"
        ]
        
        columns_to_keep = ["title", "abstract", "categories", "primary_category", "published", "arxiv_id", "url", "novelty_score", "max_similarity", "avg_similarity", "is_accepted", "acceptance_details", "authors"] + openalex_cols
        
        # Ensure ranked_new_df has all columns (fill missing with None if needed)
        for col in columns_to_keep:
            if col not in ranked_new_df.columns:
                ranked_new_df[col] = None
        
        # Filter both dfs to keep columns
        ranked_new_df_clean = ranked_new_df[columns_to_keep].copy()
        
        # Ensure existing_df has all columns
        for col in columns_to_keep:
            if col not in existing_df.columns:
                existing_df[col] = None
        existing_df_clean = existing_df[columns_to_keep].copy()
        
        # Concatenate
        final_df = pd.concat([ranked_new_df_clean, existing_df_clean], ignore_index=True)
    else:
        openalex_cols = [
            "num_authors", "max_h_index", "avg_h_index", "min_h_index", 
            "first_author_h_index", "last_author_h_index", "max_citations", 
            "avg_citations", "total_author_citations", "max_career_stage", 
            "avg_career_stage", "has_early_career_author", "has_senior_author", 
            "author_diversity_score", "has_top_author",
            "author_affiliations", "author_ids", "author_names"
        ]
        columns_to_keep = ["title", "abstract", "categories", "primary_category", "published", "arxiv_id", "url", "novelty_score", "max_similarity", "avg_similarity", "is_accepted", "acceptance_details", "authors"] + openalex_cols
        for col in columns_to_keep:
            if col not in ranked_new_df.columns:
                ranked_new_df[col] = None
        final_df = ranked_new_df[columns_to_keep].copy()
        
    # Sort final df by published date (descending)
    final_df["published"] = pd.to_datetime(final_df["published"])
    final_df = final_df.sort_values("published", ascending=False)
    # Convert back to string for HF
    final_df["published"] = final_df["published"].astype(str)
    
    # Remove any potential duplicate rows (just in case)
    final_df = final_df.drop_duplicates(subset=["arxiv_id"])
    
    # Remove artifact columns like __index_level_0__ if they exist
    if "__index_level_0__" in final_df.columns:
        final_df = final_df.drop(columns=["__index_level_0__"])
        
    # Reset index to avoid __index_level_0__ being created
    final_df = final_df.reset_index(drop=True)

    # Save results locally
    output_file = f"novelty_ranking_updated.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\nFull updated results saved to {output_file}")
    
    # 8. Push to Hugging Face
    # 8. Push to Hugging Face
    if not args.no_push:
        print("Pushing to Hugging Face...")
        try:
            from datasets import Dataset
            ds = Dataset.from_pandas(final_df)
            ds.push_to_hub("JasonYan777/novelty-ranked-preprints")
            print("Successfully pushed to JasonYan777/novelty-ranked-preprints")
            
        except Exception as e:
            print(f"Error pushing to Hugging Face: {e}")
    else:
        print("Skipping push to Hugging Face (--no-push specified).")

    # 9. Update Reference Dataset
    update_reference_dataset(ranked_new_df)

if __name__ == "__main__":
    main()
