
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
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from adapters import AutoAdapterModel

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
    MODEL_PATH = "models/siamese_scibert_multimodal/best_siamese_model.pth"

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
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Scalar output
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

def fetch_recent_papers(days=30, max_results=None):
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
            "Acceptance Details": "acceptance_details"
        })
        
        # Ensure required columns exist
        required_cols = ["title", "abstract", "categories", "primary_category", "published", "arxiv_id", "url"]
        for col in required_cols:
            if col not in df.columns:
                print(f"Missing column: {col}")
                return pd.DataFrame()
        
        # Keep new columns if they exist
        cols_to_keep = required_cols
        if "is_accepted" in df.columns:
            cols_to_keep.append("is_accepted")
        if "acceptance_details" in df.columns:
            cols_to_keep.append("acceptance_details")
            
        return df[cols_to_keep]
        
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
    """Compute max_similarity and avg_similarity against reference dataset."""
    print("Computing similarity features...")
    
    # Filter reference df for valid embeddings
    reference_df = reference_df[reference_df["proximity_embedding"].notna()]
    valid_mask = reference_df["proximity_embedding"].apply(lambda x: len(x) == 768 if isinstance(x, (list, np.ndarray)) else False)
    reference_df = reference_df[valid_mask]
    
    if len(reference_df) == 0:
        print("Error: No valid reference embeddings found.")
        return target_df

    try:
        ref_embs = np.stack(reference_df["proximity_embedding"].values).astype('float32')
        target_embs = np.stack(target_df["proximity_embedding"].values).astype('float32')
        
        print(f"Computing cosine similarity between {target_embs.shape} and {ref_embs.shape}...")
        
        # Compute cosine similarity matrix [n_targets, n_refs]
        sim_matrix = cosine_similarity(target_embs, ref_embs)
        
        max_sims = []
        avg_sims = []
        
        # For each target paper, find top 10 similar reference papers
        for i in range(len(target_df)):
            sims = sim_matrix[i]
            # Sort descending
            top_k_indices = np.argsort(sims)[-10:][::-1]
            top_k_sims = sims[top_k_indices]
            
            max_sims.append(float(top_k_sims[0]))
            avg_sims.append(float(np.mean(top_k_sims)))
            
        target_df["max_similarity"] = max_sims
        target_df["avg_similarity"] = avg_sims
        
    except Exception as e:
        print(f"Error in similarity computation: {e}")
        target_df["max_similarity"] = 0.0
        target_df["avg_similarity"] = 0.0
        
    return target_df

def rank_papers(df, model, tokenizer, device, config):
    """Rank papers using the Siamese model."""
    print("Ranking papers...")
    model.eval()
    
    scores = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        # Prepare inputs
        text = f"{row['title']} [SEP] {row['abstract']} [SEP] {row['categories']}"
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
    parser.add_argument("--max_results", type=int, default=1000, help="Max papers to fetch")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset instead of incremental update")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Fetch Papers (Daily Batch)
    new_papers_df = fetch_recent_papers(days=args.days, max_results=args.max_results)
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

    # 3. Generate Embeddings for NEW papers only
    unique_new_papers_df = generate_specter_embeddings(unique_new_papers_df, device)
    
    # 4. Load Reference Dataset (Training Data)
    print("Loading reference dataset for similarity...")
    dataset_name = "JasonYan777/novelty-rank-with-similarities"
    hf_ds = load_dataset(dataset_name, split="train")
    ref_df = hf_ds.to_pandas()
    
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
            
        # Select columns to keep for final dataset
        columns_to_keep = ["title", "abstract", "categories", "primary_category", "published", "arxiv_id", "url", "novelty_score", "max_similarity", "avg_similarity", "is_accepted", "acceptance_details"]
        
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
        columns_to_keep = ["title", "abstract", "categories", "primary_category", "published", "arxiv_id", "url", "novelty_score", "max_similarity", "avg_similarity", "is_accepted", "acceptance_details"]
        for col in columns_to_keep:
            if col not in ranked_new_df.columns:
                ranked_new_df[col] = None
        final_df = ranked_new_df[columns_to_keep].copy()
        
    # Sort final df by published date (descending) and then score? Or just score?
    # Usually users want recent stuff. Let's sort by published date desc.
    final_df["published"] = pd.to_datetime(final_df["published"])
    final_df = final_df.sort_values("published", ascending=False)
    # Convert back to string for HF
    final_df["published"] = final_df["published"].astype(str)

    # Save results locally
    output_file = f"novelty_ranking_updated.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\nFull updated results saved to {output_file}")
    
    # 8. Push to Hugging Face
    print("Pushing to Hugging Face...")
    try:
        from datasets import Dataset
        ds = Dataset.from_pandas(final_df)
        ds.push_to_hub("JasonYan777/novelty-ranked-preprints")
        print("Successfully pushed to JasonYan777/novelty-ranked-preprints")
        
    except Exception as e:
        print(f"Error pushing to Hugging Face: {e}")

if __name__ == "__main__":
    main()
