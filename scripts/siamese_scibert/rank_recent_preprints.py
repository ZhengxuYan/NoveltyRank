
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

def fetch_recent_papers(days=3, max_results=100):
    """Fetch recent papers from arXiv."""
    print(f"Fetching papers from the last {days} days...")
    
    # Construct query for AI/ML categories
    # cs.LG, cs.AI, cs.CV, cs.CL, cs.RO, cs.CR
    query = "cat:cs.LG OR cat:cs.AI OR cat:cs.CV OR cat:cs.CL OR cat:cs.RO OR cat:cs.CR"
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    
    for result in client.results(search):
        if result.published < cutoff_date:
            break
            
        papers.append({
            "title": result.title,
            "abstract": result.summary,
            "categories": ", ".join(result.categories),
            "primary_category": result.primary_category,
            "published": result.published,
            "arxiv_id": result.entry_id.split("/")[-1],
            "url": result.entry_id
        })
        
    print(f"Found {len(papers)} papers.")
    return pd.DataFrame(papers)

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
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Fetch Papers
    new_papers_df = fetch_recent_papers(days=args.days, max_results=args.max_results)
    if len(new_papers_df) == 0:
        print("No papers found.")
        return

    # 2. Generate Embeddings
    new_papers_df = generate_specter_embeddings(new_papers_df, device)
    
    # 3. Load Reference Dataset
    print("Loading reference dataset for similarity...")
    dataset_name = "JasonYan777/novelty-rank-with-similarities"
    hf_ds = load_dataset(dataset_name, split="train")
    ref_df = hf_ds.to_pandas()
    
    # Rename columns to match
    ref_df = ref_df.rename(columns={
        "classification_embedding": "classification_embedding", # already lowercase in HF? check
        "proximity_embedding": "proximity_embedding"
    })
    
    # Ensure embeddings are numpy arrays
    # In HF dataset they might be lists
    # We need to check one
    if isinstance(ref_df.iloc[0]["proximity_embedding"], list) or isinstance(ref_df.iloc[0]["proximity_embedding"], np.ndarray):
         pass # good
    else:
        # If it's something else, we might need to parse. But usually it's list or array.
        pass

    # 4. Compute Similarity
    new_papers_df = compute_similarity_features(new_papers_df, ref_df)
    
    # 5. Load Model and Rank
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
    
    ranked_df = rank_papers(new_papers_df, model, tokenizer, device, Config)
    
    # 6. Display Results
    print("\n" + "="*80)
    print(f"Top 10 Most Novel Papers (Last {args.days} Days)")
    print("="*80)
    
    for i, (idx, row) in enumerate(ranked_df.head(10).iterrows()):
        print(f"{i+1}. {row['title']}")
        print(f"   Score: {row['novelty_score']:.4f} | Categories: {row['categories']}")
        print(f"   URL: {row['url']}")
        print("-" * 80)
        
    # Save results locally
    output_file = f"novelty_ranking_{args.days}days.csv"
    ranked_df.to_csv(output_file, index=False)
    print(f"\nFull results saved to {output_file}")
    
    # 7. Push to Hugging Face
    print("Pushing to Hugging Face...")
    try:
        # Convert list columns to string or keep as list? HF Dataset handles lists well.
        # But embeddings might be large. We might want to drop them to save space if only for display?
        # The user wants to "deploy it for public to view", so we probably don't need embeddings in the deployed dataset, just scores and metadata.
        # However, keeping them allows for future updates/re-ranking without re-embedding.
        # Let's keep them for now, or drop if it's too big.
        
        # We should probably convert the pandas df to HF Dataset
        from datasets import Dataset
        
        # Drop embeddings to keep dataset light for the app?
        # The app only needs Title, Abstract, Categories, URL, Score, Date.
        # Let's keep a "light" version for the app.
        columns_to_keep = ["title", "abstract", "categories", "primary_category", "published", "arxiv_id", "url", "novelty_score", "max_similarity", "avg_similarity"]
        push_df = ranked_df[columns_to_keep].copy()
        
        # Ensure published is string or timestamp
        push_df["published"] = push_df["published"].astype(str)
        
        ds = Dataset.from_pandas(push_df)
        ds.push_to_hub("JasonYan777/novelty-ranked-preprints")
        print("Successfully pushed to JasonYan777/novelty-ranked-preprints")
        
    except Exception as e:
        print(f"Error pushing to Hugging Face: {e}")

if __name__ == "__main__":
    main()
