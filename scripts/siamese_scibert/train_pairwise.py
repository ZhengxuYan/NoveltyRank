
import os
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# ======================== Configuration ========================

class Config:
    # Paths
    MODEL_SAVE_DIR = "siamese_scibert_multimodal"
    
    # Model
    SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
    MAX_LEN = 512
    
    # Training
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 1e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.1
    
    # Features
    USE_CLASSIFICATION_EMB = True
    USE_PROXIMITY_EMB = True
    USE_SIMILARITY_FEATURES = True
    
    # Other
    RANDOM_SEED = 42
    GRADIENT_ACCUMULATION_STEPS = 2
    MAX_GRAD_NORM = 1.0

# ======================== Device Setup ========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

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

# ======================== Dataset ========================

class PairwiseNoveltyDataset(Dataset):
    """
    Dataset that yields pairs of papers (Paper A, Paper B) from the SAME category.
    Target is 1 if Paper A is more novel than Paper B, else 0.
    In this implementation, we ALWAYS yield (Novel, Not Novel) so target is always 1,
    but the model should learn to score Novel > Not Novel.
    """
    def __init__(self, dataframe, tokenizer, max_len, config, is_train=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.config = config
        self.is_train = is_train
        
        # Group by category
        self.category_map = {}
        # Filter for valid labels and features
        df = dataframe.copy()
        if config.USE_SIMILARITY_FEATURES:
            df = df[df["max_similarity"].notna() & df["avg_similarity"].notna()]
        
        for idx, row in df.iterrows():
            cat = row["Primary Category"]
            label = row["label"]
            
            if cat not in self.category_map:
                self.category_map[cat] = {"novel": [], "not_novel": []}
            
            if label == 1:
                self.category_map[cat]["novel"].append(row)
            elif label == 0:
                self.category_map[cat]["not_novel"].append(row)
        
        # Pre-generate pairs for stability (especially for validation/test)
        self.pairs = self._generate_pairs()
        
    def _generate_pairs(self):
        pairs = []
        # Ratio of Not-Novel papers to pair with each Novel paper
        # For training, we want more pairs to avoid overfitting.
        # For testing, we can keep it 1:1 or 1:all. Let's do 1:5 for training.
        n_negatives = 5 if self.is_train else 1
        
        for cat, papers in self.category_map.items():
            novel_papers = papers["novel"]
            not_novel_papers = papers["not_novel"]
            
            if not novel_papers or not not_novel_papers:
                continue
                
            # Strategy: Ensure we use ALL data + Augmentation
            if self.is_train:
                # Training: Use every not-novel paper at least once, 
                # AND ensure every novel paper is used at least 5 times.
                target_len = max(len(novel_papers) * 5, len(not_novel_papers))
            else:
                # Testing: Use max of both to test against all available data (comparable to previous run)
                target_len = max(len(novel_papers), len(not_novel_papers))

            # Shuffle for randomness
            if self.is_train:
                random.shuffle(novel_papers)
                random.shuffle(not_novel_papers)

            for i in range(target_len):
                p_novel = novel_papers[i % len(novel_papers)]
                p_not_novel = not_novel_papers[i % len(not_novel_papers)]
                pairs.append((p_novel, p_not_novel))
                
        if self.is_train:
            random.shuffle(pairs)
            
        return pairs

    def __len__(self):
        return len(self.pairs)


    def _process_paper(self, row):
        # Text
        title = str(row["Title"])
        abstract = str(row["Abstract"])
        # Clean abstract
        abstract = clean_text(abstract)
        categories = str(row["Categories"])
        text = f"{title} [SEP] {abstract} [SEP] {categories}"
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        
        # Embeddings
        if self.config.USE_CLASSIFICATION_EMB:
            emb = row["Classification_embedding"]
            if isinstance(emb, list): emb = np.array(emb)
            item["classification_emb"] = torch.tensor(emb, dtype=torch.float)
            
        if self.config.USE_PROXIMITY_EMB:
            emb = row["Proximity_embedding"]
            if isinstance(emb, list): emb = np.array(emb)
            item["proximity_emb"] = torch.tensor(emb, dtype=torch.float)
            
        # Similarity
        if self.config.USE_SIMILARITY_FEATURES:
            max_sim = row["max_similarity"] if pd.notna(row["max_similarity"]) else 0.0
            avg_sim = row["avg_similarity"] if pd.notna(row["avg_similarity"]) else 0.0
            item["similarity_features"] = torch.tensor([max_sim, avg_sim], dtype=torch.float)
            
        return item

    def __getitem__(self, index):
        row_a, row_b = self.pairs[index]
        
        feat_a = self._process_paper(row_a)
        feat_b = self._process_paper(row_b)
        
        return {
            "paper_a": feat_a,
            "paper_b": feat_b,
            "label": torch.tensor(1.0, dtype=torch.float) # A > B
        }

# ======================== Model ========================

class SiameseSciBERT(nn.Module):
    def __init__(self, config):
        super(SiameseSciBERT, self).__init__()
        self.config = config
        
        self.scibert = AutoModel.from_pretrained(config.SCIBERT_MODEL)
        self.hidden_size = self.scibert.config.hidden_size
        
        # Freeze embeddings and first 8 layers
        for param in self.scibert.embeddings.parameters():
            param.requires_grad = False
            
        for i in range(8):
            for param in self.scibert.encoder.layer[i].parameters():
                param.requires_grad = False
                
        # Verify trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")
        
        total_dim = self.hidden_size
        if config.USE_CLASSIFICATION_EMB: total_dim += 768
        if config.USE_PROXIMITY_EMB: total_dim += 768
        if config.USE_SIMILARITY_FEATURES: total_dim += 2
        
        # Scoring Head (Outputs a single scalar)
        # Increased Dropout to 0.5 for regularization
        self.score_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
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

    def forward(self, batch_a, batch_b):
        # Unpack batch_a
        score_a = self.forward_one(
            batch_a["input_ids"], 
            batch_a["attention_mask"],
            batch_a.get("classification_emb"),
            batch_a.get("proximity_emb"),
            batch_a.get("similarity_features")
        )
        
        # Unpack batch_b
        score_b = self.forward_one(
            batch_b["input_ids"], 
            batch_b["attention_mask"],
            batch_b.get("classification_emb"),
            batch_b.get("proximity_emb"),
            batch_b.get("similarity_features")
        )
        
        return score_a, score_b

# ======================== Training ========================

class EarlyStopping:
    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch_a = {k: v.to(device) for k, v in batch["paper_a"].items()}
        batch_b = {k: v.to(device) for k, v in batch["paper_b"].items()}
        targets = batch["label"].to(device) # All 1s
        
        score_a, score_b = model(batch_a, batch_b)
        
        # RankNet Loss: maximize log sigmoid(s_i - s_j)
        # BCEWithLogitsLoss(x, 1) = -log(sigmoid(x))
        # We want score_a > score_b, so diff = score_a - score_b should be high
        diff = score_a - score_b
        loss = loss_fn(diff.view(-1), targets)
        
        loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        
        # Accuracy: how often is score_a > score_b?
        preds = (score_a > score_b).float()
        correct += (preds.view(-1) == targets).sum().item()
        total += targets.size(0)
        
        progress_bar.set_postfix({"loss": f"{total_loss/(step+1):.4f}", "acc": f"{correct/total:.4f}"})
        
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_a = {k: v.to(device) for k, v in batch["paper_a"].items()}
            batch_b = {k: v.to(device) for k, v in batch["paper_b"].items()}
            
            score_a, score_b = model(batch_a, batch_b)
            
            # We expect score_a > score_b
            preds = (score_a > score_b)
            correct += preds.sum().item()
            total += preds.size(0)
            
    return correct / total

# ======================== Main ========================

def main():
    set_seed(Config.RANDOM_SEED)
    device = get_device()
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    dataset_name = "JasonYan777/novelty-rank-with-similarities"
    hf_ds = load_dataset(dataset_name)
    
    train_df = hf_ds["train"].to_pandas()
    test_df = hf_ds["test"].to_pandas()
    
    # Rename columns to match expected format
    rename_map = {
        "classification_embedding": "Classification_embedding",
        "proximity_embedding": "Proximity_embedding"
    }
    train_df = train_df.rename(columns=rename_map)
    test_df = test_df.rename(columns=rename_map)
    
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.SCIBERT_MODEL)
    
    train_ds = PairwiseNoveltyDataset(train_df, tokenizer, Config.MAX_LEN, Config, is_train=True)
    test_ds = PairwiseNoveltyDataset(test_df, tokenizer, Config.MAX_LEN, Config, is_train=False)
    
    print(f"Generated {len(train_ds)} training pairs")
    print(f"Generated {len(test_ds)} testing pairs")
    
    train_loader = DataLoader(train_ds, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = SiameseSciBERT(Config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    total_steps = len(train_loader) * Config.EPOCHS // Config.GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    best_acc = 0.0
    
    early_stopping = EarlyStopping(patience=2, min_delta=0.001)
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, Config)
        val_acc = evaluate(model, test_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(Config.MODEL_SAVE_DIR, "best_siamese_model.pth"))
            print(f"Saved best model with Acc: {best_acc:.4f}")
            
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    print("\nTraining Complete!")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
