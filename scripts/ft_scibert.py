"""
Fine-tune SciBERT for Binary Classification (Paper Acceptance Prediction)

This script combines:
1. Text features (Title, Abstract, Categories) processed by SciBERT
2. Pre-computed SPECTER2 embeddings (Classification + Proximity)
3. Similarity features (max_similarity, avg_similarity)
4. Aggregated embeddings from top 10 similar papers

Model Architecture:
- SciBERT encoder for text → 768-dim
- Concatenate with Classification_embedding (768-dim) + Proximity_embedding (768-dim)
- Add similarity features (2-dim)
- Add aggregated similar papers embedding (768-dim)
- Total: 768 + 768 + 768 + 2 + 768 = 3074-dim → Classification head
"""

import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from datasets import load_dataset
from tqdm import tqdm

warnings.simplefilter("ignore")


# ======================== Configuration ========================


class Config:
    # Data Source
    DATASET_NAME = "JasonYan777/novelty-rank-with-similarities"
    MODEL_SAVE_DIR = "models/scibert_multimodal"

    # Model
    SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
    MAX_LEN = 512

    # Training (Optimized for M4 Max MacBook Pro)
    # M4 Max has excellent unified memory and GPU - can handle larger batches
    TRAIN_BATCH_SIZE = 32  # Increased for M4 Max (can go even higher)
    VALID_BATCH_SIZE = 64  # Increased for faster evaluation
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01

    # Features
    USE_CLASSIFICATION_EMB = True
    USE_PROXIMITY_EMB = True
    USE_SIMILARITY_FEATURES = True
    USE_SIMILAR_PAPERS_EMB = True  # Aggregate embeddings from top 10 similar papers
    SIMILAR_PAPERS_AGGREGATION = "max"  # Options: "mean", "max"

    # Class Imbalance Handling
    USE_CLASS_WEIGHTS = True  # Weight loss by inverse class frequency
    USE_FOCAL_LOSS = False  # Alternative to class weights (set to True to use instead)
    FOCAL_LOSS_GAMMA = 2.0  # Focus more on hard examples
    OVERSAMPLE_MINORITY = False  # Oversample positive class (can slow training)

    # Other
    RANDOM_SEED = 42
    GRADIENT_ACCUMULATION_STEPS = (
        1  # M4 Max can handle larger batches, less accumulation needed
    )
    MAX_GRAD_NORM = 1.0


# ======================== Device Setup ========================


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get best available device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple M-series GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    return device


# ======================== Dataset ========================


class NoveltyDataset(Dataset):
    """Dataset for paper novelty classification with multi-modal features"""

    def __init__(self, dataframe, tokenizer, max_len, config):
        self.tokenizer = tokenizer
        self.data = dataframe.reset_index(drop=True)
        self.max_len = max_len
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Prepare text input: Title [SEP] Abstract [SEP] Categories
        title = str(row["Title"])
        abstract = str(row["Abstract"])
        categories = str(row["Categories"])
        text = f"{title} [SEP] {abstract} [SEP] {categories}"

        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Prepare feature dictionary
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(row["label"], dtype=torch.long),
        }

        # Add pre-computed embeddings
        if self.config.USE_CLASSIFICATION_EMB:
            classification_emb = row["Classification_embedding"]
            if isinstance(classification_emb, list):
                classification_emb = np.array(classification_emb)
            item["classification_emb"] = torch.tensor(
                classification_emb, dtype=torch.float
            )

        if self.config.USE_PROXIMITY_EMB:
            proximity_emb = row["Proximity_embedding"]
            if isinstance(proximity_emb, list):
                proximity_emb = np.array(proximity_emb)
            item["proximity_emb"] = torch.tensor(proximity_emb, dtype=torch.float)

        # Add similarity features
        if self.config.USE_SIMILARITY_FEATURES:
            # Handle None values for similarity features
            max_sim = row["max_similarity"] if pd.notna(row["max_similarity"]) else 0.0
            avg_sim = row["avg_similarity"] if pd.notna(row["avg_similarity"]) else 0.0
            item["similarity_features"] = torch.tensor(
                [max_sim, avg_sim], dtype=torch.float
            )

        # Add aggregated similar papers embeddings
        if self.config.USE_SIMILAR_PAPERS_EMB:
            top_10_similar = row.get("top_10_similar", [])

            if top_10_similar and len(top_10_similar) > 0:
                # Extract embeddings from similar papers
                similar_embs = []
                for similar_paper in top_10_similar:
                    emb = similar_paper.get("embedding", None)
                    if emb is not None:
                        if isinstance(emb, list):
                            emb = np.array(emb)
                        similar_embs.append(emb)

                if len(similar_embs) > 0:
                    # Stack and aggregate
                    similar_embs = np.stack(similar_embs)  # Shape: (N, 768)

                    if self.config.SIMILAR_PAPERS_AGGREGATION == "mean":
                        aggregated_emb = np.mean(similar_embs, axis=0)
                    elif self.config.SIMILAR_PAPERS_AGGREGATION == "max":
                        aggregated_emb = np.max(similar_embs, axis=0)
                    else:
                        # Default to mean if invalid option
                        aggregated_emb = np.mean(similar_embs, axis=0)

                    item["similar_papers_emb"] = torch.tensor(
                        aggregated_emb, dtype=torch.float
                    )
                else:
                    # No valid embeddings, use zeros
                    item["similar_papers_emb"] = torch.zeros(768, dtype=torch.float)
            else:
                # No similar papers, use zeros
                item["similar_papers_emb"] = torch.zeros(768, dtype=torch.float)

        return item


# ======================== Model ========================


class MultiModalSciBERT(nn.Module):
    """
    Multi-modal SciBERT model combining:
    1. Text encoding via SciBERT
    2. Pre-computed SPECTER2 embeddings
    3. Similarity features
    """

    def __init__(self, config):
        super(MultiModalSciBERT, self).__init__()
        self.config = config

        # SciBERT encoder
        self.scibert = AutoModel.from_pretrained(config.SCIBERT_MODEL)
        self.hidden_size = self.scibert.config.hidden_size  # 768

        # Calculate total feature dimension
        total_dim = self.hidden_size  # SciBERT CLS token

        if config.USE_CLASSIFICATION_EMB:
            total_dim += 768
        if config.USE_PROXIMITY_EMB:
            total_dim += 768
        if config.USE_SIMILARITY_FEATURES:
            total_dim += 2
        if config.USE_SIMILAR_PAPERS_EMB:
            total_dim += 768  # Aggregated embedding from similar papers

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),  # Binary classification
        )

        print(f"📊 Model architecture: {total_dim}-dim input → 512 → 128 → 2")

    def forward(
        self,
        input_ids,
        attention_mask,
        classification_emb=None,
        proximity_emb=None,
        similarity_features=None,
        similar_papers_emb=None,
    ):
        # Encode text with SciBERT
        outputs = self.scibert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Concatenate features
        features = [cls_output]

        if self.config.USE_CLASSIFICATION_EMB and classification_emb is not None:
            features.append(classification_emb)

        if self.config.USE_PROXIMITY_EMB and proximity_emb is not None:
            features.append(proximity_emb)

        if self.config.USE_SIMILARITY_FEATURES and similarity_features is not None:
            features.append(similarity_features)

        if self.config.USE_SIMILAR_PAPERS_EMB and similar_papers_emb is not None:
            features.append(similar_papers_emb)

        # Concatenate all features
        combined_features = torch.cat(features, dim=1)

        # Classification
        logits = self.classifier(combined_features)
        return logits


# ======================== Loss Functions ========================


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses more on hard examples and down-weights easy examples
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction="none")(
            inputs, targets
        )
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(labels, device):
    """
    Compute class weights inversely proportional to class frequencies
    """
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    return class_weights


# ======================== Training Functions ========================


def train_epoch(
    model, dataloader, optimizer, scheduler, device, config, epoch, loss_fn
):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")

    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Optional features
        classification_emb = batch.get("classification_emb", None)
        if classification_emb is not None:
            classification_emb = classification_emb.to(device)

        proximity_emb = batch.get("proximity_emb", None)
        if proximity_emb is not None:
            proximity_emb = proximity_emb.to(device)

        similarity_features = batch.get("similarity_features", None)
        if similarity_features is not None:
            similarity_features = similarity_features.to(device)

        similar_papers_emb = batch.get("similar_papers_emb", None)
        if similar_papers_emb is not None:
            similar_papers_emb = similar_papers_emb.to(device)

        # Forward pass
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            classification_emb=classification_emb,
            proximity_emb=proximity_emb,
            similarity_features=similarity_features,
            similar_papers_emb=similar_papers_emb,
        )

        # Compute loss (using provided loss function with class weights)
        loss = loss_fn(logits, labels)
        loss = loss / config.GRADIENT_ACCUMULATION_STEPS

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Track metrics
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{total_loss / (step + 1):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
        )

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, dataloader, device, config, loss_fn, split_name="Valid"):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    progress_bar = tqdm(dataloader, desc=f"{split_name}")

    with torch.no_grad():
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Optional features
            classification_emb = batch.get("classification_emb", None)
            if classification_emb is not None:
                classification_emb = classification_emb.to(device)

            proximity_emb = batch.get("proximity_emb", None)
            if proximity_emb is not None:
                proximity_emb = proximity_emb.to(device)

            similarity_features = batch.get("similarity_features", None)
            if similarity_features is not None:
                similarity_features = similarity_features.to(device)

            similar_papers_emb = batch.get("similar_papers_emb", None)
            if similar_papers_emb is not None:
                similar_papers_emb = similar_papers_emb.to(device)

            # Forward pass
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                classification_emb=classification_emb,
                proximity_emb=proximity_emb,
                similarity_features=similarity_features,
                similar_papers_emb=similar_papers_emb,
            )

            # Compute loss (using provided loss function)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class

    avg_loss = total_loss / len(dataloader)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Handle case where only one class is present
        auc_roc = 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def print_metrics(metrics, split_name="Validation"):
    """Pretty print evaluation metrics"""
    print(f"\n{'='*60}")
    print(f"{split_name} Metrics")
    print(f"{'='*60}")
    print(f"Loss:      {metrics['loss']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"{'='*60}")

    # Confusion matrix
    cm = confusion_matrix(metrics["labels"], metrics["predictions"])
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Neg    Pos")
    print(f"Actual  Neg  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"        Pos  {cm[1,0]:5d}  {cm[1,1]:5d}")
    print(f"{'='*60}\n")


# ======================== Main Training Loop ========================


def main(args):
    # Set seed
    set_seed(Config.RANDOM_SEED)

    # Get device
    device = get_device()

    # Update config from args
    if args.epochs:
        Config.EPOCHS = args.epochs
    if args.batch_size:
        Config.TRAIN_BATCH_SIZE = args.batch_size
    if args.learning_rate:
        Config.LEARNING_RATE = args.learning_rate
    if args.dataset:
        Config.DATASET_NAME = args.dataset

    print("\n" + "=" * 60)
    print("📚 Loading Data from HuggingFace...")
    print("=" * 60)
    print(f"Dataset: {Config.DATASET_NAME}")

    # Load dataset from HuggingFace
    dataset = load_dataset(Config.DATASET_NAME)

    # Convert to pandas DataFrames
    train_df = pd.DataFrame(dataset["train"])
    val_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])

    print(f"✓ Loaded train set: {len(train_df)} samples")
    print(f"✓ Loaded validation set: {len(val_df)} samples")
    print(f"✓ Loaded test set: {len(test_df)} samples")

    # Rename columns if needed (HuggingFace uses lowercase_with_underscores)
    column_mapping = {
        "classification_embedding": "Classification_embedding",
        "proximity_embedding": "Proximity_embedding",
    }

    for df in [train_df, val_df, test_df]:
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)

    # Filter out papers without similarity scores if using similarity features
    if Config.USE_SIMILARITY_FEATURES:
        print("\n🔍 Filtering papers without similarity scores...")

        # Count before filtering
        train_before = len(train_df)
        val_before = len(val_df)
        test_before = len(test_df)

        # Filter: keep only papers with non-null similarity scores
        train_df = train_df[
            train_df["max_similarity"].notna() & train_df["avg_similarity"].notna()
        ].reset_index(drop=True)

        val_df = val_df[
            val_df["max_similarity"].notna() & val_df["avg_similarity"].notna()
        ].reset_index(drop=True)

        test_df = test_df[
            test_df["max_similarity"].notna() & test_df["avg_similarity"].notna()
        ].reset_index(drop=True)

        # Report filtering results
        train_removed = train_before - len(train_df)
        val_removed = val_before - len(val_df)
        test_removed = test_before - len(test_df)

        print(f"  Train: Removed {train_removed} papers without similarity scores")
        print(f"  Validation: Removed {val_removed} papers without similarity scores")
        print(f"  Test: Removed {test_removed} papers without similarity scores")
        print(f"  ✓ Filtered train set: {len(train_df)} samples")
        print(f"  ✓ Filtered validation set: {len(val_df)} samples")
        print(f"  ✓ Filtered test set: {len(test_df)} samples")

    print(f"\n✓ Final train set: {len(train_df)} samples")
    print(f"✓ Final validation set: {len(val_df)} samples")
    print(f"✓ Final test set: {len(test_df)} samples")
    print(f"✓ Train label distribution: {train_df['label'].value_counts().to_dict()}")
    print(
        f"✓ Validation label distribution: {val_df['label'].value_counts().to_dict()}"
    )
    print(f"✓ Test label distribution: {test_df['label'].value_counts().to_dict()}")

    # Initialize tokenizer
    print("\n" + "=" * 60)
    print("🔧 Initializing Tokenizer and Model...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(Config.SCIBERT_MODEL)

    # Create datasets
    train_dataset = NoveltyDataset(train_df, tokenizer, Config.MAX_LEN, Config)
    val_dataset = NoveltyDataset(val_df, tokenizer, Config.MAX_LEN, Config)
    test_dataset = NoveltyDataset(test_df, tokenizer, Config.MAX_LEN, Config)

    # Handle class imbalance with weighted sampling (optional)
    if Config.OVERSAMPLE_MINORITY:
        print("\n⚖️  Setting up balanced sampling (oversampling minority class)...")
        # Compute sample weights (higher weight for minority class)
        train_labels = train_df["label"].values
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]
        sample_weights = torch.FloatTensor(sample_weights)

        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        shuffle = False  # Don't shuffle when using sampler
        print("  ✓ Using weighted sampling (minority class oversampled)")
    else:
        sampler = None
        shuffle = True

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,  # Parallel data loading for M4 Max
        pin_memory=True,  # Faster CPU->GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # Initialize model
    model = MultiModalSciBERT(Config)
    model.to(device)

    # Setup loss function with class imbalance handling
    print("\n" + "=" * 60)
    print("⚖️  Configuring Loss Function for Class Imbalance...")
    print("=" * 60)

    train_labels = train_df["label"].values
    class_counts = np.bincount(train_labels)
    pos_weight = class_counts[0] / class_counts[1]  # negative / positive ratio

    print("Class distribution:")
    print(
        f"  Negative (0): {class_counts[0]} samples ({class_counts[0]/len(train_labels)*100:.2f}%)"
    )
    print(
        f"  Positive (1): {class_counts[1]} samples ({class_counts[1]/len(train_labels)*100:.2f}%)"
    )
    print(f"  Imbalance ratio: {pos_weight:.2f}:1")

    if Config.USE_FOCAL_LOSS:
        # Focal Loss - focuses on hard examples
        class_weights = compute_class_weights(train_labels, device)
        loss_fn = FocalLoss(alpha=class_weights, gamma=Config.FOCAL_LOSS_GAMMA)
        print(f"\n✓ Using Focal Loss (gamma={Config.FOCAL_LOSS_GAMMA})")
        print(f"  Class weights: [1.00, {class_weights[1]/class_weights[0]:.2f}]")
    elif Config.USE_CLASS_WEIGHTS:
        # Standard CrossEntropy with class weights
        class_weights = compute_class_weights(train_labels, device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print("\n✓ Using Weighted CrossEntropy Loss")
        print("  Class weights: [1.00, {class_weights[1]/class_weights[0]:.2f}]")
        print(
            f"  (Positive class weighted {class_weights[1]/class_weights[0]:.2f}x more)"
        )
    else:
        # Standard CrossEntropy (no weighting)
        loss_fn = nn.CrossEntropyLoss()
        print("\n✓ Using Standard CrossEntropy Loss (no class weighting)")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    total_steps = (
        len(train_loader) * Config.EPOCHS // Config.GRADIENT_ACCUMULATION_STEPS
    )
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"✓ Total training steps: {total_steps}")
    print(f"✓ Warmup steps: {warmup_steps}")

    # Training loop
    print("\n" + "=" * 60)
    print("🚀 Starting Training...")
    print("=" * 60)

    best_val_f1 = 0
    history = []

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, Config, epoch, loss_fn
        )

        # Evaluate on validation set
        val_metrics = evaluate(
            model, val_loader, device, Config, loss_fn, split_name="Validation"
        )

        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}")

        # Save history
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_auc_roc": val_metrics["auc_roc"],
            }
        )

        # Save best model based on validation F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)

            model_path = os.path.join(Config.MODEL_SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"💾 Saved best model (Val F1: {best_val_f1:.4f})")

    # Final evaluation with best model
    print("\n" + "=" * 60)
    print("📊 Final Evaluation (Best Model)")
    print("=" * 60)

    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(Config.MODEL_SAVE_DIR, "best_model.pth"))
    )

    # Evaluate on validation set
    print("\n🔍 Evaluating on Validation Set...")
    val_metrics = evaluate(
        model, val_loader, device, Config, loss_fn, split_name="Validation"
    )
    print_metrics(val_metrics, split_name="Validation Set (Best Model)")

    # Evaluate on test set
    print("\n🔍 Evaluating on Test Set...")
    test_metrics = evaluate(
        model, test_loader, device, Config, loss_fn, split_name="Test"
    )
    print_metrics(test_metrics, split_name="Test Set (Best Model)")

    # Save results
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)

    # Save tokenizer
    tokenizer.save_pretrained(Config.MODEL_SAVE_DIR)

    # Save config
    config_dict = {
        "model_name": Config.SCIBERT_MODEL,
        "max_len": Config.MAX_LEN,
        "use_classification_emb": Config.USE_CLASSIFICATION_EMB,
        "use_proximity_emb": Config.USE_PROXIMITY_EMB,
        "use_similarity_features": Config.USE_SIMILARITY_FEATURES,
        "use_similar_papers_emb": Config.USE_SIMILAR_PAPERS_EMB,
        "similar_papers_aggregation": Config.SIMILAR_PAPERS_AGGREGATION,
        "use_class_weights": Config.USE_CLASS_WEIGHTS,
        "use_focal_loss": Config.USE_FOCAL_LOSS,
        "oversample_minority": Config.OVERSAMPLE_MINORITY,
        "best_val_f1": best_val_f1,
        "final_test_f1": test_metrics["f1"],
        "final_test_accuracy": test_metrics["accuracy"],
        "final_test_auc_roc": test_metrics["auc_roc"],
    }

    with open(os.path.join(Config.MODEL_SAVE_DIR, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(Config.MODEL_SAVE_DIR, "training_history.csv"), index=False
    )

    # Save test predictions
    results_df = pd.DataFrame(
        {
            "true_label": test_metrics["labels"],
            "predicted_label": test_metrics["predictions"],
            "probability_positive": test_metrics["probabilities"],
        }
    )
    results_df.to_csv(
        os.path.join(Config.MODEL_SAVE_DIR, "test_predictions.csv"), index=False
    )

    print(f"\n✅ Training complete! Model saved to: {Config.MODEL_SAVE_DIR}")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Final Test F1: {test_metrics['f1']:.4f}")
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final Test AUC-ROC: {test_metrics['auc_roc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune SciBERT for novelty classification"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Learning rate"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (default: JasonYan777/novelty-rank-with-similarities)",
    )

    args = parser.parse_args()
    main(args)
