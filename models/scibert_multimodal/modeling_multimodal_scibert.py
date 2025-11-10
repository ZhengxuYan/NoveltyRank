
import torch
import torch.nn as nn
from transformers import AutoModel

class MultiModalSciBERT(nn.Module):
    def __init__(self, scibert_model="allenai/scibert_scivocab_uncased",
                 use_classification_emb=True, use_proximity_emb=True, use_similarity_features=True):
        super().__init__()
        self.USE_CLASSIFICATION_EMB = use_classification_emb
        self.USE_PROXIMITY_EMB = use_proximity_emb
        self.USE_SIMILARITY_FEATURES = use_similarity_features

        self.scibert = AutoModel.from_pretrained(scibert_model)
        hidden_size = self.scibert.config.hidden_size

        total_dim = hidden_size
        if self.USE_CLASSIFICATION_EMB:
            total_dim += 768
        if self.USE_PROXIMITY_EMB:
            total_dim += 768
        if self.USE_SIMILARITY_FEATURES:
            total_dim += 2

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, input_ids, attention_mask,
                classification_emb=None, proximity_emb=None, similarity_features=None):
        outputs = self.scibert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        feats = [cls_output]
        if self.USE_CLASSIFICATION_EMB and classification_emb is not None:
            feats.append(classification_emb)
        if self.USE_PROXIMITY_EMB and proximity_emb is not None:
            feats.append(proximity_emb)
        if self.USE_SIMILARITY_FEATURES and similarity_features is not None:
            feats.append(similarity_features)
        combined = torch.cat(feats, dim=1)
        return self.classifier(combined)
