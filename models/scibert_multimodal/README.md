# SciBERT Multimodal Acceptance Classifier

Binary classifier for paper acceptance prediction.

## Inputs
Text concatenated as: Title [SEP] Abstract [SEP] Categories, tokenized by `allenai/scibert_scivocab_uncased`, max length 512
Optional features:
- SPECTER2 classification embedding 768
- SPECTER2 proximity embedding 768
- Similarity features max_similarity and avg_similarity

## Architecture
SciBERT CLS 768
Concatenate optional features to get 2306 total when all are on
MLP head 2306 -> 512 -> 128 -> 2

## Files
- model.safetensors
- modeling_multimodal_scibert.py
- tokenizer files saved by AutoTokenizer.save_pretrained
- config.json and logs if present

## Minimal usage

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer
import importlib.util, sys

repo_id = "JasonYan777/scibert-multimodal-novelty"
tok = AutoTokenizer.from_pretrained(repo_id)
ckpt = hf_hub_download(repo_id, "model.safetensors")
code = hf_hub_download(repo_id, "modeling_multimodal_scibert.py")

spec = importlib.util.spec_from_file_location("modeling_multimodal_scibert", code)
mod = importlib.util.module_from_spec(spec)
sys.modules["modeling_multimodal_scibert"] = mod
spec.loader.exec_module(mod)
Model = mod.MultiModalSciBERT

model = Model(
    scibert_model="allenai/scibert_scivocab_uncased",
    use_classification_emb=True,
    use_proximity_emb=True,
    use_similarity_features=True,
)
state = load_file(ckpt)
model.load_state_dict(state)
model.eval()
