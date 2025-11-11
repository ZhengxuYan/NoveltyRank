"""
Script to push SciBERT Multimodal model to Hugging Face Hub.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from getpass import getpass

# Configuration
MODEL_DIR = Path("/Users/jasonyan/Desktop/CS 230/NoveltyRank/models/scibert_multimodal")
REPO_ID = "JasonYan777/scibert-multimodal-novelty"  # Change this to your username/repo


def push_model_to_hf():
    """Push the model to Hugging Face Hub."""

    print("=" * 80)
    print("PUSH SCIBERT MULTIMODAL MODEL TO HUGGING FACE")
    print("=" * 80)

    # Check if model directory exists
    if not MODEL_DIR.exists():
        print(f"\n❌ Error: Model directory not found at {MODEL_DIR}")
        return

    # List files to be uploaded
    files_to_upload = [
        "model.safetensors",
        "modeling_multimodal_scibert.py",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "README.md",
        "training_history.csv",
        "test_predictions.csv",
    ]

    print("\nFiles to upload:")
    for f in files_to_upload:
        file_path = MODEL_DIR / f
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {f} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {f} (not found)")

    # Get HF token
    print("\n" + "=" * 80)
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Hugging Face token not found in environment.")
        print(
            "Please enter your Hugging Face token (get it from https://huggingface.co/settings/tokens):"
        )
        token = getpass("Token: ")

    if not token:
        print("\n❌ Error: No token provided. Cannot upload to Hugging Face.")
        return

    print("\n✓ Token received")

    # Initialize API
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    print(f"\nCreating/accessing repository: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            token=token,
            private=False,
            exist_ok=True,
            repo_type="model",
        )
        print(f"✓ Repository ready: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"⚠ Warning during repo creation: {e}")
        print("Continuing with upload...")

    # Upload folder
    print("\nUploading model files to Hugging Face Hub...")
    print("This may take a few minutes depending on file sizes...")

    try:
        # Upload all files from the model directory
        api.upload_folder(
            folder_path=str(MODEL_DIR),
            repo_id=REPO_ID,
            token=token,
            ignore_patterns=[
                "*.pth",  # Skip PyTorch checkpoint (we have safetensors)
                "analysis/*",  # Skip analysis folder
                "hub_config.json",  # Will be auto-generated
                "*.pyc",
                "__pycache__/*",
            ],
        )

        print("\n" + "=" * 80)
        print("✅ SUCCESS! Model uploaded to Hugging Face Hub")
        print("=" * 80)
        print(f"\nModel URL: https://huggingface.co/{REPO_ID}")
        print(f"\nYou can now use your model with:")
        print(f"\n  from transformers import AutoTokenizer")
        print(f"  from huggingface_hub import hf_hub_download")
        print(f"  from safetensors.torch import load_file")
        print(f"  import importlib.util, sys")
        print(f"\n  repo_id = '{REPO_ID}'")
        print(f"  tok = AutoTokenizer.from_pretrained(repo_id)")
        print(f"  ckpt = hf_hub_download(repo_id, 'model.safetensors')")
        print(f"  code = hf_hub_download(repo_id, 'modeling_multimodal_scibert.py')")
        print(f"\n  # Load model class")
        print(
            f"  spec = importlib.util.spec_from_file_location('modeling_multimodal_scibert', code)"
        )
        print(f"  mod = importlib.util.module_from_spec(spec)")
        print(f"  sys.modules['modeling_multimodal_scibert'] = mod")
        print(f"  spec.loader.exec_module(mod)")
        print(f"  Model = mod.MultiModalSciBERT")
        print(f"\n  # Initialize and load weights")
        print(f"  model = Model(")
        print(f"      scibert_model='allenai/scibert_scivocab_uncased',")
        print(f"      use_classification_emb=True,")
        print(f"      use_proximity_emb=True,")
        print(f"      use_similarity_features=True,")
        print(f"  )")
        print(f"  state = load_file(ckpt)")
        print(f"  model.load_state_dict(state)")
        print(f"  model.eval()")

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR during upload")
        print("=" * 80)
        print(f"\n{str(e)}")
        print("\nTroubleshooting:")
        print("  1. Check your Hugging Face token is valid")
        print("  2. Ensure you have write access to the repository")
        print("  3. Check your internet connection")
        return


def main():
    print("\n📤 Starting Hugging Face upload process...\n")
    push_model_to_hf()


if __name__ == "__main__":
    main()
