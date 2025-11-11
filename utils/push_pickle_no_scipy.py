import argparse
import os
import sys
import subprocess
from pathlib import Path


def convert_pickle_to_parquet(pickle_file, parquet_file):
    """Convert pickle to parquet in a separate process to handle numpy version"""
    print("Converting pickle to parquet...")
    
    conversion_script = f"""
import pandas as pd
import sys

try:
    # Try loading with numpy 2.x first
    import numpy as np
    print(f"Using numpy {{np.__version__}}")
    
    df = pd.read_pickle("{pickle_file}")
    print(f"Loaded {{len(df)}} rows, {{len(df.columns)}} columns")
    
    # Save as parquet (no numpy version dependency)
    df.to_parquet("{parquet_file}", index=False)
    print(f"Saved to {{'{parquet_file}'}}")
    
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    
    # Run in subprocess
    result = subprocess.run(
        [sys.executable, "-c", conversion_script],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Error converting pickle:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def push_parquet_to_hf(parquet_file, repo_name, private=False, split_name="train", hf_token=None):
    """Push parquet file to Hugging Face"""
    # Import here to avoid scipy issues during pickle conversion
    from datasets import Dataset
    from huggingface_hub import login
    import pandas as pd
    
    print("\n" + "=" * 80)
    print("PUSH DATASET TO HUGGING FACE")
    print("=" * 80)
    
    # Authenticate
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if hf_token:
        print("\nAuthenticating with Hugging Face...")
        login(token=hf_token)
        print("✓ Authentication successful!")
    else:
        print("\nNo HF token provided. You may be prompted to login.")
    
    # Load parquet
    print("\nLoading parquet file...")
    df = pd.read_parquet(parquet_file)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Convert to dataset
    print("\nConverting to Hugging Face Dataset...")
    dataset = Dataset.from_pandas(df, preserve_index=False)
    print(f"✓ Created dataset with {len(dataset)} examples")
    
    # Push to Hub
    print("\n" + "=" * 80)
    print("Pushing to Hugging Face Hub...")
    print("=" * 80)
    print(f"Repository: {repo_name}")
    print(f"Split: {split_name}")
    print(f"Privacy: {'Private' if private else 'Public'}")
    
    try:
        dataset.push_to_hub(repo_name, split=split_name, private=private, token=hf_token)
        
        print("\n" + "=" * 80)
        print("✅ SUCCESS!")
        print("=" * 80)
        print(f"\nDataset URL: https://huggingface.co/datasets/{repo_name}")
        print(f"\nLoad with:")
        print(f"  from datasets import load_dataset")
        print(f"  ds = load_dataset('{repo_name}')")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Push pickle to HF (numpy-safe)")
    parser.add_argument("--pickle-file", type=str, required=True, help="Path to pickle file")
    parser.add_argument("--repo", type=str, required=True, help="HF repo (username/dataset-name)")
    parser.add_argument("--split", type=str, default="train", help="Split name")
    parser.add_argument("--private", action="store_true", help="Make private")
    parser.add_argument("--hf-token", type=str, default=None, help="HF token")
    parser.add_argument("--keep-parquet", action="store_true", help="Keep temporary parquet file")
    
    args = parser.parse_args()
    
    pickle_path = Path(args.pickle_file)
    if not pickle_path.exists():
        print(f"❌ File not found: {args.pickle_file}")
        sys.exit(1)
    
    # Create temporary parquet file
    parquet_file = pickle_path.parent / f"{pickle_path.stem}_temp.parquet"
    
    print("=" * 80)
    print("STEP 1: Convert pickle to parquet")
    print("=" * 80)
    
    if not convert_pickle_to_parquet(str(pickle_path), str(parquet_file)):
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("STEP 2: Push to Hugging Face")
    print("=" * 80)
    
    success = push_parquet_to_hf(
        parquet_file=str(parquet_file),
        repo_name=args.repo,
        private=args.private,
        split_name=args.split,
        hf_token=args.hf_token,
    )
    
    # Clean up temporary file
    if not args.keep_parquet and parquet_file.exists():
        parquet_file.unlink()
        print(f"\n🧹 Cleaned up temporary file: {parquet_file.name}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

