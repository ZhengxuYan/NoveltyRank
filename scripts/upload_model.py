import os
from huggingface_hub import HfApi, create_repo

# Configuration
MODEL_PATH = "models/siamese_scibert_multimodal/best_siamese_model_v3.pth"
REPO_ID = "JasonYan777/siamese_scibert_multimodal"

def upload_model():
    print(f"Uploading {MODEL_PATH} to {REPO_ID}...")
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True)
        print(f"Repository {REPO_ID} is ready.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Upload file
    try:
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo="best_siamese_model_v3.pth",
            repo_id=REPO_ID,
            repo_type="model"
        )
        print("Upload successful!")
    except Exception as e:
        print(f"Error uploading file: {e}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
    else:
        upload_model()
