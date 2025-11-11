"""
Simple script to create a Hugging Face repository.
Run this first to verify your token and create the repo.
"""

import os
from getpass import getpass
from huggingface_hub import HfApi, create_repo

# Configuration
REPO_ID = "JasonYan777/scibert-multimodal-novelty"


def create_repository():
    """Create a Hugging Face repository."""

    print("=" * 80)
    print("CREATE HUGGING FACE REPOSITORY")
    print("=" * 80)

    # Get HF token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\nHugging Face token not found in environment.")
        print(
            "Please enter your Hugging Face token (get it from https://huggingface.co/settings/tokens):"
        )
        print("Make sure it has WRITE permissions!")
        token = getpass("Token: ")

    if not token:
        print("\n❌ Error: No token provided.")
        return

    print("\n✓ Token received")

    # Test token by getting user info
    print("\nTesting token...")
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"✓ Token valid! Logged in as: {user_info['name']}")
    except Exception as e:
        print(f"\n❌ Token is invalid!")
        print(f"Error: {e}")
        print("\nPlease make sure:")
        print("  1. You copied the entire token (starts with 'hf_')")
        print("  2. The token has 'write' permissions")
        print("  3. The token hasn't expired")
        return

    # Create repository
    print(f"\nCreating repository: {REPO_ID}")
    try:
        repo_url = create_repo(
            repo_id=REPO_ID,
            token=token,
            private=False,
            exist_ok=True,
            repo_type="model",
        )
        print("\n" + "=" * 80)
        print("✅ SUCCESS! Repository created/verified")
        print("=" * 80)
        print(f"\nRepository URL: {repo_url}")
        print(f"\nYou can now run: python scripts/push_to_huggingface.py")

        # Save token for next script (optional)
        save_token = input(
            "\nWould you like to save the token to HF_TOKEN environment variable for this session? (y/n): "
        )
        if save_token.lower() == "y":
            print(f"\nRun this command in your terminal:")
            print(f'export HF_TOKEN="{token}"')

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR during repository creation")
        print("=" * 80)
        print(f"\n{e}")
        print("\nTroubleshooting:")
        print("  1. Check if the repository name is available")
        print("  2. Make sure you have permission to create repos under this username")
        print("  3. Verify your token has 'write' permissions")


if __name__ == "__main__":
    create_repository()
