from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch
from tqdm import tqdm
import os


def generate_embeddings_with_adapter(
    data, group, adapter_source, adapter_name, batch_size=8
):
    print(f"Generating {adapter_name} embeddings for {group} split")

    print("Loading SPECTER2 base model...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")

    print(f"Loading adapter: {adapter_source}")
    model.load_adapter(
        adapter_source, source="hf", load_as=adapter_name, set_active=True
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple M2 GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model.to(device)
    model.eval()

    print(f"Generating embeddings (batch_size={batch_size})...")
    embeddings = []

    for i in tqdm(range(0, len(data), batch_size), desc=f"Processing {group}"):
        batch_end = min(i + batch_size, len(data))

        batch_texts = []
        for idx in range(i, batch_end):
            item = data[idx]
            text = f"Title: {item['Title']} Abstract: {item['Abstract']}"
            batch_texts.append(text)

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)
            batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()

        for embedding in batch_embeddings:
            embeddings.append(embedding)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {embeddings[0].shape[0]}")

    del model
    del tokenizer

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return embeddings


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SPECTER2 embeddings for dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="JasonYan777/novelty-rank-dataset",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--save-local",
        action="store_true",
        help="Also save embeddings locally as pickle file",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push to HuggingFace (only save locally)",
    )
    args = parser.parse_args()

    print("Loading dataset from HuggingFace...")
    print(f"Dataset: {args.dataset}")
    ds = load_dataset(args.dataset)

    print("Dataset structure:")
    print(f"  Available splits: {list(ds.keys())}")
    print(f"  Number of samples: {len(ds['train'])}")
    print(f"  Columns: {ds['train'].column_names}")

    print("\nProcessing dataset")

    classification_embeddings = generate_embeddings_with_adapter(
        data=ds["train"],
        group="train",
        adapter_source="allenai/specter2_classification",
        adapter_name="classification",
        batch_size=args.batch_size,
    )

    proximity_embeddings = generate_embeddings_with_adapter(
        data=ds["train"],
        group="train",
        adapter_source="allenai/specter2",
        adapter_name="proximity",
        batch_size=args.batch_size,
    )

    print("\nAdding embedding columns to dataset...")

    data_dict = ds["train"].to_dict()
    data_dict["classification_embedding"] = classification_embeddings
    data_dict["proximity_embedding"] = proximity_embeddings

    dataset_with_embeddings = Dataset.from_dict(data_dict)

    print("Added embedding columns")
    print(f"New columns: {dataset_with_embeddings.column_names}")
    print(
        f"Classification embedding dimension: {classification_embeddings[0].shape[0]}"
    )
    print(f"Proximity embedding dimension: {proximity_embeddings[0].shape[0]}")

    if args.save_local:
        import pandas as pd

        os.makedirs("output", exist_ok=True)
        output_file = "output/dataset_with_embeddings.pkl"
        print(f"\nSaving to local file: {output_file}")
        df = dataset_with_embeddings.to_pandas()
        df.to_pickle(output_file)
        print(f"Saved {len(df)} samples locally")

    if not args.no_push:
        print("\nPushing dataset with embeddings to HuggingFace...")
        print(f"Repository: {args.dataset}")
        print("Note: This will update your dataset on HuggingFace Hub")

        dataset_dict = DatasetDict({"train": dataset_with_embeddings})

        dataset_dict.push_to_hub(args.dataset, private=False)

        print("Dataset pushed successfully!")
        print(f"View at: https://huggingface.co/datasets/{args.dataset}")

    print("\nProcessing complete!")
    print(f"Total samples processed: {len(dataset_with_embeddings)}")
    print(f"New dataset columns: {dataset_with_embeddings.column_names}")
