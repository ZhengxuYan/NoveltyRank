from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel
from adapters import AutoAdapterModel
import torch
from tqdm import tqdm
import os
import pandas as pd
import argparse


def _get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple M2 GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def _cleanup_memory(device, model, tokenizer):
    del model
    del tokenizer
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def generate_specter2_embeddings(
    data, group, adapter_source, adapter_name, batch_size=8
):
    print(f"Generating {adapter_name} embeddings for {group} split")

    print("Loading SPECTER2 base model...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")

    model.load_adapter(
        adapter_source, source="hf", load_as=adapter_name, set_active=True
    )

    print(f"Active adapters: {model.active_adapters}")

    device = _get_device()
    model.to(device)
    
    model.eval()

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

    _cleanup_memory(device, model, tokenizer)

    return embeddings


def generate_scibert_embeddings(data, group, batch_size=8):
    print(f"Generating SciBERT embeddings for {group} split")

    print("Loading SciBERT model...")
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

    device = _get_device()
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
            # Use [CLS] token embedding as sentence representation
            batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()

        for embedding in batch_embeddings:
            embeddings.append(embedding)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {embeddings[0].shape[0]}")

    _cleanup_memory(device, model, tokenizer)

    return embeddings


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SPECTER2 and SciBERT embeddings for dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="JasonYan777/novelty-ranking-dataset",
        help="Source HuggingFace dataset name",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        default="JasonYan777/novelty-rank-embeddings",
        help="Output HuggingFace dataset name",
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
    print(f"Source Dataset: {args.dataset}")
    ds = load_dataset(args.dataset)
            
    print("\nProcessing dataset splits")
    
    # Process both train and test splits
    split_datasets = {}
    
    for split_name in ['train', 'test']:            
        print("\n" + "=" * 80)
        print(f"PROCESSING {split_name.upper()} SPLIT ({len(ds[split_name])} samples)")
        print("=" * 80)

        # Generate classification embeddings
        classification_embeddings = generate_specter2_embeddings(
            data=ds[split_name],
            group=split_name,
            adapter_source="allenai/specter2_classification",
            adapter_name="classification",
            batch_size=args.batch_size,
        )

        # Generate proximity embeddings
        proximity_embeddings = generate_specter2_embeddings(
            data=ds[split_name],
            group=split_name,
            adapter_source="allenai/specter2",
            adapter_name="proximity",
            batch_size=args.batch_size,
        )

        # Generate SciBERT embeddings
        scibert_embeddings = generate_scibert_embeddings(
            data=ds[split_name],
            group=split_name,
            batch_size=args.batch_size,
        )

        print(f"\nAdding embedding columns to {split_name} dataset...")
        data_dict = ds[split_name].to_dict()
        data_dict["classification_embedding"] = classification_embeddings
        data_dict["proximity_embedding"] = proximity_embeddings
        data_dict["scibert_embedding"] = scibert_embeddings

        split_datasets[split_name] = Dataset.from_dict(data_dict)

        print(f"Added embedding columns to {split_name}")
        print(f"Classification embedding dimension: {classification_embeddings[0].shape[0]}")
        print(f"Proximity embedding dimension: {proximity_embeddings[0].shape[0]}")
        print(f"SciBERT embedding dimension: {scibert_embeddings[0].shape[0]}")

    # Create final dataset with all processed splits
    final_dataset = DatasetDict(split_datasets)

    if args.save_local:
        os.makedirs("output", exist_ok=True)
        for split_name in split_datasets.keys():
            output_file = f"output/{split_name}_with_embeddings.pkl"
            print(f"\nSaving {split_name} to local file: {output_file}")
            df = split_datasets[split_name].to_pandas()
            df.to_pickle(output_file)
            print(f"Saved {len(df)} {split_name} samples locally")

    if not args.no_push:
        print("\n" + "=" * 80)
        print("Pushing dataset with embeddings to HuggingFace...")
        print("=" * 80)
        print(f"Output Repository: {args.output_dataset}")
        print("Note: This will UPDATE the dataset if it exists, or CREATE it if it doesn't")

        final_dataset.push_to_hub(args.output_dataset, private=False)

        print("Dataset pushed successfully!")
        print(f"View at: https://huggingface.co/datasets/{args.output_dataset}")

    print("\n" + "=" * 80)
    print("ALL PROCESSING COMPLETE!")
    print("=" * 80)
    for split_name in split_datasets.keys():
        print(f"  {split_name}: {len(split_datasets[split_name])} samples")
    print("=" * 80)