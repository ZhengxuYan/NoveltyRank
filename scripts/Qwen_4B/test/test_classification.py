# __init__.py
"""
Initialize model samplers and handle prompt construction logic.
"""
import os
import sys
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)
from pathlib import Path
from typing import Optional

from models.Qwen_4B.sft_env import create_sft_example, clean_dataset
from models.Qwen_4B.data_access import (
    DEFAULT_CATEGORIES,
    category_to_token,
    normalize_category,
)
import asyncio
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

import re
import tinker
from dotenv import load_dotenv
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Initialize Tinker Service Client
load_dotenv()
service_client = tinker.ServiceClient()

CATEGORY_TOKEN_TO_NORMALIZED = {
    category_to_token(category): normalize_category(category)
    for category in DEFAULT_CATEGORIES
}
CATEGORY_TOKENS = list(CATEGORY_TOKEN_TO_NORMALIZED.keys())

# Define TinkerSampler class
class TinkerSampler():
    """A simple wrapper around Tinker ServiceClient to do sampling."""
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,  # tinker://..., obtained from Tinker training job
        temperature: float = 0.9,
        max_tokens=1024,
        top_p=1,
        top_k=-1,  # -1 means no limit
    ):
        tokenizer = get_tokenizer(model_name)
        renderer_name = get_recommended_renderer_name(model_name)
        # Read https://tinker-docs.thinkingmachines.ai/rendering to understand what renderer is
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
        self.sampling_params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=self.renderer.get_stop_sequences(),
        )
        self.sampling_client = service_client.create_sampling_client(
            model_path=model_path,
            base_model=model_name,
        )
        
    async def generate(self, messages: list[renderers.Message]) -> renderers.Message:
        prompt = self.renderer.build_generation_prompt(messages) #tokens is inside
        sampling_result = await self.sampling_client.sample_async(
            prompt=prompt,
            sampling_params=self.sampling_params,
            num_samples=1
        )
        response_tokens = sampling_result.sequences[0].tokens 

        raw_text_output = self.renderer.tokenizer.decode(response_tokens)
        response_message = {'role': 'assistant', 'content': raw_text_output}
        response_message['content'] = response_message['content'].strip()

        return response_message
    
async def process_one(data, sampler, include_similarity_report: bool):
    user_prompt, label = create_sft_example(
        data,
        include_similarity_report=include_similarity_report,
    )
    messages = [
        renderers.Message(role="user", content=user_prompt)
    ]
    prediction = await sampler.generate(messages)
    prediction = prediction['content']
    # Extract "0" or "1" from prediction using regex
    match = re.search(r'\b(0|1)\b', prediction)
    prediction = match[0] if match else ""
    return prediction, label


def _load_category_sft_split(category_token: str) -> Optional[Dataset]:
    path = Path("data_cache") / "categories" / category_token / "sft" / "test"
    if not path.exists():
        return None
    print(f"Loading category test split from {path}...")
    return load_from_disk(str(path))


def _ensure_all_category_sft_splits(dataset_path: str, *, filter_year: bool = True) -> None:
    """Rebuild and cache category-specific SFT test splits from the source dataset."""
    print("Rebuilding category-level SFT test caches from source dataset...")
    dataset_raw = load_dataset(dataset_path, split="test")
    cleaned_dataset = clean_dataset(dataset_raw, filter_year=filter_year)

    for token, normalized in CATEGORY_TOKEN_TO_NORMALIZED.items():
        target = normalized.lower()
        category_ds = cleaned_dataset.filter(
            lambda example, target=target: str(example.get("Primary Category", "")).lower()
            == target
        )
        if len(category_ds) == 0:
            print(f"Warning: no samples found for category '{normalized}'. Skipping cache save.")
            continue

        save_path = Path("data_cache") / "categories" / token / "sft" / "test"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        category_ds.save_to_disk(str(save_path))


def _load_all_categories_dataset(dataset_path: str) -> Dataset:
    datasets: list[Dataset] = []
    missing_tokens = [token for token in CATEGORY_TOKENS if _load_category_sft_split(token) is None]

    if missing_tokens:
        print(
            "Missing cached SFT test splits for categories: "
            + ", ".join(sorted(missing_tokens))
        )
        _ensure_all_category_sft_splits(dataset_path, filter_year=True)

    remaining_missing = []
    for token in CATEGORY_TOKENS:
        ds = _load_category_sft_split(token)
        if ds is None:
            remaining_missing.append(token)
        else:
            datasets.append(ds)

    if remaining_missing:
        print(
            "Warning: still missing category splits after rebuild: "
            + ", ".join(sorted(remaining_missing))
        )

    if not datasets:
        raise RuntimeError("No category test splits available. Please regenerate dataset caches.")

    combined = concatenate_datasets(datasets)
    print(
        f"Merged {len(datasets)} category test splits into a {len(combined)}-row dataset."
    )
    return combined


async def main():
    # --- Configuration ---
    dataset_path = "JasonYan777/novelty-rank-with-similarities"
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="WHOLE_DATASET",
                        help="Data category to evaluate (e.g., CS_CV, CS_RO, WHOLE_DATASET)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base model name registered with Tinker")
    parser.add_argument("--model-path", type=str,
                        default="tinker://2e566352-bd3a-5c10-8e5a-2743d49bc353:train:0/sampler_weights/final",
                        help="Tinker checkpoint URI (tinker://...) for sampling; leave empty for base model")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=10,
                        help="Maximum tokens to sample per response")
    parser.add_argument("--limit", type=int, default=500,
                        help="Number of evaluation examples to sample")
    parser.add_argument(
        "--include-similarity-report",
        action="store_true",
        help="Augment prompts with the precomputed similarity report",
    )
    args = parser.parse_args()

    category_arg = args.category or "WHOLE_DATASET"
    if category_arg.upper() in {"ALL", "WHOLE", "WHOLE_DATASET"}:
        category_arg = "WHOLE_DATASET"
    normalized_category = normalize_category(category_arg)
    category = normalized_category if normalized_category != "" else "WHOLE_DATASET"
    model_name = args.model_name
    model_path = args.model_path or None
    temperature = args.temperature
    max_tokens = args.max_tokens
    dataset_limit = max(1, args.limit)
    include_similarity_report = args.include_similarity_report

    if category == "whole_dataset":
        dataset = _load_all_categories_dataset(dataset_path)
    else:
        category_token = category_to_token(category)
        dataset = _load_category_sft_split(category_token)
        if dataset is None:
            print(
                f"No cached test split found for category '{category}'. Building from source dataset..."
            )
            _ensure_all_category_sft_splits(dataset_path, filter_year=True)
            dataset = _load_category_sft_split(category_token)
            if dataset is None:
                raise RuntimeError(
                    f"Failed to build test split for category '{category}'."
                )

    # --- End of main function logic ---
    print(f"Successfully loaded and cleaned dataset of size: {len(dataset)}")
    # shuffle and limit dataset size for testing
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(min(dataset_limit, len(dataset))))

    # Initialize sampler
    print("-------- Initializing sampler -------")
    print(f"Model Name: {model_name}")
    print(f"Model Path: {model_path}")
    print(f"Temperature: {temperature}")
    print(f"Max Tokens: {max_tokens}")
    print("-------- Sampler initialized -------")
    sampler = TinkerSampler(model_name=model_name,
                            model_path=model_path,
                            temperature=temperature, max_tokens=max_tokens)

    # asyncio generate predictions
    results = await asyncio.gather(
        *[
            process_one(
                data,
                sampler,
                include_similarity_report=include_similarity_report,
            )
            for data in dataset
        ]
    )
    predictions, labels = zip(*results)

    # Compute accuracy, precision, recall
    tp = sum(1 for p, l in zip(predictions, labels) if p == "1" and l == "1")
    fp = sum(1 for p, l in zip(predictions, labels) if p == "1" and l == "0")
    fn = sum(1 for p, l in zip(predictions, labels) if p == "0" and l == "1")
    tn = sum(1 for p, l in zip(predictions, labels) if p == "0" and l == "0")
    un_resolved = sum(1 for p in predictions if p not in ["0", "1"])

    total = tp + fp + fn + tn + un_resolved
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"test_accuracy": accuracy, "test_precision": precision, "test_recall": recall, "test_F1": F1}

if __name__ == "__main__":

    results = asyncio.run(main())
    print(results)