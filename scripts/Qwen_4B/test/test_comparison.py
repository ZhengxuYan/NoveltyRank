# __init__.py
"""
Initialize model samplers and handle prompt construction logic for comparison tasks.
"""
import os
import sys
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)
from pathlib import Path

from models.Qwen_4B.data_access import (
    DEFAULT_CATEGORIES,
    category_to_token,
    normalize_category,
)
import asyncio
from datasets import Dataset, concatenate_datasets, load_from_disk

import re
import tinker
from dotenv import load_dotenv
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer

try:
    from tqdm import tqdm
except ImportError:  # tqdm provides nicer progress but is optional
    tqdm = None

# Initialize Tinker Service Client
load_dotenv()
service_client = tinker.ServiceClient()

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
    
def _comparison_split_path(category_token: str, *, root: Path) -> Path:
    return root / category_token / "dpo" / "comparison" / "test"


def _load_category_comparison_split(category_token: str, *, root: Path) -> Dataset:
    path = _comparison_split_path(category_token, root=root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing comparison test split for token '{category_token}' at {path}."
        )
    print(f"Loading comparison test split from {path}...")
    return load_from_disk(str(path))


def _load_all_categories_comparison(*, root: Path) -> Dataset:
    datasets: list[Dataset] = []
    missing_tokens: list[str] = []

    for category in DEFAULT_CATEGORIES:
        token = category_to_token(category)
        try:
            datasets.append(_load_category_comparison_split(token, root=root))
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")
            missing_tokens.append(token)

    if missing_tokens:
        raise FileNotFoundError(
            "Missing comparison test splits for categories: "
            + ", ".join(sorted(missing_tokens))
        )

    combined = concatenate_datasets(datasets)
    print(
        f"Merged {len(datasets)} category comparison splits into {len(combined)} total examples."
    )
    return combined


async def process_one(data, sampler):
    # data is a row from the DPO pairs dataset
    messages = data["prompt_conversation"]
    label = data["chosen"][0]["content"]
    
    prediction = await sampler.generate(messages)
    prediction = prediction['content']
    
    # Extract "A" or "B" from prediction using regex
    match = re.search(r'\b(A|B)\b', prediction, re.IGNORECASE)
    prediction = match.group(0).upper() if match else ""
    
    return prediction, label

async def main():
    # --- Configuration ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="WHOLE_DATASET",
                        help="Data category to build DPO pairs for (e.g., CS_CV, CS_RO, WHOLE_DATASET)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base model name registered with Tinker")
    parser.add_argument("--model-path", type=str,
                        default="tinker://c2cf9723-af77-5458-9032-a7f5b10b20da:train:0/sampler_weights/final",
                        help="Tinker checkpoint URI (tinker://...) for sampling; leave empty for base model")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=10,
                        help="Maximum tokens to sample per response")
    parser.add_argument("--limit", type=int, default=100,
                        help="Number of comparison examples to evaluate")
    parser.add_argument(
        "--use-similarity-report",
        action="store_true",
        help="Load comparison splits that include similarity reports (from similiarity_aware_categories).",
    )
    args = parser.parse_args()
    category_arg = args.category or "WHOLE_DATASET"
    if category_arg.upper() in {"ALL", "WHOLE", "WHOLE_DATASET"}:
        category_arg = "WHOLE_DATASET"
    normalized_category = normalize_category(category_arg)
    if not normalized_category:
        category = "whole_dataset"
    elif normalized_category.replace(".", "_") == "whole_dataset":
        category = "whole_dataset"
    else:
        category = normalized_category
    model_name = args.model_name
    model_path = args.model_path or None
    temperature = args.temperature
    max_tokens = args.max_tokens
    dataset_limit = max(1, args.limit)

    category_root = Path("data_cache") / (
        "similiarity_aware_categories" if args.use_similarity_report else "categories"
    )
    if not category_root.exists():
        raise FileNotFoundError(
            f"Category root '{category_root}' not found. Ensure the required caches are generated."
        )

    if category == "whole_dataset":
        dataset = _load_all_categories_comparison(root=category_root)
    else:
        category_token = category_to_token(category)
        dataset = _load_category_comparison_split(category_token, root=category_root)

    # --- End of main function logic ---
    print(f"Successfully loaded DPO comparison dataset of size: {len(dataset)}")
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
    if tqdm is not None:
        tasks = [
            asyncio.create_task(process_one(dataset[idx], sampler))
            for idx in range(len(dataset))
        ]
        results = []
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Evaluating",
        ):
            results.append(await coro)
    else:
        results = await asyncio.gather(
            *[process_one(dataset[idx], sampler) for idx in range(len(dataset))]
        )
    predictions, labels = zip(*results)

    # Compute accuracy
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0
    
    un_resolved = sum(1 for p in predictions if p not in ["A", "B"]) 

    print(f"Total: {total}, Correct: {correct}, Unresolved: {un_resolved}")

    return {"test_accuracy": accuracy}

if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)
