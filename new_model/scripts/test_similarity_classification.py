"""Evaluate a model on the similarity-augmented CV classification task."""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import re
import sys
from typing import List, Sequence, Tuple

from datasets import load_from_disk
from dotenv import load_dotenv

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Ensure repo root import when executed as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from new_model.models.utils import (  # pylint: disable=wrong-import-position
    create_classification_example,
    create_sft_example,
)

load_dotenv()
SERVICE_CLIENT = tinker.ServiceClient()


class TinkerSampler:
    """Wrapper around the Tinker sampling client."""

    def __init__(
        self,
        *,
        model_name: str,
        model_path: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 16,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> None:
        tokenizer = get_tokenizer(model_name)
        renderer_name = get_recommended_renderer_name(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
        self.sampling_params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=self.renderer.get_stop_sequences(),
        )
        self.sampling_client = SERVICE_CLIENT.create_sampling_client(
            model_path=model_path,
            base_model=model_name,
        )

    async def _decode(self, model_input: types.ModelInput) -> str:
        result = await self.sampling_client.sample_async(
            prompt=model_input,
            sampling_params=self.sampling_params,
            num_samples=1,
        )
        decoded = self.renderer.tokenizer.decode(result.sequences[0].tokens)
        return decoded.strip()

    async def generate(self, prompt: str) -> str:
        messages = [renderers.Message(role="user", content=prompt)]
        model_input = self.renderer.build_generation_prompt(messages)
        return await self._decode(model_input)

    async def generate_conversation(self, conversation: Sequence[dict[str, str]]) -> str:
        messages = [renderers.Message(role=entry["role"], content=entry["content"]) for entry in conversation]
        model_input = self.renderer.build_generation_prompt(messages)
        return await self._decode(model_input)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classification accuracy on similarity reports")
    parser.add_argument(
        "--dataset-path",
        default="simliarity_report/datasets/test_similarity_reports",
        help="Path to the similarity-augmented dataset saved with Hugging Face",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model name registered with Tinker",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional fine-tuned checkpoint URI (tinker://...)",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=16, help="Generation max tokens")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling probability")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k cutoff (-1 disables)")
    parser.add_argument("--limit", type=int, default=500, help="Number of examples to evaluate")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when sampling a subset",
    )
    parser.add_argument(
        "--mode",
        choices=("sft", "dpo"),
        default="sft",
        help="Evaluation prompt style: 'sft' for single-turn prompts, 'dpo' for preference pairs",
    )
    return parser.parse_args()


async def evaluate_dataset(
    dataset_path: str,
    sampler: TinkerSampler,
    limit: int,
    seed: int,
    mode: str,
) -> Tuple[List[str], List[str]]:
    dataset = load_from_disk(dataset_path)
    rng = random.Random(seed)

    prompts: List[str] = []
    conversations: List[Sequence[dict[str, str]]] = []
    labels: List[str] = []

    if mode == "dpo":
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        take = min(limit, len(indices))
        for idx in indices[:take]:
            pair = create_classification_example(dataset[idx])
            conversations.append(pair["prompt_conversation"])
            labels.append(str(pair["chosen"][0]["content"]).strip())
        generations = await asyncio.gather(
            *(sampler.generate_conversation(conversation) for conversation in conversations)
        )
    else:
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        take = min(limit, len(indices))
        for idx in indices[:take]:
            example = dataset[idx]
            prompt, label = create_sft_example(example)
            prompts.append(prompt)
            labels.append(label)
        generations = await asyncio.gather(*(sampler.generate(prompt) for prompt in prompts))

    predictions: List[str] = []
    for generation in generations:
        match = re.search(r"\b(0|1)\b", generation)
        predictions.append(match.group(1) if match else "")

    return predictions, labels


def compute_metrics(predictions: List[str], labels: List[str]) -> dict[str, float]:
    tp = sum(1 for pred, label in zip(predictions, labels) if pred == "1" and label == "1")
    tn = sum(1 for pred, label in zip(predictions, labels) if pred == "0" and label == "0")
    fp = sum(1 for pred, label in zip(predictions, labels) if pred == "1" and label == "0")
    fn = sum(1 for pred, label in zip(predictions, labels) if pred == "0" and label == "1")
    unresolved = sum(1 for pred in predictions if pred not in {"0", "1"})

    total = tp + tn + fp + fn + unresolved
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_F1": f1,
        "test_unresolved": float(unresolved),
        "test_total": float(total),
    }


async def async_main() -> None:
    args = parse_args()
    sampler = TinkerSampler(
        model_name=args.model_name,
        model_path=args.model_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    predictions, labels = await evaluate_dataset(
        dataset_path=args.dataset_path,
        sampler=sampler,
        limit=max(1, args.limit),
        seed=args.seed,
        mode=args.mode,
    )

    metrics = compute_metrics(predictions, labels)
    print(metrics)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
