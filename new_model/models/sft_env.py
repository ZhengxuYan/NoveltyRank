"""SFT training environment for the similarity-augmented CV dataset."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import asyncio
import os

import chz
import torch
from datasets import Dataset, load_from_disk

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from tinker_cookbook.supervised.types import ChatDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .utils import _balance_dataset_by_upsampling, create_sft_example

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_DATASET_PATH = "simliarity_report/datasets/train_similarity_reports"
DEFAULT_TEST_DATASET_PATH = "simliarity_report/datasets/test_similarity_reports"


class AccuracyOnLabeledTestSetEvaluator(SamplingClientEvaluator):
    """Evaluate binary classification accuracy on the similarity-aware test split."""

    def __init__(
        self,
        model_name: str,
        test_dataset_path: str = DEFAULT_TEST_DATASET_PATH,
        *,
        max_tokens: int = 16,
        sample_limit: int = 1000,
    ) -> None:
        if not test_dataset_path or not _dataset_exists(test_dataset_path):
            raise FileNotFoundError(
                f"Test dataset '{test_dataset_path}' not found. Generate the similarity report test split first."
            )

        self.test_data = load_from_disk(test_dataset_path)
        self.test_data = self.test_data.shuffle(seed=42)
        if sample_limit is not None:
            self.test_data = self.test_data.select(range(min(sample_limit, len(self.test_data))))
        logger.info("[Evaluator] Loaded %d evaluation samples", len(self.test_data))

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.tokenizer = get_tokenizer(model_name)
        self.service_client = tinker.ServiceClient()

    async def __call__(self, sampling_client: tinker.SamplingClient) -> Dict[str, float]:
        async def classify(prompt_text: str) -> str:
            response = await sampling_client.sample_async(
                prompt=types.ModelInput.from_ints(self.tokenizer.encode(prompt_text)),
                sampling_params=types.SamplingParams(max_tokens=self.max_tokens, temperature=0.0),
                num_samples=1,
            )
            decoded = self.tokenizer.decode(response.sequences[0].tokens).strip()
            for char in decoded:
                if char in {"0", "1"}:
                    return char
            return ""

        prompts: List[str] = []
        labels: List[str] = []
        for record in self.test_data:
            user_prompt, label = create_sft_example(record)
            prompts.append(user_prompt)
            labels.append(label)

        predictions = await asyncio.gather(*(classify(p) for p in prompts))

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
        }


class SFTChatRenderer:
    """Render chat-style examples into token/weight tensors."""

    def __init__(self, base_renderer: renderers.Renderer) -> None:
        self.base_renderer = base_renderer

    def to_tokens_weights(self, conversation: List[Dict[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, weights = self.base_renderer.build_supervised_example(
            conversation,
            train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )
        return tokens, weights

    @property
    def tokenizer(self):
        return self.base_renderer.tokenizer


@chz.chz
class NoveltyRankSFTDataBuilder(ChatDatasetBuilder):
    """Create an SFT dataset from the similarity-augmented CV split."""

    train_dataset_path: str = DEFAULT_TRAIN_DATASET_PATH
    sample_limit: Optional[int] = None
    balance_dataset: bool = True

    def __call__(self) -> Tuple[SupervisedDatasetFromHFDataset, None]:
        if not _dataset_exists(self.train_dataset_path):
            raise FileNotFoundError(
                f"Train dataset '{self.train_dataset_path}' not found. Generate the similarity report train split first."
            )

        train_ds: Dataset = load_from_disk(self.train_dataset_path)
        if self.sample_limit is not None:
            train_ds = train_ds.select(range(min(self.sample_limit, len(train_ds))))
        if self.balance_dataset:
            train_ds = _balance_dataset_by_upsampling(train_ds)

        logger.info("[DataBuilder] Training dataset size: %d", len(train_ds))

        renderer = SFTChatRenderer(self.renderer)

        def example_to_datum(example: Dict[str, str]):
            user_prompt, label = create_sft_example(example)
            conversation = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": label},
            ]
            tokens, weights = renderer.to_tokens_weights(conversation)
            datum = datum_from_tokens_weights(tokens, weights, self.common_config.max_length)
            return [datum]

        supervised = SupervisedDatasetFromHFDataset(
            train_ds,
            batch_size=self.common_config.batch_size,
            flatmap_fn=example_to_datum,
        )
        return supervised, None


def _dataset_exists(path: str) -> bool:
    return bool(path) and os.path.isdir(path)
