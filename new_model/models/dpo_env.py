"""DPO training environment for classification on the similarity-aware CV dataset."""
from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import chz
from datasets import Dataset as HFDataset, load_from_disk

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import Evaluator, EvaluatorBuilder, SamplingClientEvaluator
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

from .utils import create_classification_example, generate_classification_dpo_pairs

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_DATASET_PATH = "data_cache/similiarity_aware_categories/CS_CV/sft/train"
DEFAULT_TEST_DATASET_PATH = "data_cache/similiarity_aware_categories/CS_CV/sft/test"


class NoveltyDPODataset(SupervisedDataset):
    """Interleave chosen/rejected responses for DPO classification."""

    def __init__(self, pairs: List[Dict], tokenizer: Tokenizer, renderer: renderers.Renderer, batch_size: int) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.batch_size = batch_size
        self.pairs_per_batch = max(self.batch_size // 2, 1)
        self.indices = list(range(len(self.pairs)))

    def __len__(self) -> int:
        return max(len(self.pairs) * 2 // self.batch_size, 1)

    def set_epoch(self, seed: int = 0) -> None:
        rng = random.Random(seed)
        rng.shuffle(self.indices)

    def _render_pair(self, pair_data: Dict, use_rejected: bool) -> tinker.Datum:
        conversation = pair_data["prompt_conversation"] + (
            pair_data["rejected"] if use_rejected else pair_data["chosen"]
        )
        tokens, weights = self.renderer.build_supervised_example(
            conversation, train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )
        return datum_from_tokens_weights(tokens, weights)

    def get_batch(self, batch_idx: int) -> List[tinker.Datum]:
        start = batch_idx * self.pairs_per_batch
        end = start + self.pairs_per_batch
        if start >= len(self.indices):
            return []

        batch_indices = self.indices[start:end]
        datums: List[tinker.Datum] = []
        for idx in batch_indices:
            pair = self.pairs[idx]
            datums.append(self._render_pair(pair, use_rejected=False))
            datums.append(self._render_pair(pair, use_rejected=True))
        return datums


class NoveltyRankAccuracyEvaluator(SamplingClientEvaluator):
    """Evaluate 0/1 accuracy for classification pairs."""

    def __init__(self, test_dataset: HFDataset, renderer_name: str, tokenizer_model_name: str, max_tokens: int) -> None:
        self.test_ds = test_dataset
        self.renderer = renderers.get_renderer(
            renderer_name, tokenizer=get_tokenizer(tokenizer_model_name)
        )
        self.max_tokens = max_tokens

    async def __call__(self, sampling_client: tinker.SamplingClient) -> Dict[str, float]:
        prompts = [row["prompt_conversation"] for row in self.test_ds]
        labels = [row["chosen"][0]["content"].strip() for row in self.test_ds]

        async def classify(conversation: List[Dict[str, str]]) -> str:
            prompt = (
                sampling_client.renderer.build_generation_prompt(conversation)
                if hasattr(sampling_client, "renderer")
                else types.ModelInput.from_ints(
                    self.renderer.tokenizer.encode(conversation[0]["content"])
                )
            )
            result = await sampling_client.sample_async(
                prompt=prompt,
                sampling_params=types.SamplingParams(max_tokens=self.max_tokens, temperature=0.0),
                num_samples=1,
            )
            decoded = self.renderer.tokenizer.decode(result.sequences[0].tokens).strip()
            for char in decoded:
                if char in {"0", "1"}:
                    return char
            return ""

        predictions = await asyncio.gather(*(classify(conv) for conv in prompts))

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


@chz.chz
class NoveltyRankEvaluatorBuilder(EvaluatorBuilder):
    """Builder that loads the similarity-aware test split for evaluation."""

    test_dataset_path: str = DEFAULT_TEST_DATASET_PATH
    renderer_name: str
    model_name: str
    max_tokens: int = 10

    def __call__(self) -> Evaluator:
        if not os.path.isdir(self.test_dataset_path):
            raise FileNotFoundError(
                f"Test dataset '{self.test_dataset_path}' not found. Generate the similarity report test split first."
            )
        test_records = load_from_disk(self.test_dataset_path)
        dpo_pairs = test_records.map(
            create_classification_example,
            remove_columns=test_records.column_names,
            desc="Generating classification evaluation pairs",
        )
        logger.info(
            "Loaded %d similarity-aware records and built %d evaluation pairs",
            len(test_records),
            len(dpo_pairs),
        )
        return NoveltyRankAccuracyEvaluator(
            test_dataset=dpo_pairs,
            renderer_name=self.renderer_name,
            tokenizer_model_name=self.model_name,
            max_tokens=self.max_tokens,
        )


@chz.chz
class NoveltyRankDatasetLoader(ChatDatasetBuilder):
    """Produce DPO classification pairs from the similarity-aware train split."""

    common_config: ChatDatasetBuilderCommonConfig
    train_dataset_path: str = DEFAULT_TRAIN_DATASET_PATH
    sample_limit: Optional[int] = None

    def __call__(self) -> Tuple[SupervisedDataset, Optional[SupervisedDataset]]:
        if not os.path.isdir(self.train_dataset_path):
            raise FileNotFoundError(
                f"Train dataset '{self.train_dataset_path}' not found. Generate the similarity report train split first."
            )
        train_records = load_from_disk(self.train_dataset_path)
        if self.sample_limit is not None:
            train_records = train_records.select(range(min(self.sample_limit, len(train_records))))

        logger.info("Loaded %d training samples for similarity-aware DPO classification", len(train_records))

        dpo_pairs = generate_classification_dpo_pairs(train_records)
        logger.info("Generated %d classification pairs", len(dpo_pairs))

        dataset = NoveltyDPODataset(
            pairs=list(dpo_pairs),
            tokenizer=self.tokenizer,
            renderer=self.renderer,
            batch_size=self.common_config.batch_size,
        )
        return dataset, None
