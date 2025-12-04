import os
import sys
import logging
import chz
import re
import asyncio
from typing import Optional, List, Dict, Tuple, Any

import tinker
from tinker import types
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from tinker_cookbook.supervised.types import ChatDatasetBuilder

from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers
import torch

# Setup paths to find custom modules
current_dir = os.path.dirname(os.path.abspath(__file__))
# Ensure the repository root is discoverable for local imports
repo_root = os.path.dirname(os.path.dirname(current_dir))
if repo_root not in sys.path:
    sys.path.append(repo_root)
# Local utilities
from models.Qwen_4B.data_access import (
    CATEGORY_SUBDIR_DEFAULT,
    DATA_VARIANT_SIM,
    EXPECTED_DPO_COLUMNS,
    TASK_CLASSIFICATION,
    TASK_COMPARISON,
    WHOLE_DATASET,
    load_category_dataset,
)
from models.Qwen_4B.utils import _balance_dataset_by_upsampling

# Configure basic logging
logger = logging.getLogger(__name__)


def _strip_novelty_verdict(example: dict[str, Any]) -> dict[str, Any]:
    """Remove embedded novelty verdict lines from the user prompt."""
    conversation = example.get("prompt_conversation")
    if not conversation:
        return example

    new_conversation: list[dict[str, Any]] = []
    updated = False
    for message in conversation:
        if message.get("role") != "user":
            new_conversation.append(message)
            continue

        content = message.get("content", "")
        if "Novelty Verdict" not in content and "<|im_end|>" not in content:
            new_conversation.append(message)
            continue

        filtered_lines = [
            line for line in content.splitlines()
            if not line.strip().startswith("Novelty Verdict")
        ]
        cleaned = "\n".join(filtered_lines).replace("<|im_end|>", "").strip()
        if cleaned != content:
            updated = True
            new_message = dict(message)
            new_message["content"] = cleaned
            new_conversation.append(new_message)
        else:
            new_conversation.append(message)

    if not updated:
        return example

    updated_example = dict(example)
    updated_example["prompt_conversation"] = new_conversation
    return updated_example

# =========================================================
# Create AccuracyOnLabeledTestSetEvaluator
# =========================================================
class AccuracyOnLabeledTestSetEvaluator(SamplingClientEvaluator):
    """
    Evaluates model accuracy on a labeled test set (0 or 1 classification).
    This evaluator computes accuracy, precision, recall, and F1 score.
    """

    def __init__(
        self,
        model_name: str,
        *,
        max_tokens: int = 16,
        category: str = WHOLE_DATASET,
        data_variant: str = DATA_VARIANT_SIM,
        category_subdir: str = CATEGORY_SUBDIR_DEFAULT,
        sample_limit: Optional[int] = 1000,
        include_similarity_report: bool = False,
        sft_task: str = TASK_CLASSIFICATION,
    ):
        if sft_task != TASK_CLASSIFICATION:
            raise ValueError("AccuracyOnLabeledTestSetEvaluator currently supports only classification SFT tasks")
        # Load the evaluation split directly from local caches.
        self.test_data = load_category_dataset(
            dataset_type=sft_task,
            split="test",
            category=category,
            variant=data_variant,
            category_subdir=category_subdir,
            expected_columns=EXPECTED_DPO_COLUMNS,
        )

        self.test_data = self.test_data.map(_strip_novelty_verdict)

        # Ensure label column exists for evaluation metrics.
        def _extract_label(example: dict[str, Any]) -> dict[str, Any]:
            assistant_messages = example.get("chosen", [])
            if not assistant_messages:
                raise ValueError("DPO example missing 'chosen' assistant response")
            label_text = assistant_messages[-1].get("content", "").strip()
            if label_text not in {"0", "1", "A", "B"}:
                raise ValueError(f"Unexpected label content: '{label_text}'")
            return {"label": label_text}

        self.test_data = self.test_data.map(_extract_label)

        self.test_data = self.test_data.shuffle(seed=42)
        if sample_limit is not None:
            effective_limit = min(sample_limit, len(self.test_data))
            self.test_data = self.test_data.select(range(effective_limit))
        logger.info(
            "[Evaluator] Loaded %d samples for evaluation (category=%s, variant=%s)",
            len(self.test_data),
            category,
            data_variant,
        )

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.include_similarity_report = include_similarity_report

        self.tokenizer = get_tokenizer(model_name)
        self.service_client = tinker.ServiceClient()

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run evaluation asynchronously using the current policy sampling client.
        """

        async def get_prediction(prompt_text: str) -> str:
            """
            Query the model for a single prediction ("0" or "1").
            """
            # Implementation details for asynchronous sampling...
            response = await sampling_client.sample_async(
                prompt=tinker.types.ModelInput.from_ints(
                    self.tokenizer.encode(prompt_text)
                ),
                sampling_params=tinker.types.SamplingParams(
                    max_tokens=self.max_tokens, temperature=0.0
                ),
                num_samples=1,
            )
            decoded = self.tokenizer.decode(response.sequences[0].tokens).strip()
            # Use regex to find the first '0' or '1' digit output by the model
            match = re.findall(r'[01]', decoded)
            decoded = match[0] if match else ""
            return decoded

        # Collect all user prompts and ground-truth labels
        prompts = []
        labels = []
        for ex in self.test_data:
            conversation = ex.get("prompt_conversation", [])
            assistant_messages = ex.get("chosen", [])
            if not conversation:
                raise ValueError("Evaluation example missing 'prompt_conversation'")
            if not assistant_messages:
                raise ValueError("Evaluation example missing 'chosen' response")
            prompts.append(conversation[-1]["content"])
            labels.append(ex["label"])

        # Query the model for predictions
        tasks = [get_prediction(p) for p in prompts]
        predictions = await asyncio.gather(*tasks)

        # Compute accuracy, precision, recall based on binary classification metrics
        tp = sum(1 for p, l in zip(predictions, labels) if p == "1" and l == "1")
        fp = sum(1 for p, l in zip(predictions, labels) if p == "1" and l == "0")
        fn = sum(1 for p, l in zip(predictions, labels) if p == "0" and l == "1")
        tn = sum(1 for p, l in zip(predictions, labels) if p == "0" and l == "0")
        un_resolved = sum(1 for p in predictions if p not in ["0", "1"]) # Count non-binary outputs

        total = tp + fp + fn + tn + un_resolved
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"test_accuracy": accuracy, "test_precision": precision, "test_recall": recall, "test_F1": F1}


class ComparisonPairwiseAccuracyEvaluator(SamplingClientEvaluator):
    """
    Evaluates pairwise accuracy for comparison-style SFT runs that predict "A" or "B".
    """

    def __init__(
        self,
        model_name: str,
        *,
        max_tokens: int = 8,
        category: str = WHOLE_DATASET,
        data_variant: str = DATA_VARIANT_SIM,
        category_subdir: str = CATEGORY_SUBDIR_DEFAULT,
        sample_limit: Optional[int] = 1000,
    ):
        # Load comparison evaluation pairs from local cache.
        self.test_data = load_category_dataset(
            dataset_type=TASK_COMPARISON,
            split="test",
            category=category,
            variant=data_variant,
            category_subdir=category_subdir,
            expected_columns=EXPECTED_DPO_COLUMNS,
        )

        self.test_data = self.test_data.shuffle(seed=42)
        if sample_limit is not None:
            effective_limit = min(sample_limit, len(self.test_data))
            self.test_data = self.test_data.select(range(effective_limit))
        logger.info(
            "[ComparisonEvaluator] Loaded %d samples for evaluation (category=%s, variant=%s)",
            len(self.test_data),
            category,
            data_variant,
        )

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.tokenizer = get_tokenizer(model_name)

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        async def get_prediction(prompt_text: str) -> str:
            response = await sampling_client.sample_async(
                prompt=tinker.types.ModelInput.from_ints(
                    self.tokenizer.encode(prompt_text)
                ),
                sampling_params=tinker.types.SamplingParams(
                    max_tokens=self.max_tokens, temperature=0.0
                ),
                num_samples=1,
            )
            decoded = self.tokenizer.decode(response.sequences[0].tokens).strip()
            match = re.findall(r"[AB]", decoded, flags=re.IGNORECASE)
            decoded = match[0].upper() if match else ""
            return decoded

        prompts = []
        labels = []
        for ex in self.test_data:
            conversation = ex.get("prompt_conversation", [])
            assistant_messages = ex.get("chosen", [])
            if not conversation:
                raise ValueError("Evaluation example missing 'prompt_conversation'")
            if not assistant_messages:
                raise ValueError("Evaluation example missing 'chosen' response")
            prompts.append(conversation[-1]["content"])
            label_text = assistant_messages[-1]["content"].strip()
            label_match = re.findall(r"[AB]", label_text, flags=re.IGNORECASE)
            if not label_match:
                raise ValueError(f"Comparison label missing A/B tag: {label_text!r}")
            labels.append(label_match[0].upper())

        tasks = [get_prediction(p) for p in prompts]
        predictions = await asyncio.gather(*tasks)

        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        unresolved = sum(1 for p in predictions if p not in {"A", "B"})
        total = len(labels)
        accuracy = correct / total if total > 0 else 0.0
        unresolved_rate = unresolved / total if total > 0 else 0.0

        return {"test_accuracy": accuracy, "test_unresolved_rate": unresolved_rate}


# ==========================================================
# Create SFTChatRenderer
# ==========================================================
class SFTChatRenderer:
    """
    A wrapper around a Chat Renderer for standard single-turn SFT tasks.
    It converts a chat conversation into token/weight tensors.
    """

    def __init__(self, base_renderer: renderers.Renderer):
        self.base_renderer = base_renderer

    def to_tokens_weights(
        self, conversation: List[Dict[str, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a chat-style conversation (user + assistant) to token IDs and weights.
        """
        tokens, weights = self.base_renderer.build_supervised_example(
            conversation,
            train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES  # Train only on assistant replies
        )
        return tokens, weights

    @property
    def tokenizer(self):
        return self.base_renderer.tokenizer


        

# ==========================================================
# Create NoveltyRankSFTDataBuilder
# ==========================================================
@chz.chz
class NoveltyRankSFTDataBuilder(ChatDatasetBuilder):
    """
    Builds an SFT dataset for NoveltyRank directly from local caches.
    """
    category: str = chz.field(default=WHOLE_DATASET)
    data_variant: str = chz.field(default=DATA_VARIANT_SIM)
    category_subdir: str = chz.field(default=CATEGORY_SUBDIR_DEFAULT)
    balance_dataset: bool = False
    include_similarity_report: bool = chz.field(default=False)
    sft_task: str = chz.field(default=TASK_CLASSIFICATION)

    def __call__(self):
        if self.sft_task not in {TASK_CLASSIFICATION, TASK_COMPARISON}:
            raise ValueError(f"Unsupported sft_task '{self.sft_task}'")

        train_ds = load_category_dataset(
            dataset_type=self.sft_task,
            split="train",
            category=self.category,
            variant=self.data_variant,
            category_subdir=self.category_subdir,
            expected_columns=EXPECTED_DPO_COLUMNS,
        )

        train_ds = train_ds.map(_strip_novelty_verdict)

        def _extract_label(example: dict[str, Any]) -> dict[str, Any]:
            assistant_messages = example.get("chosen", [])
            if not assistant_messages:
                raise ValueError("DPO example missing 'chosen' assistant response")
            label_text = assistant_messages[-1].get("content", "").strip()
            if not label_text:
                raise ValueError("Assistant response must contain content")
            return {"label": label_text}

        train_ds = train_ds.map(_extract_label)
        train_ds = train_ds.shuffle(seed=42)

        if len(train_ds) == 0:
            raise RuntimeError(
                f"Training dataset empty for category={self.category}, variant={self.data_variant}"
            )

        if self.balance_dataset:
            if self.sft_task == TASK_CLASSIFICATION:
                train_ds = _balance_dataset_by_upsampling(train_ds)
            else:
                logger.info(
                    "Skipping balancing for non-classification SFT task '%s'",
                    self.sft_task,
                )
    
        logger.info(
            "[DataBuilder] Final training dataset size: %d (category=%s, variant=%s)",
            len(train_ds),
            self.category,
            self.data_variant,
        )

        sft_renderer = SFTChatRenderer(self.renderer)

        # define function to convert each example to Datums
        def example_to_datum(example: dict[str, Any]) -> list[types.Datum]:
            conversation = example.get("prompt_conversation", [])
            assistant_messages = example.get("chosen", [])
            if not conversation:
                raise ValueError("Training example missing 'prompt_conversation'")
            if not assistant_messages:
                raise ValueError("Training example missing 'chosen' response")

            assistant_response = assistant_messages[-1]["content"]
            conversation = [dict(msg) for msg in conversation]
            conversation.append({"role": "assistant", "content": assistant_response})

            # convert conversation to Datum
            tokens, weights = sft_renderer.to_tokens_weights(conversation)
            # Use max_length from common config
            datum = datum_from_tokens_weights(tokens, weights, self.common_config.max_length)
            return [datum]

        # transform datasets to SupervisedDatasets
        return (
            SupervisedDatasetFromHFDataset(
                train_ds,
                batch_size=self.common_config.batch_size,
                flatmap_fn=example_to_datum,
            ),
            None,  # No validation set
        )