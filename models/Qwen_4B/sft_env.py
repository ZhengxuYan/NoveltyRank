import os
import sys
import logging
import chz
import re
import asyncio
from typing import cast, Optional, List, Dict, Tuple
# Import load_from_disk here
from datasets import Dataset, load_dataset, load_from_disk 

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
# Assuming the structure involves going up two levels to find 'models'
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
# Assuming clean_dataset is available from this utility path
from models.Qwen_4B.utils import clean_dataset, generate_prompts_and_labels, _balance_dataset_by_upsampling

# Configure basic logging
logger = logging.getLogger(__name__)

# =========================================================
# Create AccuracyOnLabeledTestSetEvaluator
# =========================================================
class AccuracyOnLabeledTestSetEvaluator(SamplingClientEvaluator):
    """
    Evaluates model accuracy on a labeled test set (0 or 1 classification).
    This evaluator computes accuracy, precision, recall, and F1 score.
    """

    def __init__(self, dataset_path: str, model_name: str, max_tokens: int = 16, local_cache_path: str = "data_cache/test_sft_data"):
        
        # Define the path for the specific 'test' split inside the cache directory
        local_split_path = os.path.join(local_cache_path, "test_split_cleaned")
        
        # ------------------------------------------------------------------
        # FEATURE: Check Local -> Load Cleaned Data; Else -> Web -> Clean -> Save
        # ------------------------------------------------------------------
        if os.path.exists(local_split_path):
            logger.info(f"[Evaluator] Local CLEANED cache found at {local_split_path}. Loading from disk...")
            # OPTIMIZATION: Load the single split directly (faster)
            cleaned_test_ds = load_from_disk(local_split_path)
        else:
            logger.info(f"[Evaluator] No local cache found. Downloading {dataset_path} from web...")
            dataset = load_dataset(dataset_path)
            
            # 1. Execute cleaning operation
            logger.info("[Evaluator] Downloading complete. Starting data cleaning...")
            cleaned_test_ds = clean_dataset(dataset["test"], filter_year=False)
            
            # 2. OPTIMIZATION: Save the single split directly, avoiding Dataset.from_dict() overhead
            logger.info(f"[Evaluator] Cleaning complete, saving split directly to {local_split_path}...")
            cleaned_test_ds.save_to_disk(local_split_path)
        
        self.test_data = cleaned_test_ds
        # ------------------------------------------------------------------

        # Further sampling and processing for the evaluation run
        self.test_data = self.test_data.shuffle(seed=42)
        # Select a sample size (min(1000, total_size))
        self.test_data = self.test_data.select(range(min(1000, len(self.test_data))))
        logger.info(f"[Evaluator] Using {len(self.test_data)} samples for evaluation.")

        self.model_name = model_name
        self.max_tokens = max_tokens

        # Create a tokenizer for encoding/decoding text
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
            # generate_prompts_and_labels is defined below
            user_prompts, user_labels = generate_prompts_and_labels(ex) 
            prompts.append(user_prompts)
            labels.append(user_labels)

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
    Builds an SFT dataset for novelty ranking using a Hugging Face dataset.
    """
    dataset_path: str

    # ADDED: Configuration for local path
    local_cache_path: str = chz.field(
        default="data_cache/train_sft_data",
    )
    balance_dataset: bool = True

    def __call__(self):
        
        local_split_path = os.path.join(self.local_cache_path, "train_split_cleaned")
        
        # ------------------------------------------------------------------
        # FEATURE: Check Local -> Load Cleaned Data; Else -> Web -> Clean -> Save
        # ------------------------------------------------------------------
        if os.path.exists(local_split_path):
            logger.info(f"[DataBuilder] Local CLEANED cache found at {local_split_path}. Loading from disk...")
            # OPTIMIZATION: Load the single split directly (faster)
            train_ds = load_from_disk(local_split_path)
        else:
            logger.info(f"[DataBuilder] No local cache found. Downloading {self.dataset_path} from web...")
            dataset = load_dataset(self.dataset_path)
            
            # 1. Execute cleaning operation
            logger.info("[DataBuilder] Downloading complete. Starting data cleaning...")
            cleaned_train_ds = cast(Dataset, clean_dataset(dataset["train"]))
            
            # 2. OPTIMIZATION: Save the single split directly, avoiding Dataset.from_dict() overhead
            logger.info(f"[DataBuilder] Cleaning complete, saving split directly to {local_split_path}...")
            cleaned_train_ds.save_to_disk(local_split_path)
            
            train_ds = cleaned_train_ds
        # ------------------------------------------------------------------

        if self.balance_dataset:
            train_ds = _balance_dataset_by_upsampling(train_ds)
    
        logger.info(f"[DataBuilder] Final training dataset size: {len(train_ds)}")

        sft_renderer = SFTChatRenderer(self.renderer)

        # define function to convert each example to Datums
        def example_to_datum(example: dict[str, str]) -> list[types.Datum]:
            user_prompt, label = generate_prompts_and_labels(example)

            # convert to conversation format
            conversation = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": label}
            ]

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