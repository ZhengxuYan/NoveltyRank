import os
import sys
import logging
import random
import asyncio
from typing import Any, List, Tuple, Dict, Union, Optional
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
import chz
from datasets import load_dataset, Dataset as HFDataset

# Add current path to sys.path to ensure tinker modules can be found
sys.path.append(os.getcwd())

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer, Tokenizer
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder, 
    SupervisedDataset, 
    ChatDatasetBuilderCommonConfig
)
from tinker_cookbook.eval.evaluators import Evaluator, EvaluatorBuilder

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Import Helpers from External Module (User Logic)
# -----------------------------------------------------------------------------
# Temporarily add the grandparent directory to path to find models module
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming the structure involves going up two levels to find 'models'
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

try:
    from models.Qwen_4B.sythesize_preference_pair import (
        clean_dataset,
        generate_dpo_pairs_from_hf,
    )
except ImportError as e:
    logger.error("Failed to import helper functions. Please check your path structure.")
    raise e

# -----------------------------------------------------------------------------
# Dataset Implementation
# -----------------------------------------------------------------------------

class NoveltyDPODataset(SupervisedDataset):
    """
    A dataset that flattens DPO pairs into an interleaved sequence:
    [Chosen_0, Rejected_0, Chosen_1, Rejected_1, ...]
    This is required by the tinker-cookbook DPO implementation.
    """
    def __init__(self, pairs: List[Dict], tokenizer: Tokenizer, renderer: renderers.Renderer, batch_size: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.batch_size = batch_size
        
        # Calculate how many pairs fit in one batch
        self.pairs_per_batch = self.batch_size // 2
        
        self.indices = list(range(len(self.pairs)))

    def __len__(self) -> int:
        # Total items = 2 * number of pairs (Chosen + Rejected)
        return len(self.pairs) * 2 // self.batch_size

    def set_epoch(self, seed: int = 0):
        # Shuffle the pairs indices to ensure randomness across epochs
        rng = random.Random(seed)
        rng.shuffle(self.indices)

    def _process_single_item(self, pair_data: Dict, is_rejected: bool) -> tinker.Datum:
        """Helper to process one conversation into a Datum"""
        prompt_msgs = pair_data["prompt_conversation"]
        response_msgs = pair_data["rejected"] if is_rejected else pair_data["chosen"]
        
        full_conversation = prompt_msgs + response_msgs
        
        tokens, weights = self.renderer.build_supervised_example(
            full_conversation,
            train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )
        return datum_from_tokens_weights(tokens, weights)


    def get_batch(self, batch_idx: int) -> List[tinker.Datum]:
        """
        Returns a list of Datums of size self.batch_size
        """
        # Calculate which pairs belong to this batch
        start_idx = batch_idx * self.pairs_per_batch
        end_idx = start_idx + self.pairs_per_batch
        
        # Guard against index out of range (though __len__ should prevent this)
        if start_idx >= len(self.indices):
            return []

        batch_pair_indices = self.indices[start_idx : end_idx]
        batch_datums = []

        for pair_idx in batch_pair_indices:
            pair_data = self.pairs[pair_idx]
            
            # 1. Add Chosen
            chosen_datum = self._process_single_item(pair_data, is_rejected=False)
            batch_datums.append(chosen_datum)
            
            # 2. Add Rejected
            rejected_datum = self._process_single_item(pair_data, is_rejected=True)
            batch_datums.append(rejected_datum)
            
        return batch_datums


# -----------------------------------------------------------------------------
# Evaluator Implementation
# -----------------------------------------------------------------------------

class NoveltyRankAccuracyEvaluator(SamplingClientEvaluator):
    def __init__(self, test_dataset: HFDataset, renderer_name: str, tokenizer_model_name: str, max_tokens: int):
        self.test_ds = test_dataset
        self.renderer = renderers.get_renderer(renderer_name, tokenizer=get_tokenizer(tokenizer_model_name))
        self.max_tokens = max_tokens
        
    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        prompts = [r["prompt_conversation"] for r in self.test_ds]
        ground_truths = [r["chosen"][0]["content"].strip() for r in self.test_ds]

        async def generate(client, p_convs):
            tasks = []
            for p in p_convs:
                inp = client.renderer.build_generation_prompt(p) if hasattr(client, 'renderer') else \
                      tinker.types.ModelInput.from_ints(self.renderer.tokenizer.encode(p[0]["content"]))
                tasks.append(client.sample_async(
                    prompt=inp,
                    sampling_params=tinker.types.SamplingParams(max_tokens=self.max_tokens, temperature=0.0),
                    num_samples=1
                ))
            res = await asyncio.gather(*tasks)
            return [self.renderer.tokenizer.decode(r.sequences[0].tokens).strip() for r in res]

        preds = await generate(sampling_client, prompts)
        correct = sum(1 for p, t in zip(preds, ground_truths) if p.strip().replace("<|endoftext|>", "").replace(".","").upper() == t)
        return {"test_accuracy": correct / len(preds) if preds else 0.0}
    

@chz.chz
class NoveltyRankEvaluatorBuilder(EvaluatorBuilder):
    dataset_path: str 
    renderer_name: str
    model_name: str
    max_tokens: int = 10

    def __call__(self) -> Evaluator:
        dataset = load_dataset(self.dataset_path)
        test_raw = clean_dataset(dataset["test"], filter_year=False)
        test_dpo_pairs = generate_dpo_pairs_from_hf(test_raw)
        
        # print
        print("-----------------------------------------------")
        print(f"Loaded {len(test_dpo_pairs)} test DPO pairs for evaluation.")
        print("------------------------------------------------")
        return NoveltyRankAccuracyEvaluator(
            test_dataset=test_dpo_pairs,
            renderer_name=self.renderer_name,
            tokenizer_model_name=self.model_name,
            max_tokens=self.max_tokens
        )

# -----------------------------------------------------------------------------
# Dataset Builder
# -----------------------------------------------------------------------------

@chz.chz
class NoveltyRankDatasetLoader(ChatDatasetBuilder):
    common_config: ChatDatasetBuilderCommonConfig
    dataset_path: str

    def __call__(self) -> Tuple[SupervisedDataset, Optional[SupervisedDataset]]:
        logger.info(f"Loading HF dataset: {self.dataset_path}")
        dataset = load_dataset(self.dataset_path)
        train_raw = clean_dataset(dataset["train"], filter_year=True)
        logger.info("Converting to DPO pairs...")
        train_dpo_pairs = generate_dpo_pairs_from_hf(train_raw)
        # print
        print("-----------------------------------------------")
        print(f"Generated {len(train_dpo_pairs)} DPO comparison examples for training.")
        print("------------------------------------------------")

        # Create the training dataset wrapper
        train_ds = NoveltyDPODataset(
            pairs=train_dpo_pairs, 
            tokenizer=self.tokenizer, 
            renderer=self.renderer,
            batch_size=self.common_config.batch_size
        )
        
        return train_ds, None