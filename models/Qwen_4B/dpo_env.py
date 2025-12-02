import os
import sys
import logging
import random
import asyncio
from typing import List, Tuple, Dict, Optional, Any
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
import chz
import re
from datasets import load_from_disk, Dataset as HFDataset

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

from models.Qwen_4B.utils import create_classification_example, generate_classification_dpo_pairs
from models.Qwen_4B.utils.pipelines import (
    WHOLE_DATASET,
    DEFAULT_TRAIN_CACHE_DIR,
    DEFAULT_TEST_CACHE_DIR,
    ensure_base_sft_splits,
    ensure_category_resources,
)

logger = logging.getLogger(__name__)

# Constants for mode selection
COMPARISION_MODE = "comparison"
CLASSIFICATION_MODE = "classification"

# -----------------------------------------------------------------------------
# Import Helpers from External Module (User Logic)
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming the structure involves going up two levels to find 'models'
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

# -----------------------------------------------------------------------------
# Dataset Implementation
# -----------------------------------------------------------------------------

class NoveltyDPODataset(SupervisedDataset):
    """
    A dataset that flattens DPO pairs into an interleaved sequence:
    [Chosen_0, Rejected_0, Chosen_1, Rejected_1, ...]
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
    def __init__(self, test_dataset: HFDataset, renderer_name: str, tokenizer_model_name: str, max_tokens: int, dpo_mode: str = COMPARISION_MODE):
        self.test_ds = test_dataset
        self.renderer = renderers.get_renderer(renderer_name, tokenizer=get_tokenizer(tokenizer_model_name))
        self.max_tokens = max_tokens
        self.dpo_mode = dpo_mode
        
    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        prompts = [r["prompt_conversation"] for r in self.test_ds]
        ground_truths = [r["chosen"][0]["content"].strip() for r in self.test_ds]

        async def generate(client, p_convs):
            tasks = []
            for p in p_convs:
                inp = client.renderer.build_generation_prompt(p) if hasattr(client, "renderer") else \
                      tinker.types.ModelInput.from_ints(self.renderer.tokenizer.encode(p[0]["content"]))
                tasks.append(client.sample_async(
                    prompt=inp,
                    sampling_params=tinker.types.SamplingParams(max_tokens=self.max_tokens, temperature=0.0),
                    num_samples=1
                ))

            responses = await asyncio.gather(*tasks)
            decoded: list[str] = []
            for response in responses:
                sample = response

                # Streaming responses require an explicit read before tokens are accessible.
                if hasattr(sample, "read") and callable(sample.read):
                    sample = await sample.read()

                # Some client implementations wrap the response behind a ``result`` attribute.
                inner = getattr(sample, "result", None)
                if inner and hasattr(inner, "sequences"):
                    sample = inner

                if not hasattr(sample, "sequences"):
                    raise RuntimeError("Sampling response missing sequences attribute after read().")

                decoded.append(self.renderer.tokenizer.decode(sample.sequences[0].tokens).strip())

            return decoded

        decoded = await generate(sampling_client, prompts)

        # Branch behavior by mode: classification (0/1) vs comparison (A/B)
        if self.dpo_mode == CLASSIFICATION_MODE:
            # extract binary digits and compute precision/recall/F1
            preds = [ (re.findall(r"[01]", d)[0] if re.findall(r"[01]", d) else "") for d in decoded ]

            tp = sum(1 for p, t in zip(preds, ground_truths) if p == "1" and t == "1")
            fp = sum(1 for p, t in zip(preds, ground_truths) if p == "1" and t == "0")
            fn = sum(1 for p, t in zip(preds, ground_truths) if p == "0" and t == "1")
            tn = sum(1 for p, t in zip(preds, ground_truths) if p == "0" and t == "0")
            un_resolved = sum(1 for p in preds if p not in ["0", "1"])  # non-binary outputs

            total = tp + fp + fn + tn + un_resolved
            accuracy = (tp + tn) / total if total > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            return {
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_F1": F1,
            }
        else:
            # Comparison mode: expect 'A' or 'B' outputs. Only accuracy is meaningful.
            preds = [ (re.findall(r"[AB]", d, flags=re.IGNORECASE)[0].upper() if re.findall(r"[AB]", d, flags=re.IGNORECASE) else "") for d in decoded ]
            correct = sum(1 for p, t in zip(preds, ground_truths) if p == t)
            accuracy = correct / len(preds) if preds else 0.0
            return {"test_accuracy": accuracy}
    

@chz.chz
class NoveltyRankEvaluatorBuilder(EvaluatorBuilder):
    dataset_path: str 
    renderer_name: str
    model_name: str
    max_tokens: int = 10
    
    # Mode selection
    dpo_mode: str = COMPARISION_MODE

    # Category configuration
    category: str = WHOLE_DATASET
    category_outdir: str = "data_cache"
    category_seed: Optional[int] = None
    train_cache_path: str = DEFAULT_TRAIN_CACHE_DIR
    test_cache_path: str = DEFAULT_TEST_CACHE_DIR
    include_similarity_report: bool = chz.field(default=False)
    include_similarity_report: bool = chz.field(default=False)

    def __call__(self) -> Evaluator:
        need_classification = self.dpo_mode == CLASSIFICATION_MODE
        mode_name = "Classification (1/0)" if need_classification else "Comparison (A/B)"
        train_split, test_split = ensure_base_sft_splits(
            self.dataset_path,
            train_cache_dir=self.train_cache_path,
            test_cache_dir=self.test_cache_path,
        )
        paths = ensure_category_resources(
            category=self.category,
            dataset_path=self.dataset_path,
            train_split=train_split,
            test_split=test_split,
            outdir=self.category_outdir,
            seed=self.category_seed,
            need_sft=True,
            need_classification=need_classification,
            need_comparison=not need_classification,
        )

        if need_classification:
            logger.info("Loading classification test split from %s", paths.test_sft)
            test_sft = load_from_disk(paths.test_sft)
            def _map_eval(example: Dict[str, Any]) -> Dict[str, Any]:
                return create_classification_example(
                    example,
                    include_similarity_report=self.include_similarity_report,
                )

            test_dpo_pairs = test_sft.map(
                _map_eval,
                remove_columns=test_sft.column_names,
                desc="Preparing classification evaluation pairs",
            )
        else:
            target_cache_path = paths.test_comparison
            logger.info("Loading comparison test cache from %s", target_cache_path)
            test_dpo_pairs = load_from_disk(target_cache_path)
        
        # only take 1000 expamples for evaluation speed
        test_dpo_pairs = test_dpo_pairs.select(range(min(1000, len(test_dpo_pairs))))
        logger.info(f"Loaded {len(test_dpo_pairs)} test DPO pairs ({mode_name}) for evaluation.")
        
        return NoveltyRankAccuracyEvaluator(
            test_dataset=test_dpo_pairs,
            renderer_name=self.renderer_name,
            tokenizer_model_name=self.model_name,
            max_tokens=self.max_tokens,
            dpo_mode=self.dpo_mode,
        )

# -----------------------------------------------------------------------------
# Dataset Builder
# -----------------------------------------------------------------------------

@chz.chz
class NoveltyRankDatasetLoader(ChatDatasetBuilder):
    common_config: ChatDatasetBuilderCommonConfig
    dataset_path: str
    
    # Mode selection
    dpo_mode: str = COMPARISION_MODE # or CLASSIFICATION_MODE

    # Category configuration
    category: str = WHOLE_DATASET
    category_outdir: str = "data_cache"
    category_seed: Optional[int] = None
    train_cache_path: str = DEFAULT_TRAIN_CACHE_DIR
    test_cache_path: str = DEFAULT_TEST_CACHE_DIR
    include_similarity_report: bool = False

    def __call__(self) -> Tuple[SupervisedDataset, Optional[SupervisedDataset]]:
        need_classification = self.dpo_mode == CLASSIFICATION_MODE
        mode_name = "Classification (1/0)" if need_classification else "Comparison (A/B)"
        train_split, test_split = ensure_base_sft_splits(
            self.dataset_path,
            train_cache_dir=self.train_cache_path,
            test_cache_dir=self.test_cache_path,
        )
        paths = ensure_category_resources(
            category=self.category,
            dataset_path=self.dataset_path,
            train_split=train_split,
            test_split=test_split,
            outdir=self.category_outdir,
            seed=self.category_seed,
            need_sft=True,
            need_classification=need_classification,
            need_comparison=not need_classification,
        )
        if need_classification:
            if self.include_similarity_report:
                logger.info(
                    "Generating classification train pairs with similarity reports from %s",
                    paths.train_sft,
                )
                base_train = load_from_disk(paths.train_sft)
                train_dpo_pairs = generate_classification_dpo_pairs(
                    base_train,
                    include_similarity_report=True,
                )
            else:
                target_cache_path = paths.train_classification
                logger.info("Loading classification train cache from %s", target_cache_path)
                train_dpo_pairs = load_from_disk(target_cache_path)
        else:
            target_cache_path = paths.train_comparison
            logger.info("Loading comparison train cache from %s", target_cache_path)
            train_dpo_pairs = load_from_disk(target_cache_path)

        logger.info(f"Loaded {len(train_dpo_pairs)} DPO examples ({mode_name}) for training.")

        # Create the training dataset wrapper
        train_ds = NoveltyDPODataset(
            pairs=train_dpo_pairs, 
            tokenizer=self.tokenizer, 
            renderer=self.renderer,
            batch_size=self.common_config.batch_size
        )
        
        return train_ds, None