import logging
import asyncio
from dataclasses import dataclass
from typing import Sequence

import chz
from datasets import Dataset, load_dataset, concatenate_datasets

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
# Import required base classes for the RL trainer interface
from tinker_cookbook.rl.types import (
    RLDataset, 
    RLDatasetBuilder, 
    EnvGroupBuilder, 
    Env,
    Trajectory,
    Metrics
)
from tinker_cookbook.rl.preference_envs import PreferenceEnv
from tinker_cookbook.tokenizer_utils import get_tokenizer, Tokenizer

# Import your data logic
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
from models.Qwen_4B.sythesize_preference_pair import clean_dataset, generate_dpo_pairs_from_hf

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Adapter Classes (The "Fake" Environment Logic)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class NoveltyRankGroupBuilder(EnvGroupBuilder):
    """
    Adapts static DPO data to look like an RL Environment Group.
    This structure tells the RL trainer: "Here are two trajectories, one is Good, one is Bad."
    """
    prompt_conversation: list[renderers.Message]
    chosen: list[renderers.Message]
    rejected: list[renderers.Message]
    policy_renderer: renderers.Renderer 
    
    async def make_envs(self) -> Sequence[Env]:
        """
        Creates TWO environments for each data point:
        1. An env containing the CHOSEN response.
        2. An env containing the REJECTED response.
        
        The RL trainer will iterate through these. In Offline mode, it takes the 
        pre-filled messages as the 'trajectory'.
        """
        # Env 0: Holds the Chosen response
        env_chosen = PreferenceEnv(self.prompt_conversation, self.policy_renderer)
        # Hack: We pre-inject the response so the "rollout" just returns this static data
        # Note: Depending on Tinker version, PreferenceEnv might need extra setup to accept static data,
        # but typically passing the prompt is enough for the dataset wrapper.
        
        # Env 1: Holds the Rejected response
        env_rejected = PreferenceEnv(self.prompt_conversation, self.policy_renderer)
        
        return [env_chosen, env_rejected]

    async def compute_group_rewards(self, trajectory_group: list[Trajectory]) -> list[tuple[float, Metrics]]:
        """
        Calculates the reward for the group.
        Since we established that:
          - Index 0 corresponds to CHOSEN
          - Index 1 corresponds to REJECTED
          
        We assign a positive reward to index 0 and negative to index 1.
        This difference (1.0 - (-1.0) = 2.0) provides the gradient signal 
        that "Chosen > Rejected".
        """
        # Sanity check: ensure we have exactly 2 trajectories (Chosen vs Rejected)
        if len(trajectory_group) != 2:
            logger.warning(f"Expected 2 trajectories (Chosen/Rejected), got {len(trajectory_group)}")
            return [(0.0, {}) for _ in trajectory_group]

        # Reward Scheme:
        # Chosen (Index 0): Reward = 1.0
        # Rejected (Index 1): Reward = -1.0
        # The exact values matter less than the sign and difference.
        
        rewards_and_metrics = [
            (1.0, {"score": 1.0}),   # Index 0
            (-1.0, {"score": -1.0}) # Index 1
        ]
        
        return rewards_and_metrics


# -----------------------------------------------------------------------------
# 2. Dataset Classes
# -----------------------------------------------------------------------------

@chz.chz
class HFNoveltyRankDatasetBuilder:
    model_name_for_tokenizer: str
    renderer_name: str

    @property
    def tokenizer(self) -> Tokenizer:
        return get_tokenizer(self.model_name_for_tokenizer)

    @property
    def renderer(self) -> renderers.Renderer:
        return renderers.get_renderer(self.renderer_name, self.tokenizer)


class NoveltyRankDPODataset(RLDataset):
    """
    Wraps the dataset to return NoveltyRankGroupBuilder objects
    instead of raw examples.
    """
    def __init__(
        self, 
        batch_size: int,
        dataset_builder: HFNoveltyRankDatasetBuilder # Need renderer access
    ):
        self.batch_size = batch_size
        self.train_dataset = None
        self.dataset_builder = dataset_builder

    def set_dataset(self, train_dataset: Dataset):
        self.train_dataset = train_dataset

    def get_batch(self, index: int) -> list[NoveltyRankGroupBuilder]:
        """
        Returns a batch of GroupBuilders, satisfying rl.train's expectations.
        """
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        rows = self.train_dataset.select(range(start, end))
        
        batch_builders = []
        for row in rows:
            if row is None: continue
            
            # WRAP the static data into the Builder class
            builder = NoveltyRankGroupBuilder(
                prompt_conversation=row["prompt_conversation"],
                chosen=row["chosen"],
                rejected=row["rejected"],
                policy_renderer=self.dataset_builder.renderer
            )
            batch_builders.append(builder)
            
        return batch_builders

    def __len__(self) -> int:
        if self.train_dataset is None: return 0
        return len(self.train_dataset) // self.batch_size


@chz.chz
class NoveltyRankDPODatasetBuilder(RLDatasetBuilder):
    """
    Config builder.
    """
    comparison_dataset_builder: HFNoveltyRankDatasetBuilder # Add this field
    batch_size: int

    async def __call__(self) -> tuple[NoveltyRankDPODataset, None]:
        # We pass the builder helper to the dataset so it can access the renderer
        dpo_dataset = NoveltyRankDPODataset(
            batch_size=self.batch_size,
            dataset_builder=self.comparison_dataset_builder
        )
        return dpo_dataset, None


@chz.chz
class NoveltyRankDatasetLoader(HFNoveltyRankDatasetBuilder):
    dataset_path: str = "JasonYan777/novelty-dataset"
    multiple_num: int = 1 

    def get_train_and_test_datasets(self) -> tuple[Dataset, Dataset | None]:
        logger.info(f"Loading HF dataset: {self.dataset_path}")
        dataset = load_dataset(self.dataset_path)
        
        train_raw = clean_dataset(dataset["train"])
        test_raw = clean_dataset(dataset["test"])

        logger.info("Converting to DPO pairs...")
        train_dpo = generate_dpo_pairs_from_hf(train_raw)
        test_dpo = generate_dpo_pairs_from_hf(test_raw)

        if self.multiple_num > 1:
            logger.info(f"Replicating train dataset {self.multiple_num} times...")
            train_dpo = concatenate_datasets([train_dpo] * self.multiple_num)
            # shuffle after replication
            train_dpo = train_dpo.shuffle(seed=42)

        print("-----------------------------------------------")
        print(f"Generated {len(train_dpo)} DPO comparison examples for training.")
        print(f"Generated {len(test_dpo)} DPO comparison examples for testing.")
        print("------------------------------------------------")

        return train_dpo, test_dpo

# -----------------------------------------------------------------------------
# 3. Evaluator (Unchanged)
# -----------------------------------------------------------------------------

class NoveltyRankAccuracyEvaluator(SamplingClientEvaluator):
    def __init__(self, test_dataset: Dataset, renderer_name: str, tokenizer_model_name: str, max_tokens: int):
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