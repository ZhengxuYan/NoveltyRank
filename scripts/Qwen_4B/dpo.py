import asyncio
import os
import chz
from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train # Keep using rl.train as requested
from tinker_cookbook.rl.train import AsyncConfig

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from models.Qwen_4B.dpo_env import (
    NoveltyRankDatasetLoader,
    NoveltyRankAccuracyEvaluator,
    NoveltyRankDPODatasetBuilder, # Import the Builder class
)

def build_config(
    model_name: str,
    dataset_path: str,
    log_path: str,
    max_length: int,
    learning_rate: float,
    batch_size: int,
    eval_every: int,
    multiple_num: int = 1,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> train.Config:
    
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    
    # 1. Load Data
    dataset_loader = NoveltyRankDatasetLoader(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        dataset_path=dataset_path,
        multiple_num = multiple_num,
    )

    train_ds, test_ds = dataset_loader.get_train_and_test_datasets()

    # 2. Configure Builder
    # We use the Builder class to construct the Dataset, injecting dependencies
    builder = NoveltyRankDPODatasetBuilder(
        comparison_dataset_builder=dataset_loader, # Inject loader/renderer info
        batch_size=batch_size,
    )

    # 3. Factory Closure
    # This matches the signature expected by Tinker: async function returning (Dataset, None)
    async def dataset_factory():
        # Call the builder to get the empty dataset wrapper
        dpo_dataset, _ = await builder()
        # Inject the actual data
        dpo_dataset.set_dataset(train_ds)
        # Return dataset and None (the RL trainer accepts None for the second arg in some offline paths)
        return dpo_dataset, None

    # 4. Evaluator
    evaluators = []
    if test_ds:
        def accuracy_eval_builder():
            return NoveltyRankAccuracyEvaluator(
                test_dataset=test_ds,
                renderer_name=renderer_name,
                tokenizer_model_name=model_name,
                max_tokens=16, 
            )
        evaluators.append(accuracy_eval_builder)

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=dataset_factory,
        learning_rate=learning_rate,
        max_tokens=max_length,
        eval_every=eval_every,
        evaluator_builders=evaluators,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )


def main(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    dataset_path: str = "JasonYan777/novelty-dataset",
    log_path: str = "results/novelty_rank_dpo_run",
    max_length: int = 2048, 
    learning_rate: float = 5e-7, 
    batch_size: int = 128, 
    eval_every: int = 10,
    multiple_num: int = 10, # How many times to replicate the dataset, replace epochs
    wandb_project: str = "NoveltyRank",
    wandb_name: str = "dpo_qwen_4b",
):
    load_dotenv()
    config = build_config(
        model_name=model_name,
        dataset_path=dataset_path,
        log_path=log_path,
        max_length=max_length,
        learning_rate=learning_rate,
        batch_size=batch_size,
        eval_every=eval_every,
        multiple_num=multiple_num,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    chz.entrypoint(main, allow_hyphens=True)