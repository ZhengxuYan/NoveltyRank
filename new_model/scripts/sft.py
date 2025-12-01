"""SFT entrypoint for the similarity-aware CV dataset."""
from __future__ import annotations

import asyncio
import os
import sys

import chz
from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

# Ensure repo root is importable when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from new_model.models.sft_env import (  # pylint: disable=wrong-import-position
    DEFAULT_TEST_DATASET_PATH,
    DEFAULT_TRAIN_DATASET_PATH,
    AccuracyOnLabeledTestSetEvaluator,
    NoveltyRankSFTDataBuilder,
)


def build_config(
    *,
    model_name: str,
    log_path: str,
    train_dataset_path: str,
    test_dataset_path: str,
    max_length: int,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    eval_every: int,
    wandb_project: str,
    wandb_name: str,
) -> train.Config:
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset_builder = NoveltyRankSFTDataBuilder(
        common_config=common_config,
        train_dataset_path=train_dataset_path,
    )

    def eval_builder() -> AccuracyOnLabeledTestSetEvaluator:
        return AccuracyOnLabeledTestSetEvaluator(
            model_name=model_name,
            test_dataset_path=test_dataset_path,
            max_tokens=16,
        )

    return train.Config(
        log_path=log_path,
        model_name=model_name,
        dataset_builder=dataset_builder,
        learning_rate=learning_rate,
        lr_schedule="linear",
        num_epochs=num_epochs,
        eval_every=eval_every,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        evaluator_builders=[eval_builder],
    )


async def main(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    log_path: str = "results/new_model_sft_qwen4b_cv",
    train_dataset_path: str = DEFAULT_TRAIN_DATASET_PATH,
    test_dataset_path: str = DEFAULT_TEST_DATASET_PATH,
    max_length: int = 4096,
    learning_rate: float = 2e-4,
    batch_size: int = 64,
    num_epochs: int = 10,
    eval_every: int = 24,
    wandb_project: str = "NoveltyRank",
    wandb_name: str = "sft_qwen4b_cv_similarity",
) -> None:
    load_dotenv()
    config = build_config(
        model_name=model_name,
        log_path=log_path,
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        max_length=max_length,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        eval_every=eval_every,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    await train.main(config)


if __name__ == "__main__":
    asyncio.run(chz.entrypoint(main, allow_hyphens=True))
