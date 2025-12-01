"""DPO classification entrypoint for the similarity-aware CV dataset."""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import chz
from dotenv import load_dotenv
from tinker_cookbook import model_info
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

# Ensure repo root when running as script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from new_model.models.dpo_env import (  # pylint: disable=wrong-import-position
    DEFAULT_TEST_DATASET_PATH,
    DEFAULT_TRAIN_DATASET_PATH,
    NoveltyRankDatasetLoader,
    NoveltyRankEvaluatorBuilder,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@chz.chz
class NoveltyDPOConfig:
    """Hyperparameters for similarity-aware DPO classification."""

    model_name: str = "tinker://2e566352-bd3a-5c10-8e5a-2743d49bc353:train:0/weights/final"
    reference_model_name: Optional[str] = None

    load_checkpoint_path: Optional[str] = None
    log_path: str = "results/new_model_dpo_qwen4b_classification"

    learning_rate: float = 5e-5
    batch_size: int = 128
    num_epochs: int = 5
    dpo_beta: float = 0.1
    lora_rank: int = 32

    max_length: int = 1024
    renderer_name: Optional[str] = None

    train_dataset_path: str = DEFAULT_TRAIN_DATASET_PATH
    test_dataset_path: str = DEFAULT_TEST_DATASET_PATH

    base_url: Optional[str] = None
    wandb_project: str = "NoveltyRank"
    wandb_name: str = "dpo_qwen4b_cv_similarity"


def main(config: NoveltyDPOConfig) -> None:
    load_dotenv()
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(config.model_name)

    dataset_builder = NoveltyRankDatasetLoader(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=config.model_name,
            renderer_name=renderer_name,
            max_length=config.max_length,
            batch_size=config.batch_size,
        ),
        train_dataset_path=config.train_dataset_path,
    )

    evaluator_builder = NoveltyRankEvaluatorBuilder(
        test_dataset_path=config.test_dataset_path,
        renderer_name=renderer_name,
        model_name=config.model_name,
        max_tokens=10,
    )

    dpo_config = train_dpo.Config(
        log_path=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        model_name=config.model_name,
        reference_model_name=config.reference_model_name,
        learning_rate=config.learning_rate,
        lora_rank=config.lora_rank,
        num_epochs=config.num_epochs,
        dpo_beta=config.dpo_beta,
        dataset_builder=dataset_builder,
        evaluator_builders=[evaluator_builder],
        base_url=config.base_url,
        load_checkpoint_path=config.load_checkpoint_path,
        save_every=10,
        eval_every=10,
    )

    logger.info("Starting similarity-aware DPO classification for %s", config.model_name)
    train_dpo.main(dpo_config)


if __name__ == "__main__":
    chz.entrypoint(main)
