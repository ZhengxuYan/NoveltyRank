import asyncio
import logging
import os
import sys
import argparse
from dotenv import load_dotenv
import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)
from models.Qwen_4B.data_access import (
    CATEGORY_SUBDIR_DEFAULT,
    CS_CV,
    CS_RO,
    DATA_VARIANT_SIM,
    TASK_CLASSIFICATION,
    TASK_COMPARISON,
    WHOLE_DATASET,
)
from models.Qwen_4B.sft_env import (
    NoveltyRankSFTDataBuilder,
    AccuracyOnLabeledTestSetEvaluator,
    ComparisonPairwiseAccuracyEvaluator,
)

logger = logging.getLogger(__name__)
    
def build_config(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    log_path: str = "results/noveltyrank_sft_qwen4b_cv",
    max_length: int = 4096,
    learning_rate: float = 2e-4,
    batch_size: int = 64,
    num_epochs: int = 10,
    eval_every: int = 24,
    save_every: int = 24,
    wandb_project: str = "NoveltyRank",
    wandb_name: str = "sft_qwen_4b_cv",
    category: str = WHOLE_DATASET,
    data_variant: str = DATA_VARIANT_SIM,
    category_subdir: str = CATEGORY_SUBDIR_DEFAULT,
    balance_dataset: bool = False,
    eval_sample_limit: int = 1000,
    sft_task: str = TASK_CLASSIFICATION,
) -> train.Config:
    if sft_task not in {TASK_CLASSIFICATION, TASK_COMPARISON}:
        raise ValueError(f"Unsupported sft_task '{sft_task}'")

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    include_similarity_report = (data_variant == "sim")
    dataset = NoveltyRankSFTDataBuilder(
        common_config=common_config,
        category=category,
        data_variant=data_variant,
        category_subdir=category_subdir,
        balance_dataset=balance_dataset,
        include_similarity_report=include_similarity_report,
        sft_task=sft_task,
    )

    evaluator_builders = []
    if sft_task == TASK_CLASSIFICATION:
        def accuracy_eval_builder():
            return AccuracyOnLabeledTestSetEvaluator(
                model_name=model_name,
                max_tokens=16,
                category=category,
                data_variant=data_variant,
                category_subdir=category_subdir,
                sample_limit=eval_sample_limit,
                include_similarity_report=include_similarity_report,
            )

        evaluator_builders.append(accuracy_eval_builder)
    elif sft_task == TASK_COMPARISON:
        def comparison_eval_builder():
            return ComparisonPairwiseAccuracyEvaluator(
                model_name=model_name,
                max_tokens=8,
                category=category,
                data_variant=data_variant,
                category_subdir=category_subdir,
                sample_limit=eval_sample_limit,
            )

        evaluator_builders.append(comparison_eval_builder)

    return train.Config(
        log_path=log_path,
        model_name=model_name,
        dataset_builder=dataset,
        learning_rate=learning_rate,
        lr_schedule="linear",
        num_epochs=num_epochs,
        eval_every=eval_every,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        evaluator_builders=evaluator_builders,
        save_every=save_every,
    )


# --- MODIFIED MAIN FUNCTION ---
async def main(
    category: str = WHOLE_DATASET,
    category_seed: int | None = None,
    data_variant: str = DATA_VARIANT_SIM,
    category_subdir: str = CATEGORY_SUBDIR_DEFAULT,
    balance_dataset: bool = False,
    eval_sample_limit: int = 1000,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    num_epochs: int | None = None,
    eval_every: int | None = None,
    save_every: int | None = None,
    sft_task: str = TASK_CLASSIFICATION,
    log_path: str | None = None,
    wandb_name: str | None = None,
):
    load_dotenv()
    # --- Data preparation: ensure all required splits and DPO pairs exist ---
    try:
        from models.Qwen_4B.data_preparation import prepare_data
        prep_type = "classification" if sft_task == TASK_CLASSIFICATION else "comparison"
        print(f"[PREP] Ensuring data for SFT ({prep_type}), data_variant={data_variant}")
        prepare_data_args = argparse.Namespace(
            data_variant=data_variant,
            type=prep_type
        )
        prepare_data(prepare_data_args)
    except Exception as e:
        print(f"[WARN] Data preparation step failed: {e}")

    # chz.entrypoint automatically handles configuration parsing before calling main
    config_kwargs = dict(
        category=category,
        data_variant=data_variant,
        category_subdir=category_subdir,
        balance_dataset=balance_dataset,
        eval_sample_limit=eval_sample_limit,
        sft_task=sft_task,
    )
    if learning_rate is not None:
        config_kwargs["learning_rate"] = learning_rate
    if batch_size is not None:
        config_kwargs["batch_size"] = batch_size
    if num_epochs is not None:
        config_kwargs["num_epochs"] = num_epochs
    if eval_every is not None:
        config_kwargs["eval_every"] = eval_every
    if save_every is not None:
        config_kwargs["save_every"] = save_every
    if category_seed is not None:
        logger.warning("category_seed is deprecated and ignored in the local pipeline.")
    if log_path is not None:
        config_kwargs["log_path"] = log_path
    if wandb_name is not None:
        config_kwargs["wandb_name"] = wandb_name

    config = build_config(**config_kwargs)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    await train.main(config) 


if __name__ == "__main__":
    asyncio.run(chz.entrypoint(main, allow_hyphens=True))
