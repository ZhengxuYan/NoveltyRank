from dotenv import load_dotenv
import chz
import asyncio
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook import model_info
from tinker_cookbook import cli_utils
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.renderers import TrainOnWhat

from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)
from models.Qwen_4B.sft_env import NoveltyRankSFTDataBuilder, AccuracyOnLabeledTestSetEvaluator
from models.Qwen_4B.utils.pipelines import WHOLE_DATASET, CS_RO, CS_CV

_KNOWN_SFT_CATEGORIES = (WHOLE_DATASET, CS_RO, CS_CV)
    
def build_config(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    log_path: str = "results/noveltyrank_sft_qwen4b_cv",
    dataset_path: str = "JasonYan777/novelty-rank-with-similarities",
    max_length: int = 4096,
    learning_rate: float = 2e-4,
    batch_size: int = 64,
    num_epochs: int = 10,
    eval_every: int = 24,
    wandb_project: str = "NoveltyRank",
    wandb_name: str = "sft_qwen_4b_cv",
    category: str = WHOLE_DATASET,
    category_outdir: str = "data_cache",
    category_seed: int | None = None,
    include_similarity_report: bool = False,
) -> train.Config:
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset = NoveltyRankSFTDataBuilder(
        dataset_path=dataset_path,
        common_config=common_config,
        category=category,
        category_outdir=category_outdir,
        category_seed=category_seed,
        include_similarity_report=include_similarity_report,
    )

    def accuracy_eval_builder():
        return AccuracyOnLabeledTestSetEvaluator(
            dataset_path=dataset_path,
            model_name=model_name,
            max_tokens=16,
            category=category,
            category_outdir=category_outdir,
            category_seed=category_seed,
            train_cache_path=dataset.local_cache_path,
            local_cache_path=dataset.test_cache_path,
            include_similarity_report=include_similarity_report,
        )

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
        evaluator_builders=[accuracy_eval_builder],
    )


# --- MODIFIED MAIN FUNCTION ---
async def main(
    category: str = WHOLE_DATASET,
    category_seed: int | None = None,
    include_similarity_report: bool = False,
    log_path: str | None = None,
    wandb_name: str | None = None,
):
    load_dotenv()
    # chz.entrypoint automatically handles configuration parsing before calling main
    config_kwargs = dict(
        category=category,
        category_seed=category_seed,
        include_similarity_report=include_similarity_report,
    )
    if log_path is not None:
        config_kwargs["log_path"] = log_path
    if wandb_name is not None:
        config_kwargs["wandb_name"] = wandb_name

    config = build_config(**config_kwargs)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    await train.main(config) 


if __name__ == "__main__":
    asyncio.run(chz.entrypoint(main, allow_hyphens=True))
