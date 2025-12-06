import os
import sys
import logging
import argparse
import chz
from typing import Optional

# 1. Setup paths to ensure local modules can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Tinker imports
from dotenv import load_dotenv
from tinker_cookbook.preference import train_dpo
from tinker_cookbook import model_info
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

# Import our custom dataset loader
from models.Qwen_4B.dpo_env import (
    NoveltyRankDatasetLoader,
    NoveltyRankEvaluatorBuilder,
    CLASSIFICATION_MODE,
    COMPARISION_MODE,
)
from models.Qwen_4B.data_access import (
    CATEGORY_SUBDIR_DEFAULT,
    CS_CV,
    CS_RO,
    DATA_VARIANT_BASE,
    DATA_VARIANT_SIM,
    WHOLE_DATASET,
)

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRENT_MODE = COMPARISION_MODE  # Change to COMPARISION_MODE for pairwise DPO
# -----------------------------------------------------------------------------
# Configuration Section
# -----------------------------------------------------------------------------

@chz.chz
class NoveltyDPOConfig:
    """
    Configuration specifically for the Novelty Rank DPO experiment.
    This acts as the hyperparameter definition.
    """
    # Core Model settings
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    reference_model_name: Optional[str] = None  # If None, uses model_name (weights frozen)

    # Checkpointing
    load_checkpoint_path: Optional[str] = None
    log_path: str = f"results/noveltyrank_dpo_qwen4b_{CURRENT_MODE}"

    # Training Hyperparameters
    learning_rate: float = 5e-6
    batch_size: int = 128
    num_epochs: int = 5
    dpo_beta: float = 0.1
    lora_rank: int = 32
    save_every: int = 10
    eval_every: int = 10

    # Data processing
    max_length: int = 4096
    renderer_name: Optional[str] = None  # Will auto-detect if None

    # DPO Mode Selection: "comparison" (Pairwise A/B) or "classification" (Pointwise 1/0)
    dpo_mode: str = CURRENT_MODE
    category: str = WHOLE_DATASET
    data_variant: str = DATA_VARIANT_BASE
    classification_variant: str = DATA_VARIANT_SIM
    comparison_variant: str = DATA_VARIANT_BASE
    category_subdir: str = CATEGORY_SUBDIR_DEFAULT
    eval_sample_limit: int = 1000

    # Infrastructure
    base_url: Optional[str] = None  # For local/remote service URL
    wandb_project: str = "NoveltyRank"
    wandb_name: str = f"DPO_qwen_4b_{CURRENT_MODE}"


# -----------------------------------------------------------------------------
# Main Execution Logic
# -----------------------------------------------------------------------------

def main(env_config: NoveltyDPOConfig):
    # --- Data preparation: ensure all required splits and DPO pairs exist ---
    try:
        from models.Qwen_4B.data_preparation import prepare_data
        prep_type = env_config.dpo_mode if env_config.dpo_mode in ["classification", "comparison"] else "comparison"
        print(f"[PREP] Ensuring data for DPO ({prep_type}), data_variant={env_config.data_variant}")
        prepare_data_args = argparse.Namespace(
            data_variant=env_config.data_variant,
            type=prep_type
        )
        prepare_data(prepare_data_args)
    except Exception as e:
        print(f"[WARN] Data preparation step failed: {e}")

    logger.info(f"Starting DPO training for model: {env_config.model_name}")
    logger.info(f"DPO Mode: {env_config.dpo_mode}")

    # 2. Determine Renderer
    renderer_name = env_config.renderer_name or model_info.get_recommended_renderer_name(
        env_config.model_name
    )

    # 3. Initialize Dataset Builder
    common_ds_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=env_config.model_name,
        renderer_name=renderer_name,
        max_length=env_config.max_length,
        batch_size=env_config.batch_size
    )
    
    include_similarity_report = (env_config.data_variant == "sim")
    dataset_builder = NoveltyRankDatasetLoader(
        common_config=common_ds_config,
        dpo_mode=env_config.dpo_mode,
        category=env_config.category,
        classification_variant=env_config.classification_variant,
        comparison_variant=env_config.comparison_variant,
        category_subdir=env_config.category_subdir,
        include_similarity_report=include_similarity_report,
    )

    # 4. Initialize Evaluator Builder
    eval_builder = NoveltyRankEvaluatorBuilder(
        renderer_name=renderer_name,
        model_name=env_config.model_name,
        dpo_mode=env_config.dpo_mode,
        category=env_config.category,
        classification_variant=env_config.classification_variant,
        comparison_variant=env_config.comparison_variant,
        category_subdir=env_config.category_subdir,
        include_similarity_report=include_similarity_report,
        eval_sample_limit=env_config.eval_sample_limit,
    )

    # 5. Construct the Main DPO Training Configuration
    dpo_config = train_dpo.Config(
        # Output & Logging
        log_path=env_config.log_path,
        wandb_project=env_config.wandb_project,
        wandb_name=env_config.wandb_name,
        
        # Model & Optimization
        model_name=env_config.model_name,
        reference_model_name=env_config.reference_model_name,
        learning_rate=env_config.learning_rate,
        lora_rank=env_config.lora_rank,
        num_epochs=env_config.num_epochs,
        dpo_beta=env_config.dpo_beta,
        
        # Data & Eval
        dataset_builder=dataset_builder,
        evaluator_builders=[eval_builder],
        
        # Infrastructure
        base_url=env_config.base_url,
        load_checkpoint_path=env_config.load_checkpoint_path,
        
        # Frequency settings
        save_every=env_config.save_every,
        eval_every=env_config.eval_every,
    )

    # 6. Run Training
    train_dpo.main(dpo_config)

if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file if present
    chz.entrypoint(main)