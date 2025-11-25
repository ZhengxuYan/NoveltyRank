import os
import sys
import logging
import chz
from typing import Optional

# 1. Setup paths to ensure local modules can be found
sys.path.append(os.getcwd())

# Tinker imports
from tinker_cookbook.preference import train_dpo
from tinker_cookbook import model_info
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

# Import our custom dataset loa
from models.Qwen_4B.dpo_env import NoveltyRankDatasetLoader, NoveltyRankEvaluatorBuilder

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    log_path: str = "results/noveltyrank_dpo_qwen4b_new"
    
    # Training Hyperparameters
    learning_rate: float = 5e-6
    batch_size: int = 128
    num_epochs: int = 10
    dpo_beta: float = 0.1
    lora_rank: int = 32
    
    # Data processing
    max_length: int = 1024
    renderer_name: Optional[str] = None  # Will auto-detect if None

    # Infrastructure
    base_url: Optional[str] = None  # For local/remote service URL
    wandb_project: str = "NoveltyRank"
    wandb_name: str = "DPO_qwen_4b_new"


# -----------------------------------------------------------------------------
# Main Execution Logic
# -----------------------------------------------------------------------------

def main(env_config: NoveltyDPOConfig):
    
    # dataset_path = "JasonYan777/novelty-dataset"
    dataset_path = "JasonYan777/novelty-rank-with-similarities"

    logger.info(f"Starting DPO training for model: {env_config.model_name}")

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
    
    dataset_builder = NoveltyRankDatasetLoader(
        common_config=common_ds_config,
        dataset_path=dataset_path
    )

    # 4. Initialize Evaluator Builder
    eval_builder = NoveltyRankEvaluatorBuilder(
        dataset_path=dataset_path,
        renderer_name=renderer_name,
        model_name=env_config.model_name,
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
        save_every=10,
        eval_every=10
    )

    # 6. Run Training
    train_dpo.main(dpo_config)

if __name__ == "__main__":
    chz.entrypoint(main)