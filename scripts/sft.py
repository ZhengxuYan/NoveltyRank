import os
import sys
import asyncio
import json
import chz
import wandb
import numpy as np
import tinker

from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer


# ========== Evaluator Definition ==========
class AccuracyOnLabeledTestSetEvaluator(SamplingClientEvaluator):
    """
    Evaluate model accuracy on a labeled test set.
    The test set contains user prompts and assistant responses ("0" or "1") as ground-truth labels.

    Each test example:
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "1"}
        ]
    }

    This evaluator feeds the user message to the model, 
    generates a prediction ("0" or "1"), 
    and computes accuracy = (# of correct predictions / total examples).
    """

    def __init__(self, test_dataset_path: str, model_name: str, max_tokens: int = 16):
        self.test_dataset_path = test_dataset_path
        self.model_name = model_name
        self.max_tokens = max_tokens

        # Create a renderer + tokenizer for encoding/decoding text
        self.tokenizer = get_tokenizer(model_name)
        self.service_client = tinker.ServiceClient()

        # Load test dataset
        with open(test_dataset_path, "r", encoding="utf-8") as f:
            self.test_data = [json.loads(line) for line in f]

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run evaluation asynchronously using the current policy sampling client.
        """

        async def get_prediction(prompt_text: str) -> str:
            """
            Query the model for a single prediction ("0" or "1").
            """
            response = await sampling_client.sample_async(
                prompt=tinker.types.ModelInput.from_ints(
                    self.tokenizer.encode(prompt_text)
                ),
                sampling_params=tinker.types.SamplingParams(
                    max_tokens=self.max_tokens, temperature=0.0
                ),
                num_samples=1,
            )
            decoded = self.tokenizer.decode(response.sequences[0].tokens).strip()
            # Only keep the first character if model outputs e.g. "1\n" or "1."
            return decoded[0] if decoded else ""

        # Collect all user prompts and ground-truth labels
        prompts = []
        labels = []
        for ex in self.test_data:
            messages = ex.get("messages", [])
            if len(messages) >= 2:
                user_msg = messages[0]["content"]
                label = messages[1]["content"].strip()
                prompts.append(user_msg)
                labels.append(label)

        # Query the model for predictions
        tasks = [get_prediction(p) for p in prompts]
        predictions = await asyncio.gather(*tasks)

        # Compute accuracy, precision, recall
        tp = sum(1 for p, l in zip(predictions, labels) if p == "1" and l == "1")
        fp = sum(1 for p, l in zip(predictions, labels) if p == "1" and l != "1")
        fn = sum(1 for p, l in zip(predictions, labels) if p != "1" and l == "1")
        tn = sum(1 for p, l in zip(predictions, labels) if p == "0" and l == "0")

        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {"test_accuracy": accuracy, "test_precision": precision, "test_recall": recall}


# ========== Build Config ==========
def build_config(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    train_dataset_path: str = "/home/yumingfeng/repo/NoveltyRank/dataset/data_train_prompts.jsonl",
    test_dataset_path: str = "/home/yumingfeng/repo/NoveltyRank/dataset/data_test_prompts.jsonl",
    log_path: str = "/home/yumingfeng/repo/NoveltyRank/results/sft_model_v3",
    learning_rate: float = 2e-4,
    num_epochs: int = 5,
    eval_every: int = 10,
    wandb_project: str = "NoveltyRank",
    wandb_name: str = "4b_eval_v3",
) -> train.Config:
    """
    Build the supervised fine-tuning (SFT) config with an accuracy-based evaluator.
    """

    # Prepare tokenizer/rendering configuration
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4096,
        batch_size=4096,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    # Build training dataset
    train_dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path=train_dataset_path,
    )

    # Define evaluator builder
    def accuracy_eval_builder():
        return AccuracyOnLabeledTestSetEvaluator(
            test_dataset_path=test_dataset_path,
            model_name=model_name,
            max_tokens=16,
        )

    # Assemble final training configuration
    return chz.Blueprint(train.Config).apply(
        {
            "log_path": log_path,
            "model_name": model_name,
            "dataset_builder": train_dataset,
            "learning_rate": learning_rate,
            "lr_schedule": "linear",
            "num_epochs": num_epochs,
            "eval_every": eval_every,
            "wandb_project": wandb_project,
            "wandb_name": wandb_name,
            "evaluator_builders": [accuracy_eval_builder],
        }
    ).make()


# ========== Main Entrypoint ==========
def main():
    load_dotenv()
    wandb.login()

    os.environ["WANDB_ENTITY"] = "230project"
    os.environ["WANDB_PROJECT"] = "NoveltyRank"

    config = build_config()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
