from typing import cast
from datasets import Dataset, load_dataset

import chz
import re
import asyncio
import tinker
from tinker import types
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from tinker_cookbook.supervised.types import ChatDatasetBuilder

from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers
from typing import List, Dict, Tuple
import torch


# clean the unexsited similarity entries and rescale similarity scores
def clean_dataset(dataset: Dataset) -> Dataset:
    """
    Clean the dataset by removing any unwanted or irrelevant examples.
    """
    # Implement your cleaning logic here
    dataset = dataset.filter(
        lambda example: example.get("max_similarity") not in [None, "null"]
        and example.get("avg_similarity") not in [None, "null"]
    )

    # Rescale similarity scores from 0-1 to min - max
    max_similarity_values = [float(item["max_similarity"]) for item in dataset]
    avg_similarity_values = [float(item["avg_similarity"]) for item in dataset]

    max_similarity = max(max_similarity_values) if max_similarity_values else 1
    min_similarity = min(avg_similarity_values) if avg_similarity_values else 0
    for item in dataset:
        item["max_similarity"] = (float(item["max_similarity"]) - min_similarity) / (max_similarity - min_similarity) if max_similarity != min_similarity else 0
        item["avg_similarity"] = (float(item["avg_similarity"]) - min_similarity) / (max_similarity - min_similarity) if max_similarity != min_similarity else 0

    return dataset

# =========================================================
# Create AccuracyOnLabeledTestSetEvaluator
# =========================================================
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

    def __init__(self, dataset_path: str, model_name: str, max_tokens: int = 16):
        dataset = load_dataset(dataset_path)
        self.train_data = clean_dataset(dataset["train"])
        self.test_data = clean_dataset(dataset["test"])

        self.model_name = model_name
        self.max_tokens = max_tokens

        # Create a renderer + tokenizer for encoding/decoding text
        self.tokenizer = get_tokenizer(model_name)
        self.service_client = tinker.ServiceClient()

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
            match = re.findall(r'[01]', decoded)
            decoded = match[0] if match else ""
            return decoded

        # Collect all user prompts and ground-truth labels
        prompts = []
        labels = []
        for ex in self.test_data:
            user_prompts, user_labels = generate_prompts_and_labels(ex)
            prompts.append(user_prompts)
            labels.append(user_labels)

        # Query the model for predictions
        tasks = [get_prediction(p) for p in prompts]
        predictions = await asyncio.gather(*tasks)

        # Compute accuracy, precision, recall
        tp = sum(1 for p, l in zip(predictions, labels) if p == "1" and l == "1")
        fp = sum(1 for p, l in zip(predictions, labels) if p == "1" and l == "0")
        fn = sum(1 for p, l in zip(predictions, labels) if p == "0" and l == "1")
        tn = sum(1 for p, l in zip(predictions, labels) if p == "0" and l == "0")
        un_resolved = sum(1 for p in predictions if p not in ["0", "1"])

        total = tp + fp + fn + tn + un_resolved
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"test_accuracy": accuracy, "test_precision": precision, "test_recall": recall, "test_F1": F1}


# ==========================================================
# Create SFTChatRenderer
# ==========================================================
class SFTChatRenderer:
    """
    A wrapper around a Chat Renderer (e.g., Qwen3InstructRenderer)
    for standard single-turn SFT tasks.
    Converts a conversation [{'role': 'user', ...}, {'role': 'assistant', ...}]
    into token/weight tensors usable by datum_from_tokens_weights.
    """

    def __init__(self, base_renderer: renderers.Renderer):
        self.base_renderer = base_renderer

    def to_tokens_weights(
        self, conversation: List[Dict[str, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a chat-style conversation (user + assistant) to token IDs and weights.

        conversation example:
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "1"}
        ]
        """
        tokens, weights = self.base_renderer.build_supervised_example(
            conversation,
            train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES  # train only on assistant replies
        )
        return tokens, weights

    @property
    def tokenizer(self):
        return self.base_renderer.tokenizer

# generate prompts for testing
def generate_prompts_and_labels(example: dict[str, str]) -> list[dict[str, str]]:
    title = example.get("Title", "")
    authors = example.get("Authors", "")
    abstract = example.get("Abstract", "")
    max_sim = example.get("max_similarity", "N/A")
    avg_sim = example.get("average_similarity", "N/A")
    label = example.get("label", "")
    
    # generate user prompt
    user_prompt = f"""
            You are an expert AI researcher and senior conference reviewer (NeurIPS/ICLR level). 
            Your goal is to **assess the conceptual novelty** of a new research paper,
            based on not only comparing surface similarity but also reasoning about the *conceptual depth* and *potential paradigm shift* introduced by the paper.
            
            ---
            ### Step-by-step reasoning:
            1. **Understand** the paper’s main idea, contribution, and context from its title and abstract.
            2. **Compare** it conceptually against prior literature (based on the given similarity metrics).
            3. **Evaluate** whether the paper represents:
               - Incremental improvement (minor variation),
               - A novel combination or extension,
               - Or a potentially *field-shaping* innovation that may start a new research direction.
            4. **Predict** whether this project is likely to have *high influence* or *define a new paradigm*.
            5. Finally, output a binary decision:
               - Output `'1'` if the paper is conceptually novel and likely to influence future research,
               - Output `'0'` otherwise.
            
            ---
            ### Input
            Title: {title}
            Authors: {authors}
            Abstract: {abstract}
            Max similarity to prior work: {max_sim}
            Average similarity to prior work: {avg_sim}
            
            ---
            ### Few-shot Examples
            
            **Example 1**
            Title: "Attention Is All You Need"
            Abstract: Introduces the Transformer architecture, replacing recurrence with attention mechanisms for sequence modeling.
            Max similarity: 0.68 | Avg similarity: 0.55  
            **Reasoning:** Although not completely dissimilar to prior seq2seq models, the conceptual innovation (self-attention, parallelization) defines a new paradigm.  
            **Output:** 1
            
            **Example 2**
            Title: "A Slightly Improved BERT Model with Layer-wise Dropout"
            Abstract: Modifies BERT with small regularization tweaks and hyperparameter tuning for better stability.
            Max similarity: 0.91 | Avg similarity: 0.82  
            **Reasoning:** Highly similar to prior work and lacks new conceptual insights.  
            **Output:** 0
            
            **Example 3**
            Title: "Aligning LLMs with Value-Constrained Reinforcement Learning"
            Abstract: Proposes a reinforcement learning approach with explicit value constraints to align LLM behavior.
            Max similarity: 0.73 | Avg similarity: 0.64  
            **Reasoning:** Builds on known RLHF methods but introduces a novel constrained optimization view; moderately novel.  
            **Output:** 1
            
            ---
            Now, reason through the given paper step by step and output only '1' or '0'.
            """
    return (user_prompt, str(label))
        

# ==========================================================
# Create NoveltyRankSFTDataBuilder
# ==========================================================
@chz.chz
class NoveltyRankSFTDataBuilder(ChatDatasetBuilder):
    """
    Build an SFT dataset for novelty ranking using a Hugging Face dataset.
    """
    dataset_path: str = chz.field(
        default="JasonYan777/novelty-ranking-dataset"
    )

    def __call__(self):
        # load the dataset from Hugging Face
        dataset = load_dataset(self.dataset_path)
        train_ds = cast(Dataset, clean_dataset(dataset["train"]))
        test_ds = cast(Dataset, clean_dataset(dataset["test"]))

        sft_renderer = SFTChatRenderer(self.renderer)

        # define function to convert each example to Datums
        def example_to_datum(example: dict[str, str]) -> list[types.Datum]:
            user_prompt, label = generate_prompts_and_labels(example)

            # convert to conversation format
            conversation = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": label}
            ]

            # convert conversation to Datum
            tokens, weights = sft_renderer.to_tokens_weights(conversation)
            datum = datum_from_tokens_weights(tokens, weights, self.common_config.max_length)
            return [datum]

        # transform datasets to SupervisedDatasets
        return (
            SupervisedDatasetFromHFDataset(
                train_ds,
                batch_size=self.common_config.batch_size,
                flatmap_fn=example_to_datum,
            ),
            SupervisedDatasetFromHFDataset(
                test_ds,
                batch_size=self.common_config.batch_size,
                flatmap_fn=example_to_datum,
            ),
        )