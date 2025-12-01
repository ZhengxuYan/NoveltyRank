# __init__.py
"""
Initialize model samplers and handle prompt construction logic.
"""
import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.Qwen_4B.sft_env import create_sft_example, clean_dataset
import asyncio
from datasets import Dataset, load_dataset, load_from_disk

import re
import tinker
from dotenv import load_dotenv
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Initialize Tinker Service Client
load_dotenv()
service_client = tinker.ServiceClient()

# Define TinkerSampler class
class TinkerSampler():
    """A simple wrapper around Tinker ServiceClient to do sampling."""
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,  # tinker://..., obtained from Tinker training job
        temperature: float = 0.9,
        max_tokens=1024,
        top_p=1,
        top_k=-1,  # -1 means no limit
    ):
        tokenizer = get_tokenizer(model_name)
        renderer_name = get_recommended_renderer_name(model_name)
        # Read https://tinker-docs.thinkingmachines.ai/rendering to understand what renderer is
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
        self.sampling_params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=self.renderer.get_stop_sequences(),
        )
        self.sampling_client = service_client.create_sampling_client(
            model_path=model_path,
            base_model=model_name,
        )
        
    async def generate(self, messages: list[renderers.Message]) -> renderers.Message:
        prompt = self.renderer.build_generation_prompt(messages) #tokens is inside
        sampling_result = await self.sampling_client.sample_async(
            prompt=prompt,
            sampling_params=self.sampling_params,
            num_samples=1
        )
        response_tokens = sampling_result.sequences[0].tokens 

        raw_text_output = self.renderer.tokenizer.decode(response_tokens)
        response_message = {'role': 'assistant', 'content': raw_text_output}
        response_message['content'] = response_message['content'].strip()

        return response_message
    
async def process_one(data, sampler):
    user_prompt, label = create_sft_example(data)
    messages = [
        renderers.Message(role="user", content=user_prompt)
    ]
    prediction = await sampler.generate(messages)
    prediction = prediction['content']
    # Extract "0" or "1" from prediction using regex
    match = re.search(r'\b(0|1)\b', prediction)
    prediction = match[0] if match else ""
    return prediction, label

async def main():
    # --- Configuration ---
    dataset_path = "JasonYan777/novelty-rank-with-similarities"
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="WHOLE_DATASET",
                        help="Data category to evaluate (e.g., CS_CV, CS_RO, WHOLE_DATASET)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base model name registered with Tinker")
    parser.add_argument("--model-path", type=str,
                        default="tinker://2e566352-bd3a-5c10-8e5a-2743d49bc353:train:0/sampler_weights/final",
                        help="Tinker checkpoint URI (tinker://...) for sampling; leave empty for base model")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=10,
                        help="Maximum tokens to sample per response")
    parser.add_argument("--limit", type=int, default=500,
                        help="Number of evaluation examples to sample")
    args = parser.parse_args()

    category = args.category
    model_name = args.model_name
    model_path = args.model_path or None
    temperature = args.temperature
    max_tokens = args.max_tokens
    dataset_limit = max(1, args.limit)

    local_cache_path = "data_cache/whole_dataset/test_sft_data/test_split_cleaned"
    # If a category-specific cleaned SFT split exists, prefer it
    category_sft_path = f"data_cache/categories/{category}/sft/test"
    if os.path.exists(category_sft_path):
        local_cache_path = category_sft_path
    
    # --- Data Loading with Caching Feature ---
    
    if os.path.exists(local_cache_path):
        print(f"Local cleaned cache found at {local_cache_path}. Loading from disk...")
        # Load the single cleaned split directly
        dataset = load_from_disk(local_cache_path)
    else:
        print(f"No local cache found. Downloading {dataset_path} from web...")
        
        # 1. Download the raw dataset (synchronous operation)
        dataset_raw = load_dataset(dataset_path, split="test")
        
        # 2. Clean the dataset
        print("Downloading complete. Starting data cleaning...")
        cleaned_dataset = clean_dataset(dataset_raw)
        
        # 3. Save the cleaned result to disk for next time (synchronous operation)
        print(f"Cleaning complete, saving cleaned dataset to: {local_cache_path}...")
        cleaned_dataset.save_to_disk(local_cache_path)
        
        dataset = cleaned_dataset

    # --- End of main function logic ---
    print(f"Successfully loaded and cleaned dataset of size: {len(dataset)}")
    # shuffle and limit dataset size for testing
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(min(dataset_limit, len(dataset))))

    # Initialize sampler
    print("-------- Initializing sampler -------")
    print(f"Model Name: {model_name}")
    print(f"Model Path: {model_path}")
    print(f"Temperature: {temperature}")
    print(f"Max Tokens: {max_tokens}")
    print("-------- Sampler initialized -------")
    sampler = TinkerSampler(model_name=model_name,
                            model_path=model_path,
                            temperature=temperature, max_tokens=max_tokens)

    # asyncio generate predictions
    results = await asyncio.gather(*[process_one(data, sampler) for data in dataset])
    predictions, labels = zip(*results)

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

if __name__ == "__main__":

    results = asyncio.run(main())
    print(results)