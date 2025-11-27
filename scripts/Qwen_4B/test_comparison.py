# __init__.py
"""
Initialize model samplers and handle prompt construction logic for comparison tasks.
"""
import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.Qwen_4B.utils import clean_dataset, generate_comparison_dpo_pairs
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
    # data is a row from the DPO pairs dataset
    messages = data["prompt_conversation"]
    label = data["chosen"][0]["content"]
    
    prediction = await sampler.generate(messages)
    prediction = prediction['content']
    
    # Extract "A" or "B" from prediction using regex
    match = re.search(r'\b(A|B)\b', prediction, re.IGNORECASE)
    prediction = match.group(0).upper() if match else ""
    
    return prediction, label

async def main():
    # --- Configuration ---
    dataset_path = "JasonYan777/novelty-rank-with-similarities"
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="WHOLE_DATASET",
                        help="Data category to build DPO pairs for (e.g., CS_CV, CS_RO, WHOLE_DATASET)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base model name registered with Tinker")
    parser.add_argument("--model-path", type=str,
                        default="tinker://c2cf9723-af77-5458-9032-a7f5b10b20da:train:0/sampler_weights/final",
                        help="Tinker checkpoint URI (tinker://...) for sampling; leave empty for base model")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=10,
                        help="Maximum tokens to sample per response")
    parser.add_argument("--limit", type=int, default=100,
                        help="Number of comparison examples to evaluate")
    args = parser.parse_args()
    category = args.category
    model_name = args.model_name
    model_path = args.model_path or None
    temperature = args.temperature
    max_tokens = args.max_tokens
    dataset_limit = max(1, args.limit)

    local_sft_cache_path = "data_cache/test_sft_data/test_split_cleaned"
    local_dpo_cache_path = "data_cache/test_dpo_pairs_comparison"

    # Paths for category-specific caches
    category_sft_path = f"data_cache/categories/{category}/sft/test"
    category_dpo_path = f"data_cache/categories/{category}/dpo/comparison/test"
    if os.path.exists(category_dpo_path):
        local_dpo_cache_path = category_dpo_path
    elif os.path.exists(category_sft_path):
        local_sft_cache_path = category_sft_path
    
    # --- Data Loading with Caching Feature ---
    if os.path.exists(local_dpo_cache_path):
        print(f"Local DPO comparison cache found at {local_dpo_cache_path}. Loading from disk...")
        dataset = load_from_disk(local_dpo_cache_path)
    else:
        print(f"No local DPO comparison cache found. Processing data...")
        if os.path.exists(local_sft_cache_path):
            print(f"Loading cleaned data from {local_sft_cache_path}...")
            cleaned_dataset = load_from_disk(local_sft_cache_path)
        else:
            print(f"No cleaned data cache found. Downloading {dataset_path} from web...")
            dataset_raw = load_dataset(dataset_path, split="test")
            
            print("Downloading complete. Starting data cleaning...")
            cleaned_dataset = clean_dataset(dataset_raw, filter_year=False)
            
            print(f"Cleaning complete, saving cleaned dataset to: {local_sft_cache_path}...")
            cleaned_dataset.save_to_disk(local_sft_cache_path)

        print("Generating comparison DPO pairs...")
        dataset = generate_comparison_dpo_pairs(cleaned_dataset)
        # If a category was specified, save into the category dpo path to persist
        save_path = local_dpo_cache_path if local_dpo_cache_path != "data_cache/test_dpo_pairs_comparison" else "data_cache/test_dpo_pairs_comparison"
        print(f"Saving DPO comparison dataset to: {save_path}...")
        dataset.save_to_disk(save_path)

    # --- End of main function logic ---
    print(f"Successfully loaded DPO comparison dataset of size: {len(dataset)}")
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

    # Compute accuracy
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0
    
    un_resolved = sum(1 for p in predictions if p not in ["A", "B"]) 

    print(f"Total: {total}, Correct: {correct}, Unresolved: {un_resolved}")

    return {"test_accuracy": accuracy}

if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)
