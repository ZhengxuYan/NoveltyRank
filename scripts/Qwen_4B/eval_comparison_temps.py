#!/usr/bin/env python3
"""
Evaluate DPO comparison performance for models across temperatures.
Saves per-run metrics to `results/eval_comparison_<category>.jsonl`.

Usage example:
python scripts/Qwen_4B/eval_comparison_temps.py --category CS_CV \
    --models Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-235B-A22B-Instruct-2507 --temps 0.0,0.3,0.5,1.0 --max-samples 300
"""
import os
import sys
import argparse
import asyncio
import json
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker import types
from datasets import load_from_disk, load_dataset

from models.Qwen_4B.utils import clean_dataset, generate_comparison_dpo_pairs

load_dotenv()
service_client = tinker.ServiceClient()


class TinkerSampler:
    def __init__(self, model_name: str, model_path: str | None = None, temperature: float = 0.0, max_tokens: int = 16):
        tokenizer = get_tokenizer(model_name)
        renderer_name = get_recommended_renderer_name(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
        self.sampling_params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            top_k=-1,
            stop=self.renderer.get_stop_sequences(),
        )
        self.sampling_client = service_client.create_sampling_client(
            model_path=model_path,
            base_model=model_name,
        )

    async def generate(self, messages: List[renderers.Message]):
        prompt = self.renderer.build_generation_prompt(messages)
        sampling_result = await self.sampling_client.sample_async(
            prompt=prompt,
            sampling_params=self.sampling_params,
            num_samples=1,
        )
        response_tokens = sampling_result.sequences[0].tokens
        raw_text_output = self.renderer.tokenizer.decode(response_tokens)
        return raw_text_output.strip()


async def evaluate_model_on_dpo(model_name: str, model_path: str | None, temp: float, dataset, max_samples: int):
    sampler = TinkerSampler(model_name=model_name, model_path=model_path, temperature=temp, max_tokens=10)

    n = min(max_samples, len(dataset))

    predictions = []
    labels = []

    async def process_one(row):
        messages = row["prompt_conversation"]
        # tinker renderer expects list[renderers.Message] dict-like with role/content
        # messages are already in that format in dataset
        out = await sampler.generate(messages)
        import re
        m = re.search(r"\b(A|B)\b", out, re.IGNORECASE)
        pred = m.group(0).upper() if m else ""
        label = row["chosen"][0]["content"] if row.get("chosen") else ""
        return pred, label

    tasks = [process_one(dataset[i]) for i in range(n)]
    results = await asyncio.gather(*tasks)
    for pred, label in results:
        predictions.append(pred)
        labels.append(label)

    # compute metrics: for comparison/DPO we only report accuracy and unresolved
    total = len(predictions)
    unresolved = sum(1 for p in predictions if p not in ['A', 'B'])
    # accuracy: proportion of exact matches (A/B)
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "model": model_name,
        "model_path": model_path,
        "temperature": temp,
        "total": total,
        "unresolved": unresolved,
        "accuracy": accuracy,
    }


def load_dpo_dataset_for_category(category: str):
    # prefer already materialized comparison DPO dataset for category
    cat_dpo = f"data_cache/categories/{category}/dpo/comparison/test"
    if os.path.exists(cat_dpo):
        return load_from_disk(cat_dpo)
    # fallback to global cached dpo path
    global_dpo = "data_cache/test_dpo_pairs_comparison"
    if os.path.exists(global_dpo):
        return load_from_disk(global_dpo)
    # otherwise, build from cleaned sft test split
    cat_sft = f"data_cache/categories/{category}/sft/test"
    if os.path.exists(cat_sft):
        cleaned = load_from_disk(cat_sft)
    else:
        ds = load_dataset("JasonYan777/novelty-rank-with-similarities", split="test")
        cleaned = clean_dataset(ds)
    dpo = generate_comparison_dpo_pairs(cleaned)
    # save to category path for future
    save_path = f"data_cache/categories/{category}/dpo/comparison/test"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dpo.save_to_disk(save_path)
    return dpo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="WHOLE_DATASET")
    parser.add_argument("--models", type=str,
                        default="Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-235B-A22B-Instruct-2507",
                        help="Comma-separated list of model names to evaluate")
    parser.add_argument("--model-paths", type=str, default=None,
                        help="Optional comma-separated list of model_path URIs to pass to Tinker (aligns with --models)")
    parser.add_argument("--temps", type=str, default="0.0,0.3,0.5,1.0",
                        help="Comma-separated list of temperatures to evaluate")
    parser.add_argument("--max-samples", type=int, default=300,
                        help="Maximum number of test examples to evaluate per run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to shuffle the test dataset")
    parser.add_argument("--out", type=str, default=None, help="Output JSONL file path")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    model_paths = [p.strip() for p in args.model_paths.split(",")] if args.model_paths else [None] * len(models)
    temps = [float(x) for x in args.temps.split(",")]
    category = args.category
    max_samples = args.max_samples

    dpo = load_dpo_dataset_for_category(category)
    # shuffle for reproducibility
    try:
        dpo = dpo.shuffle(seed=args.seed)
    except Exception:
        pass
    print(f"Loaded DPO dataset for category {category}, size={len(dpo)}")

    out_path = args.out or f"results/eval_comparison_{category}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    loop = asyncio.get_event_loop()

    with open(out_path, "a") as fout:
        for model_name, model_path in zip(models, model_paths):
            for temp in temps:
                print(f"Evaluating model={model_name} temp={temp} max_samples={max_samples}")
                res = loop.run_until_complete(evaluate_model_on_dpo(model_name, model_path, temp, dpo, max_samples))
                print(json.dumps(res, indent=2))
                fout.write(json.dumps(res) + "\n")

    print("Done. Results appended to", out_path)


if __name__ == "__main__":
    main()
