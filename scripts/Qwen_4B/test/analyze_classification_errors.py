import os
import sys
import json
import argparse
import asyncio
import re
from pathlib import Path

from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)
from models.Qwen_4B.sft_env import create_sft_example, clean_dataset  # noqa: E402
from tinker import types  # noqa: E402
from tinker_cookbook import renderers  # noqa: E402
from tinker_cookbook.model_info import get_recommended_renderer_name  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402
import tinker  # noqa: E402


load_dotenv()
service_client = tinker.ServiceClient()


class TinkerSampler:
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 10,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> None:
        tokenizer = get_tokenizer(model_name)
        renderer_name = get_recommended_renderer_name(model_name)
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
        prompt = self.renderer.build_generation_prompt(messages)
        sampling_result = await self.sampling_client.sample_async(
            prompt=prompt,
            sampling_params=self.sampling_params,
            num_samples=1,
        )
        response_tokens = sampling_result.sequences[0].tokens
        raw_text_output = self.renderer.tokenizer.decode(response_tokens)
        response_message = {"role": "assistant", "content": raw_text_output.strip()}
        return response_message


async def classify_one(
    example: dict,
    sampler: TinkerSampler,
    include_similarity_report: bool,
) -> tuple[str, str]:
    user_prompt, label = create_sft_example(
        example,
        include_similarity_report=include_similarity_report,
    )
    messages = [renderers.Message(role="user", content=user_prompt)]
    prediction = await sampler.generate(messages)
    match = re.search(r"\b(0|1)\b", prediction["content"])
    predicted_label = match.group(1) if match else ""
    return predicted_label, label


def load_split(category: str, dataset_path: str, seed: int) -> list[dict]:
    base_cache = "data_cache/whole_dataset/test_sft_data/test_split_cleaned"
    category_cache = f"data_cache/categories/{category}/sft/test"

    if os.path.exists(category_cache):
        dataset = load_from_disk(category_cache)
    elif os.path.exists(base_cache):
        dataset = load_from_disk(base_cache)
    else:
        raw = load_dataset(dataset_path, split="test")
        dataset = clean_dataset(raw)
        Path(base_cache).parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(base_cache)

    dataset = dataset.shuffle(seed=seed)
    return [dataset[i] for i in range(len(dataset))]


def ensure_output_path(path: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="CS_CV")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--model-path",
        type=str,
        default="tinker://2e566352-bd3a-5c10-8e5a-2743d49bc353:train:0/sampler_weights/001060",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="results/analysis/cs_cv_classification_error_examples.json",
    )
    parser.add_argument(
        "--include-similarity-report",
        action="store_true",
        help="Augment prompts with the precomputed similarity report",
    )
    args = parser.parse_args()

    dataset = load_split(args.category, "JasonYan777/novelty-rank-with-similarities", args.seed)
    if args.limit < len(dataset):
        dataset = dataset[: args.limit]

    sampler = TinkerSampler(
        model_name=args.model_name,
        model_path=args.model_path or None,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    predictions = await asyncio.gather(
        *[
            classify_one(
                example,
                sampler,
                include_similarity_report=args.include_similarity_report,
            )
            for example in dataset
        ]
    )

    fp_examples: list[dict] = []
    fn_examples: list[dict] = []

    tp = fp = fn = tn = unresolved = 0
    for row, (predicted, label) in zip(dataset, predictions):
        label = str(label)
        predicted = str(predicted)
        record = {
            "title": row.get("Title", ""),
            "abstract": row.get("Abstract", ""),
            "true_label": label,
            "predicted_label": predicted,
            "similarity_score": {
                "max": row.get("max_similarity"),
                "avg": row.get("avg_similarity"),
            },
        }

        if predicted not in {"0", "1"}:
            unresolved += 1
            continue
        if label == "1" and predicted == "1":
            tp += 1
        elif label == "0" and predicted == "0":
            tn += 1
        elif label == "0" and predicted == "1":
            fp += 1
            if len(fp_examples) < 3:
                fp_examples.append(record)
        elif label == "1" and predicted == "0":
            fn += 1
            if len(fn_examples) < 3:
                fn_examples.append(record)

    total = tp + tn + fp + fn + unresolved
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    output_payload = {
        "category": args.category,
        "model_name": args.model_name,
        "model_path": args.model_path,
        "temperature": args.temperature,
        "limit": len(dataset),
        "include_similarity_report": args.include_similarity_report,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "unresolved": unresolved,
        },
        "false_positives": fp_examples,
        "false_negatives": fn_examples,
    }

    output_path = ensure_output_path(args.output)
    output_path.write_text(json.dumps(output_payload, indent=2))

    print(json.dumps(output_payload["metrics"], indent=2))
    def print_case(header: str, cases: list[dict]) -> None:
        print(f"\n{header}")
        if not cases:
            print("  (none)")
            return
        for idx, item in enumerate(cases, start=1):
            print(f"[{idx}] Title: {item['title']}")
            print("    Abstract:")
            print(f"      {item['abstract']}")
            print(
                f"    Labels: predicted={item['predicted_label']} / true={item['true_label']}"
            )
            sim = item.get("similarity_score", {})
            print(
                f"    Similarity Score: max={sim.get('max')} avg={sim.get('avg')}"
            )

    print_case("False Positives (first 3):", fp_examples)
    print_case("False Negatives (first 3):", fn_examples)


if __name__ == "__main__":
    asyncio.run(main())
