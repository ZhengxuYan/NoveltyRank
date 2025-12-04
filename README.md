# NoveltyRank

<div align="center">
  <img src="web/public/logo.png" alt="NoveltyRank Logo" width="200"/>
  <br>
  <a href="https://novelty-rank.vercel.app/"><strong>Explore the Leaderboard »</strong></a>
</div>


## Overview

This repository contains the implementation of **NoveltyRank**, a system designed to estimate the **conceptual novelty** of AI papers based on their semantic embeddings, similarity with prior literature, and metadata.  
It includes dataset processing, embedding generation, and model training pipelines for both **decoder-based LLM fine-tuning** and **encoder-based multimodal classification**.

This project is part of **Stanford University’s CS230: Deep Learning (Fall 2025)**.  
Dataset hosted at: [Hugging Face – novelty-dataset-with-similarities](https://huggingface.co/datasets/JasonYan777/novelty-rank-with-similarities)

---

## Installation

1. **Create a Conda environment**

   ```bash
   conda create -n noveltyrank python=3.11
   conda activate noveltyrank
   ```

2. **Install dependencies**

   ```bash
   pip install tinker
   pip install git+https://github.com/thinking-machines-lab/tinker-cookbook.git
   ```

3. **Clone and install the repository**

   ```bash
   git clone git@github.com:ZhengxuYan/NoveltyRank.git
   cd NoveltyRank
   pip install -r requirements.txt
   ```

4. **create .env file**
   Please create a `.env` file in the root directory and add the following lines with your respective API tokens:
   ```
   HUGGINGFACEHUB_API_TOKEN=your_hugging_face_token_here
   WANDB_API_KEY=your_wandb_api_key_here
   TINKER_API_KEY=your_tinker_api_key_here
   ```
   If you don't have these tokens, please sign up on the respective platforms to obtain them.

---

## Code Structure

```
NoveltyRank/
│
├── scrapers/               # Data collection
│   ├── arxiv_scraper/     # ArXiv paper scraping & author enrichment
│   └── conference_scraper/ # OpenReview, CVF conference scrapers
│
├── embedding/              # Embedding generation
│   ├── embeddings.py      # SPECTER2 embedding generation
│   └── similarity.py      # FAISS similarity computation
│
├── models/                 # Trained model artifacts
│   ├── Qwen_4B/           # Qwen fine-tuned model
│   └── scibert_multimodal/ # SciBERT multimodal model
│
├── scripts/                # Model training scripts
│   ├── Qwen_4B/           # Qwen fine-tuning (sft.py)
│   └── Sci_BERT/          # SciBERT fine-tuning & analysis
│
├── utils/                  # Utility scripts
│   ├── create_hf_dataset.py    # Dataset creation
│   ├── create_novelty_dataset.py
│   ├── push_to_huggingface.py  # HuggingFace upload utilities
│   └── *.py               # Other dataset utilities
│
├── results/                # Output files and datasets
│
├── README.md
└── requirements.txt
```

## Usage

### 1. Scraping Papers

Collect papers from ArXiv and conferences:

```bash
# ArXiv papers
cd scrapers/arxiv_scraper
scrapy crawl arxiv -o results/arxiv_results.csv

# Conference papers (OpenReview)
cd scrapers/conference_scraper
python scrape_openreview.py --conference ICLR --start-year 2024 --end-year 2025
```

### 2. Generating Embeddings

Generate SPECTER2 embeddings for the collected papers:

```bash
cd embedding
python embeddings.py --dataset JasonYan777/novelty-rank-dataset --batch-size 8
```

This creates both classification and proximity embeddings for each paper.

### 3. Computing Similarities

**Step 1 – Build the FAISS index**

```bash
python similarity.py
```

- Writes FAISS indices and caches per category.
- Emits the top-10 similar papers (and cosine scores) for every paper.

**Step 2 – Materialize similarity-aware SFT caches (optional)**

```bash
python embedding/simliarity_report/generate_similarity_reports.py \
   --split-root data_cache/categories/CS_CV/sft \
   --output-root data_cache/similiarity_aware_categories/CS_CV/sft \
   --preview-prompts 1 \
   --preview-outputs 1
```

- Scans the cached cs.CV SFT splits under `--split-root`.
- Writes augmented train/test JSONL files to the mirrored path under `--output-root`.
- Previews a single prompt/output pair so you can sanity-check the summaries.

Pass `include_similarity_report=true` to the SFT or DPO scripts to inject these summaries into downstream prompts.


### 4. Model Training

#### Qwen3-4B SFT and DPO Fine-Tuning

- Classification SFT (with similarity-aware data)
  ```bash
  python scripts/Qwen_4B/train/sft.py \
    log_path=results/aligned_sft_classification_sim_v2 \
    wandb_name=aligned_sft_classification_sim_v2 \
    learning_rate=2e-5 \
    batch_size=256 \
    num_epochs=12 \
    eval_every=250 \
    save_every=250
  ```

- Classification DPO (warm-start from SFT checkpoint)
  ```bash
  python scripts/Qwen_4B/train/dpo.py \
    env_config.dpo_mode=classification \
    env_config.load_checkpoint_path=tinker://b5ca8513-464a-563b-b0e0-6e3a26fe90f9:train:0/weights/final \
    env_config.log_path=results/aligned_dpo_classification_sim \
    env_config.wandb_name=aligned_dpo_classification_sim \
    env_config.learning_rate=1e-6 \
    env_config.batch_size=128 \
    env_config.num_epochs=1 \
    env_config.eval_every=100 \
    env_config.save_every=100
  ```

- Comparison SFT (base data without similarity-aware prompts)
  ```bash
  python scripts/Qwen_4B/train/sft.py \
    data_variant=base \
    sft_task=comparison \
    log_path=results/aligned_sft_comparison_base \
    wandb_name=aligned_sft_comparison_base \
    learning_rate=3e-5 \
    batch_size=64 \
    num_epochs=10 \
    eval_every=40 \
    save_every=40
  ```

- Comparison DPO (warm-start from comparison SFT)
  ```bash
  python scripts/Qwen_4B/train/dpo.py \
    env_config.dpo_mode=comparison \
    env_config.load_checkpoint_path=tinker://da6dc40b-ab62-5509-9a69-4691d2b5e044:train:0/weights/final \
    env_config.log_path=results/aligned_dpo_comparison_base \
    env_config.wandb_name=aligned_dpo_comparison_base \
    env_config.learning_rate=1.5e-6 \
    env_config.batch_size=64 \
    env_config.num_epochs=4 \
    env_config.eval_every=50 \
    env_config.save_every=50
  ```

All runs write checkpoints and W&B logs into the specified `log_path`. Use unique directories when launching multiple experiments in parallel.

**Description**

- These scripts fine-tune `Qwen/Qwen3-4B-Instruct-2507` on the NoveltyRank dataset using supervised learning and DPO where applicable.
- We rely on the Tinker Cookbook for data loading, distributed training, and other low-level details so you can focus on configuration and hyperparameters.

#### Qwen3-4B Evaluations

Quickly sanity-check the latest checkpoints with the lightweight evaluation scripts:

```bash
# Classification accuracy on whole dataset with similarity report
python scripts/Qwen_4B/test/test_classification.py \
   --category WHOLE_DATASET \
   --model-path tinker://aa417c54-d5ee-5b99-a04e-66a1f4b40a51:train:0/sampler_weights/000900 \
   --include-similarity-report

# Comparison accuracy on whole dataset
python scripts/Qwen_4B/test/test_comparison.py \
   --category WHOLE_DATASET \
   --model-path tinker://a69d7dfe-9693-5eeb-9dea-b065b39b13e0:train:0/sampler_weights/000700
```

Key flags:
- `--category`: Choose a category-specific cache if present (e.g. `CS_CV`, `CS_AI`). If not provided, the scripts fall back to the whole-dataset cache.
- `--model-name`: Base model registered with Tinker (defaults to the production Qwen3-4B checkpoint).
- `--model-path`: Tinker checkpoint URI. Pass an empty string to fall back to the base model.
- `--temperature`: Sampling temperature (default `0.0` for deterministic decoding).
- `--max-tokens`: Maximum generated tokens (`10` for classification, `512` for comparison).
- `--limit`: Caps the number of evaluation examples (useful for quick smoke tests).
- `--include-similarity-report`: Add aggregated similarity context to each classification prompt.




#### SciBERT Multimodal Training

Fine-tunes SciBERT with text + embeddings + similarity features. Config options are at the top of the file.

```bash
python scripts/Sci_BERT/ft_scibert.py
```

#### Siamese SciBERT Training

Trains a Siamese network with SciBERT backbone to predict pairwise novelty preferences.

```bash
python scripts/siamese_scibert/train_pairwise.py
```

#### Siamese SciBERT Evaluation

Evaluate the pairwise agreement rate of the trained Siamese SciBERT model:


```bash
python scripts/siamese_scibert/evaluate_agreement.py
```

**Results (Agreement Rates):**

| Category | Agreement | Count |
|----------|-----------|-------|
| cs.AI    | 72.32%    | 719   |
| cs.CL    | 67.39%    | 1429  |
| cs.CR    | 86.62%    | 1353  |
| cs.CV    | 71.54%    | 2727  |
| cs.LG    | 73.26%    | 1963  |
| cs.RO    | 84.25%    | 1340  |
| **COMBINED** | **75.26%** | **9531** |

#### OpenAI Baseline Evaluation

Evaluates frontier models (e.g., GPT-5.1) on the pairwise novelty task using the `bespokelabs-curator` library.

```bash
python scripts/openai/test_frontier_novelty.py
```

**OpenAI Model Agreement Rates**

| Category | Count | 4omini | gpt4o | 5-mini | 5.1 |
|----------|-------|--------|-------|--------|-----|
| cs.AI    | 65    | 58.46% | 58.46% | 66.15% | 67.69% |
| cs.CL    | 326   | 55.52% | 55.52% | 56.13% | 56.13% |
| cs.CR    | 48    | 66.67% | 68.75% | 66.67% | 75.00% |
| cs.CV    | 631   | 52.14% | 56.89% | 57.05% | 58.00% |
| cs.LG    | 243   | 51.85% | 54.73% | 66.67% | 61.32% |
| cs.RO    | 45    | 75.56% | 64.44% | 73.33% | 60.00% |
| **COMBINED** | **1358** | **54.49%** | **56.92%** | **59.87%** | **59.28%** |
