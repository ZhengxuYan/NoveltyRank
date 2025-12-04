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
Dataset hosted at: [Hugging Face – novelty-ranked-preprints](https://huggingface.co/datasets/JasonYan777/novelty-ranked-preprints/viewer/default/train?sort%5Bcolumn%5D=novelty_score&sort%5Bdirection%5D=desc&views%5B%5D=train&row=5073)

---

## Methodology & Results

### Siamese SciBERT Architecture
![Siamese SciBERT Structure](web/public/siamese_scibert_structure.png)

### Performance Comparison
We compared our Siamese SciBERT model against frontier LLMs (GPT-4o, GPT-5.1, etc.) on the pairwise novelty ranking task.

**Agreement Rates (Siamese SciBERT vs. Frontier Models)**

| Model | cs.AI | cs.CL | cs.CR | cs.CV | cs.LG | cs.RO | **COMBINED** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-4o-mini | 58.46% | 55.52% | 66.67% | 52.14% | 51.85% | 75.56% | 54.49% |
| GPT-4o | 58.46% | 55.52% | 68.75% | 56.89% | 54.73% | 64.44% | 56.92% |
| GPT-5-mini | 66.15% | 56.13% | 66.67% | 57.05% | 66.67% | 73.33% | 59.87% |
| GPT-5.1 | 67.69% | 56.13% | 75.00% | 58.00% | 61.32% | 60.00% | 59.28% |
| **Siamese SciBERT** | **72.32%** | **67.39%** | **86.62%** | **71.54%** | **73.26%** | **84.25%** | **75.26%** |

### Binary Novelty Classification
We also evaluated frontier models on a binary classification task (Novel vs Not Novel).

**Binary Classification Metrics**

| Model | Category | Acc | Prec | Rec | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **GPT-4o-mini** | cs.AI | 12.50% | 9.79% | 100.00% | 17.84% |
|  | cs.CL | 16.50% | 15.23% | 100.00% | 26.43% |
|  | cs.CR | 9.50% | 3.72% | 100.00% | 7.18% |
|  | cs.CV | 16.50% | 16.08% | 100.00% | 27.71% |
|  | cs.LG | 14.00% | 11.79% | 100.00% | 21.10% |
|  | cs.RO | 7.00% | 3.63% | 100.00% | 7.00% |
|  | COMBINED | **12.67%** | **10.12%** | **100.00%** | **18.38%** |
| **GPT-4o** | cs.AI | 19.50% | 9.66% | 89.47% | 17.44% |
|  | cs.CL | 29.50% | 17.16% | 96.67% | 29.15% |
|  | cs.CR | 26.00% | 4.52% | 100.00% | 8.64% |
|  | cs.CV | 23.00% | 17.20% | 100.00% | 29.36% |
|  | cs.LG | 18.00% | 11.89% | 95.65% | 21.15% |
|  | cs.RO | 13.50% | 3.89% | 100.00% | 7.49% |
|  | COMBINED | **21.58%** | **10.85%** | **96.61%** | **19.50%** |
| **GPT-5-mini** | cs.AI | 68.00% | 14.29% | 47.37% | 21.95% |
|  | cs.CL | 56.00% | 17.05% | 50.00% | 25.42% |
|  | cs.CR | 56.00% | 3.45% | 42.86% | 6.38% |
|  | cs.CV | 56.00% | 21.43% | 65.62% | 32.31% |
|  | cs.LG | 53.00% | 16.19% | 73.91% | 26.56% |
|  | cs.RO | 73.00% | 8.77% | 71.43% | 15.62% |
|  | COMBINED | **60.33%** | **14.06%** | **59.32%** | **22.73%** |
| **GPT-5.1** | cs.AI | 24.32% | 10.75% | 90.91% | 19.23% |
|  | cs.CL | 34.23% | 22.34% | 100.00% | 36.52% |
|  | cs.CR | 25.23% | 3.49% | 100.00% | 6.74% |
|  | cs.CV | 21.62% | 13.86% | 100.00% | 24.35% |
|  | cs.LG | 24.32% | 17.65% | 100.00% | 30.00% |
|  | cs.RO | 15.32% | 3.09% | 100.00% | 6.00% |
|  | COMBINED | **24.17%** | **12.04%** | **98.57%** | **21.46%** |
| **SciBERT Multimodal** | cs.AI | 74.74% | 13.66% | 38.46% | 20.16% |
|  | cs.CL | 63.13% | 21.08% | 35.89% | 26.56% |
|  | cs.CR | 96.43% | 0.00% | 0.00% | 0.00% |
|  | cs.CV | 67.42% | 21.38% | 27.42% | 24.03% |
|  | cs.LG | 66.82% | 15.42% | 44.86% | 22.95% |
|  | cs.RO | 95.38% | 4.76% | 2.22% | 3.03% |
|  | COMBINED | **74.42%** | **18.66%** | **31.30%** | **23.38%** |

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




#### SciBERT and Siamese SciBERT Training

- Classification (SciBERT Multimodal)
  ```bash
  python scripts/Sci_BERT/ft_scibert.py \
    --dataset JasonYan777/novelty-rank-with-similarities \
    --batch-size 32 \
    --learning-rate 2e-5 \
    --epochs 5
  ```

- Comparison (Siamese SciBERT)

  ```bash
  python scripts/siamese_scibert/train_pairwise.py
  ```

#### SciBERT Evaluations

- Siamese SciBERT Agreement Rate
  ```bash
  python scripts/siamese_scibert/evaluate_agreement.py
  ```

#### OpenAI Baseline Evaluation

Evaluates frontier models (e.g., GPT-5.1) on the pairwise novelty task using the `bespokelabs-curator` library.
```bash
python scripts/openai/test_frontier_novelty_classification.py
```

```bash
python scripts/openai/test_frontier_novelty_comparison.py
```
