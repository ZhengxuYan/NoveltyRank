# NoveltyRank

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

#### Qwen3-4B Fine-Tuning
Launch supervised fine-tuning (SFT) runs with the following commands:

```bash
# Whole-dataset SFT (default)
python scripts/Qwen_4B/train/sft.py \
   log_path=results/sft_whole \
   wandb_name=sft_whole

# Category-specific SFT (example: cs.CV)
python scripts/Qwen_4B/train/sft.py \
   category=cs.CV \
   category_seed=42 \
   log_path=results/sft_cv \
   wandb_name=sft_cv

# Enable similarity-conditioned prompts
python scripts/Qwen_4B/train/sft.py \
   category=cs.CV \
   category_seed=42 \
   log_path=results/sft_cv_sim \
   wandb_name=sft_cv_sim \
   include_similarity_report=true
```

If the script detects an existing log directory it will ask whether to delete, resume, or exit. Respond at the prompt to control the behavior.

If you want to customize hyperparameters, you can modify the `build_config` function in 
`scripts/Qwen_4B/train/sft.py`.

#### Qwen3-4B DPO Training
Classification-style (1/0) and comparison-style (A/B) DPO share the same entrypoint. When either `model_name` or `env_config.load_checkpoint_path` references a Tinker URI, add `env_config.renderer_name=qwen3_instruct` so the script skips renderer autodetection (which otherwise fails on Tinker IDs). For warm starts, always point `env_config.load_checkpoint_path` at the `…/weights/final` artifact rather than `…/sampler_weights/final`.

```bash
# Whole-dataset classification DPO
python scripts/Qwen_4B/train/dpo.py \
   env_config.dpo_mode=classification \
   env_config.log_path=results/dpo_classification_whole \
   env_config.wandb_name=dpo_classification_whole

# Category-specific classification DPO (cs.CV)
python scripts/Qwen_4B/train/dpo.py \
   env_config.dpo_mode=classification \
   env_config.category=cs.CV \
   env_config.log_path=results/dpo_classification_cv \
   env_config.wandb_name=dpo_classification_cv

# Resume legacy classification DPO (cs.CV) from an SFT checkpoint
python scripts/Qwen_4B/train/dpo.py \
   env_config.dpo_mode=classification \
   env_config.category=CS_CV \
   env_config.load_checkpoint_path=tinker://b134fa47-0ac6-57bc-b8c7-9cf138a3ecaa:train:0/weights/final \
   env_config.log_path=results/dpo_classification_cv_sftinit \
   env_config.wandb_name=dpo_classification_cv_sftinit 

# Include similarity reports during classification DPO(cs.CV)
python scripts/Qwen_4B/train/dpo.py \
   env_config.dpo_mode=classification \
   env_config.category=cs.CV \
   env_config.log_path=results/dpo_classification_cv_sim \
   env_config.wandb_name=dpo_classification_cv_sim \
   env_config.include_similarity_report=true

# Include similarity reports during classification DPO (load from SFT, cs.CV)
python scripts/Qwen_4B/train/dpo.py \
   env_config.dpo_mode=classification \
   env_config.category=cs.CV \
   env_config.load_checkpoint_path=tinker://4ba31574-b75b-52fc-a87a-408e984590d0:train:0/weights/final \
   env_config.log_path=results/dpo_classification_cv_sftinit_sim \
   env_config.wandb_name=dpo_classification_cv_sftinit_sim \
   env_config.include_similarity_report=true

# Category-specific comparison DPO (cs.CV)
python scripts/Qwen_4B/train/dpo.py \
   env_config.dpo_mode=comparison \
   env_config.category=cs.CV \
   env_config.log_path=results/dpo_comparison_cv \
   env_config.wandb_name=dpo_comparison_cv
```

All runs write checkpoints and W&B logs into the specified `log_path`. Use unique directories when launching multiple experiments in parallel.

- **Description**
  - The script fine-tunes the Qwen/Qwen3-4B-Instruct-2507 on the NoveltyRank dataset using supervised learning with Tinker Cookbook.
  - Tinker Cookbook is used for data loading, model training, and evaluation, without worrying about low-level details(e.g., distributed training, mixed precision, etc.).

#### Qwen3-4B Evaluations

Quickly sanity-check the latest checkpoints with the lightweight evaluation scripts:

```bash
# Classification accuracy on cs.CV split
python scripts/Qwen_4B/test/test_classification.py \
   --category CS_CV \
   --model-path tinker://YOUR-JOB-ID:train:0/sampler_weights/final

# Pairwise comparison accuracy on cs.CV split
python scripts/Qwen_4B/test/test_comparison.py \
   --category CS_CV \
   --model-path tinker://YOUR-JOB-ID:train:0/sampler_weights/final
```

Key flags:
- `--category`: Chooses a category-specific cache if present; defaults to the full dataset cache.
- `--model-name`: Base model registered with Tinker (defaults to the production Qwen3-4B checkpoint).
- `--model-path`: Tinker checkpoint URI (defaults to the reference fine-tune in this repo). Pass an empty string to fall back to the base model.
- `--temperature`: Sampling temperature (default `0.0` for deterministic decoding).
- `--max-tokens`: Maximum generated tokens (`10` for classification, `512` for comparison).
- `--limit`: Caps the number of evaluation examples (useful for quick smoke tests).
- `--include-similarity-report`: Adds the aggregated similarity context to each classification prompt.


#### SciBERT Multimodal Training

Fine-tunes SciBERT with text + embeddings + similarity features. Config options are at the top of the file.

```bash
python scripts/Sci_BERT/ft_scibert.py
```

Trains SciBERT with text + embeddings + similarity features. Config options are at the top of the file.


