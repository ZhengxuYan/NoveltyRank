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

Calculate paper similarities using FAISS:

```bash
python similarity.py
```

Outputs include top-10 similar papers with scores for each paper in the dataset.

### 4. Model Training

#### Qwen3-4B Fine-Tuning

```bash
python scripts/Qwen_4B/sft.py
```


Common invocations:

```bash
# Whole-dataset SFT (default)
python scripts/Qwen_4B/sft.py

# Category-specific SFT (example: cs.CV)
python scripts/Qwen_4B/sft.py category=cs.CV category_seed=42
```

If the script detects an existing log directory it will ask whether to delete, resume, or exit. Respond at the prompt to control the behavior.

If you want to customize hyperparameters, you can modify the `build_config` function in 
`scripts/Qwen_4B/sft.py`.

#### Qwen3-4B DPO Training
Classification-style (1/0) and comparison-style (A/B) DPO share the same entrypoint. When using a Tinker checkpoint URI for `model_name`, set `env_config.renderer_name=qwen3_instruct` to skip autodetection.

```bash
# Whole-dataset classification DPO
python scripts/Qwen_4B/dpo.py \
   env_config.dpo_mode=classification \
   env_config.model_name=Qwen/Qwen3-4B-Instruct-2507 \
   env_config.wandb_name=DPO_qwen_4b_classification

# Category-specific classification DPO (cs.CV)
python scripts/Qwen_4B/dpo.py \
   env_config.dpo_mode=classification \
   env_config.category=cs.CV \
   env_config.model_name=Qwen/Qwen3-4B-Instruct-2507 \
   env_config.log_path=results/noveltyrank_dpo_qwen4b_classification_cs_cv \
   env_config.wandb_name=DPO_qwen_4b_classification_cs_cv

# Category-specific comparison DPO (cs.CV)
python scripts/Qwen_4B/dpo.py \
   env_config.dpo_mode=comparison \
   env_config.category=cs.CV \
   env_config.model_name=Qwen/Qwen3-4B-Instruct-2507 \
   env_config.log_path=results/noveltyrank_dpo_qwen4b_comparison_cs_cv \
   env_config.wandb_name=DPO_qwen_4b_comparison_cs_cv
```

All runs write checkpoints and W&B logs into the specified `log_path`. Use unique directories when launching multiple experiments in parallel.

- **Description**
  - The script fine-tunes the Qwen/Qwen3-4B-Instruct-2507 on the NoveltyRank dataset using supervised learning with Tinker Cookbook.
  - Tinker Cookbook is used for data loading, model training, and evaluation, without worrying about low-level details(e.g., distributed training, mixed precision, etc.).

#### Qwen3-4B Evaluations

Quickly sanity-check the latest checkpoints with the lightweight evaluation scripts:

```bash
# Classification accuracy on cs.CV split
python scripts/Qwen_4B/test_classification.py \
   --category CS_CV \
   --model-name Qwen/Qwen3-4B-Instruct-2507 \
   --model-path tinker://YOUR-JOB-ID:train:0/sampler_weights/final \
   --temperature 0.0

# Pairwise comparison accuracy on cs.CV split
python scripts/Qwen_4B/test_comparison.py \
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


### Similarity-Aware CV Pipeline (`new_model/`)

These scripts operate on the similarity-augmented CS_CV splits stored under `data_cache/similiarity_aware_categories/CS_CV/sft/`. They mirror the legacy Qwen flows but default to the new dataset builders and evaluators.

- **Generate similarity reports**

   ```bash
   python embedding/simliarity_report/generate_similarity_reports.py \
      --train-split data_cache/categories/CS_CV/sft/train \
      --test-split data_cache/categories/CS_CV/sft/test \
      --output-dir data_cache/similiarity_aware_categories/CS_CV/sft/train \
      --model-name Qwen/Qwen3-235B-A22B-Instruct-2507 \
      --max-concurrency 2 \
      --preview-prompts 2 \
      --preview-outputs 2
   ```
   Adjust `--offset` / `--limit` to process shards, and pass `--model-path` for a fine-tuned frontier checkpoint. Generated HF datasets land under `data_cache/similiarity_aware_categories/CS_CV/sft/`.
   Adjust `--offset` / `--limit` to process shards, and pass `--model-path` for a fine-tuned frontier checkpoint. Generated HF datasets land under `data_cache/similiarity_aware_categories/CS_CV/sft/`.
   
- **Supervised fine-tuning**

   ```bash
   python new_model/scripts/sft.py \
      model_name=Qwen/Qwen3-4B-Instruct-2507 \
      train_dataset_path=data_cache/similiarity_aware_categories/CS_CV/sft/train \
      test_dataset_path=data_cache/similiarity_aware_categories/CS_CV/sft/test \
      log_path=results/new_model_sft_cv \
      wandb_name=sft_qwen4b_cv_similarity
   ```

   The defaults target the CS_CV slice; override `log_path`, `learning_rate`, `num_epochs`, and related flags to launch new runs.

- **Classification DPO**

   ```bash
   # Start from an SFT sampler (recommended)
   python new_model/scripts/dpo_classification.py \
      config.model_name=tinker://4ba31574-b75b-52fc-a87a-408e984590d0:train:0/sampler_weights/final \
      config.load_checkpoint_path=tinker://4ba31574-b75b-52fc-a87a-408e984590d0:train:0/weights/final \
      config.log_path=results/new_model_dpo_cv_v1 \
      config.wandb_name=dpo_qwen4b_cv_similarity_sftinit

   # Continue from a prior DPO state
   python new_model/scripts/dpo_classification.py \
      config.model_name=tinker://9a82def9-793e-51f8-8a7a-cd23781cbdd4:train:0/sampler_weights/000310 \
      config.load_checkpoint_path=tinker://9a82def9-793e-51f8-8a7a-cd23781cbdd4:train:0/weights/000310 \
      config.log_path=results/new_model_dpo_cv_v1_resume
   ```

   Remove `--config.load-checkpoint-path` to fine-tune directly from the supplied `model-name` weights.

- **Similarity classification evals**

   ```bash
   python new_model/scripts/test_similarity_classification.py \
      --mode dpo \
      --dataset-path data_cache/similiarity_aware_categories/CS_CV/sft/test \
      --model-name Qwen/Qwen3-4B-Instruct-2507 \
      --model-path tinker://9a82def9-793e-51f8-8a7a-cd23781cbdd4:train:0/sampler_weights/000310 \
      --temperature 0.0 \
      --limit 500
   ```

   Use `--mode sft` to replicate the single-turn prompt style. The script reports accuracy, precision, recall, F1, and unresolved counts.



#### SciBERT Multimodal Training

Fine-tunes SciBERT with text + embeddings + similarity features. Config options are at the top of the file.

```bash
python scripts/Sci_BERT/ft_scibert.py
```

Trains SciBERT with text + embeddings + similarity features. Config options are at the top of the file.


