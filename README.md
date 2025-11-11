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
   conda create -n noveltyrank python=3.10
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
   pip install -e .
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

If you want to customize hyperparameters, you can modify the `build_config` function in 
`scripts/Qwen_4B/sft.py`.

- **Description**
  - The script fine-tunes the Qwen/Qwen3-4B-Instruct-2507 on the NoveltyRank dataset using supervised learning with Tinker Cookbook.
  - Tinker Cookbook is used for data loading, model training, and evaluation, without worrying about low-level details(e.g., distributed training, mixed precision, etc.).

#### SciBERT Multimodal Training

Fine-tunes SciBERT with text + embeddings + similarity features. Config options are at the top of the file.

```bash
python scripts/Sci_BERT/ft_scibert.py
```

Trains SciBERT with text + embeddings + similarity features. Config options are at the top of the file.
