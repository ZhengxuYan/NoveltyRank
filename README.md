# NoveltyRank

## Overview ##
This repository contains the implementation of **NoveltyRank**, a system designed to estimate the **conceptual novelty** of AI papers based on their semantic embeddings, similarity with prior literature, and metadata.  
It includes dataset processing, embedding generation, and model training pipelines for both **decoder-based LLM fine-tuning** and **encoder-based multimodal classification**.

This project is part of **Stanford University’s CS230: Deep Learning (Fall 2025)**.  
Dataset hosted at: [Hugging Face – novelty-dataset](https://huggingface.co/datasets/JasonYan777/novelty-dataset)

---

## Installation ##

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

## Code Structure ##

NOVELTYRANK/
│
├── data scraping/
│
├── embedding/
│
├── models/
│   ├── Qwen_4B/
│   └── scibert_multimodal/
│
├── scripts/
│   ├── Qwen_4B/
│   └── scibert_multimodal/
│   └── utils/
│
├── README.md
└── requirements.txt


## Usage ##
### 1. Dataset Preparation ###

#### 2.1 Qwen3-4B Supervised Fine-Tuning ####
- **Command Line**
    ```bash
    python scripts/Qwen_4B/sft.py
    ```
    If you want to customize hyperparameters, you can modify the `build_config` function in `scripts/Qwen_4B/sft.py`.

- **Description**
    - The script fine-tunes the Qwen/Qwen3-4B-Instruct-2507 on the NoveltyRank dataset using supervised learning with Tinker Cookbook.
    - Tinker Cookbook is used for data loading, model training, and evaluation, without worrying about low-level details(e.g., distributed training, mixed precision, etc.).

#### 2.2 SciBERT Multi-Modal Fine-Tuning ####
