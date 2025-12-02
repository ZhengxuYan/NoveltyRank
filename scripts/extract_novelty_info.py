import pandas as pd
import requests
import os
import time
import tarfile
import shutil
import json
import glob
from tqdm import tqdm
from pypdf import PdfReader
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from bespokelabs import curator
from dotenv import load_dotenv
import argparse

load_dotenv()

from datasets import load_dataset

# Configuration
DATASET_NAME = 'JasonYan777/novelty-rank-balanced'
OUTPUT_FILE = 'data/processed/paper_novelty_analysis.jsonl'
TEMP_DIR = 'data/temp_papers'
ARXIV_SRC_URL = 'https://arxiv.org/src/{}'
ARXIV_PDF_URL = 'https://arxiv.org/pdf/{}.pdf'
DELAY_SECONDS = 2

# --- Pydantic Models for LLM ---

class PaperNoveltyAnalysis(BaseModel):
    # Delta Signals
    contribution_statements: List[str] = Field(default_factory=list, description="Concise sentences where authors state their contribution.")
    gap_identification: List[str] = Field(default_factory=list, description="Concise sentences describing limitations of SOTA.")
    novelty_claim_type: str = Field(default="N/A", description="Type of novelty: 'Architecture', 'Data', 'Application', 'Efficiency', 'Theory', or 'Other'.")
    
    # Problem Solution
    input_description: str = Field(default="N/A", description="Short description of the input.")
    output_description: str = Field(default="N/A", description="Short description of the target output.")
    method_name: str = Field(default="N/A", description="Specific name or description of the mechanism used.")
    objective_function: str = Field(default="N/A", description="Specific loss function or optimization target.")
    
    # Quantitative Evidence
    sota_comparisons: List[str] = Field(default_factory=list, description="Brief text summarizing performance gains over baselines.")
    efficiency_metrics: str = Field(default="N/A", description="Parameter counts, inference time, or FLOPs, if mentioned.")
    
    # Novelty Abstract
    novelty_abstract_context: str = Field(default="N/A", description="Context: Current methods for [Task] struggle with [Limitation].")
    novelty_abstract_proposal: str = Field(default="N/A", description="Proposal: We introduce [Method Name], which utilizes [Mechanism].")
    novelty_abstract_difference: str = Field(default="N/A", description="Difference: Unlike [Previous Method], our approach [Key Differentiator].")
    novelty_abstract_result: str = Field(default="N/A", description="Result: This leads to [Quantitative Improvement] on [Benchmark].")

ANALYSIS_PROMPT = """Analyze the provided text from a research paper to extract information for a novelty ranking task.

Paper Text:
{text}

Instructions:
You are an expert researcher helping to fine-tune a model to detect scientific novelty. 
Your goal is to extract specific signals that define the "delta" between this work and prior art.

**CRITICAL: Be extremely concise. Use short sentences and bullet points. Save tokens.**

Return a JSON object with the following keys:

1. **contribution_statements**: List of concise sentences where authors state their contribution.
2. **gap_identification**: List of concise sentences describing limitations of SOTA.
3. **novelty_claim_type**: Type of novelty ('Architecture', 'Data', 'Application', 'Efficiency', 'Theory', 'Other').
4. **input_description**: Short description of the input.
5. **output_description**: Short description of the target output.
6. **method_name**: Specific name or description of the mechanism used.
7. **objective_function**: Specific loss function or optimization target.
8. **sota_comparisons**: List of brief text summarizing performance gains over baselines.
9. **efficiency_metrics**: Parameter counts, inference time, or FLOPs.
10. **novelty_abstract_context**: Context: Current methods for [Task] struggle with [Limitation].
11. **novelty_abstract_proposal**: Proposal: We introduce [Method Name], which utilizes [Mechanism].
12. **novelty_abstract_difference**: Difference: Unlike [Previous Method], our approach [Key Differentiator].
13. **novelty_abstract_result**: Result: This leads to [Quantitative Improvement] on [Benchmark].

Constraints:
- Be **grounded**: Only report information explicitly stated in the text.
- If a field is not applicable or found, use "N/A" or an empty list.
- **Conciseness is key.**
"""

# --- Curator LLM ---

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class NoveltyAnalyzer(curator.LLM):
    response_format = PaperNoveltyAnalysis

    def prompt(self, input: dict) -> str:
        # Truncate text to fit context window (approx 100k chars is usually safe for large context models)
        text = input["text"][:100000] 
        return ANALYSIS_PROMPT.format(text=text)

    def parse(self, input: dict, response: PaperNoveltyAnalysis) -> List[Dict]:
        return [
            {
                "arxiv_id": input["arxiv_id"],
                "novelty_analysis": response.dict()
            }
        ]

# --- Helper Functions ---

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_tar_gz(file_path, extract_path):
    try:
        if tarfile.is_tarfile(file_path):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=extract_path)
            return True
        return False
    except Exception:
        return False

def get_text_from_latex(directory):
    text_content = []
    # Find all .tex files recursively
    tex_files = glob.glob(os.path.join(directory, '**/*.tex'), recursive=True)
    
    # Heuristic: Try to find main.tex or similar first, or just read all
    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                text_content.append(f"--- File: {os.path.basename(tex_file)} ---\n")
                text_content.append(f.read())
                text_content.append("\n")
        except Exception as e:
            print(f"Error reading {tex_file}: {e}")
            
    return "\n".join(text_content)

def get_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

def sanitize_text(text):
    """
    Remove surrogate characters that cause issues with pyarrow/UTF-8 encoding.
    """
    if not isinstance(text, str):
        return ""
    return text.encode('utf-8', 'ignore').decode('utf-8')

def sanitize_record(record):
    if isinstance(record, dict):
        return {k: sanitize_record(v) for k, v in record.items()}
    elif isinstance(record, list):
        return [sanitize_record(v) for v in record]
    elif isinstance(record, str):
        return sanitize_text(record)
    else:
        return record

def process_papers(limit=None, model_name="gpt-5-mini", batch_size=10):
    ensure_dir(TEMP_DIR)
    ensure_dir(os.path.dirname(OUTPUT_FILE))
    
    # Load existing progress
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'arxiv_id' in data:
                        processed_ids.add(str(data['arxiv_id']).strip())
                except:
                    pass
    
    print(f"Found {len(processed_ids)} already processed papers.")
    
    print(f"Loading dataset: {DATASET_NAME}")
    try:
        hf_ds = load_dataset(DATASET_NAME)
        # Combine train and test splits
        dfs = []
        if 'train' in hf_ds: dfs.append(hf_ds['train'].to_pandas())
        if 'test' in hf_ds: dfs.append(hf_ds['test'].to_pandas())
        df = pd.concat(dfs, ignore_index=True)
        
        # Rename columns to match expected format
        df = df.rename(columns={'arXiv ID': 'arxiv_id'})
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Filter out already processed
    # Ensure arxiv_id is string
    if 'arxiv_id' not in df.columns:
        print("Error: 'arxiv_id' column not found in dataset.")
        print(f"Available columns: {df.columns}")
        return

    df['arxiv_id'] = df['arxiv_id'].astype(str).str.strip()
    
    # Filter out already processed
    initial_count = len(df)
    papers_to_process = df[~df['arxiv_id'].isin(processed_ids)]
    skipped_count = initial_count - len(papers_to_process)
    
    print(f"Skipping {skipped_count} papers that are already processed.")
    
    # Deduplicate by arxiv_id just in case
    papers_to_process = papers_to_process.drop_duplicates(subset=['arxiv_id'])
    
    print(f"Found {len(papers_to_process)} papers to process.")
    
    if limit:
        papers_to_process = papers_to_process.head(limit)
        print(f"Processing next {limit} papers...")
    
    analyzer = NoveltyAnalyzer(model_name=model_name)
    # Allow partial responses for batch processing
    if hasattr(analyzer, '_request_processor') and hasattr(analyzer._request_processor, 'config'):
        analyzer._request_processor.config.require_all_responses = False
    
    batch_input = []
    
    for index, row in tqdm(papers_to_process.iterrows(), total=len(papers_to_process)):
        arxiv_id = str(row['arxiv_id'])
        paper_dir = os.path.join(TEMP_DIR, arxiv_id)
        ensure_dir(paper_dir)
        
        text = ""
        
        # 1. Try Source
        src_url = ARXIV_SRC_URL.format(arxiv_id)
        src_save_path = os.path.join(paper_dir, 'source.tar.gz')
        
        download_success = False
        if download_file(src_url, src_save_path):
            if extract_tar_gz(src_save_path, paper_dir):
                extracted_text = get_text_from_latex(paper_dir)
                if len(extracted_text) > 100: 
                    text = extracted_text
                    download_success = True
        
        # 2. Fallback to PDF
        if not download_success:
            pdf_url = ARXIV_PDF_URL.format(arxiv_id)
            pdf_save_path = os.path.join(paper_dir, 'paper.pdf')
            if download_file(pdf_url, pdf_save_path):
                extracted_text = get_text_from_pdf(pdf_save_path)
                if len(extracted_text) > 100:
                    text = extracted_text
                    download_success = True
        
        # Cleanup temp files immediately
        try:
            shutil.rmtree(paper_dir)
        except Exception as e:
            print(f"Error cleaning up {paper_dir}: {e}")

        if download_success and text:
            text = sanitize_text(text)
            batch_input.append({"arxiv_id": arxiv_id, "text": text})
        else:
            print(f"Failed to get content for {arxiv_id}")
            
        # Process batch if full
        if len(batch_input) >= batch_size:
            try:
                result = analyzer(batch_input)
                with open(OUTPUT_FILE, 'a') as f:
                    for record in result.dataset.to_pandas().to_dict('records'):
                        record = sanitize_record(record)
                        f.write(json.dumps(record, cls=NumpyEncoder) + "\n")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing batch: {e}")
            
            batch_input = [] # Reset batch
            time.sleep(DELAY_SECONDS)

    # Process remaining items
    if batch_input:
        try:
            result = analyzer(batch_input)
            with open(OUTPUT_FILE, 'a') as f:
                for record in result.dataset.to_pandas().to_dict('records'):
                    record = sanitize_record(record)
                    f.write(json.dumps(record, cls=NumpyEncoder) + "\n")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing final batch: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract novelty information from papers.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of papers to process.")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Model to use for extraction.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing.")
    args = parser.parse_args()

    process_papers(limit=args.limit, model_name=args.model, batch_size=args.batch_size)
