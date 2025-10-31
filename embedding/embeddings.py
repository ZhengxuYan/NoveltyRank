"""
Generate SPECTER2 embeddings (classification and proximity model)  
SPECTER2 is fine-tuned for scientific paper representation for various tasks.

This script:
1. Loads dataset from HuggingFace
2. For each split (train and test), generates two types of embeddings:
   - Classification embeddings using the "allenai/specter2_classification" adapter
        For intrinsic content-level semantic identity
   - Proximity embeddings using the "allenai/specter2" adapter 
        For similarity computation (citation-based contrastive learning)
3. Saves the dataset with embeddings as pickle files

"""
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch
from tqdm import tqdm
import os

def generate_embeddings_with_adapter(
    data,
    group, 
    adapter_source, 
    adapter_name,
    batch_size=8
):
    """
    Generate embeddings for a dataset split using a specific SPECTER2 adapter
    
    Args:
        data: Subset data
        group: Name of split ('train' or 'test')
        adapter_source: HuggingFace adapter identifier ('allenai/specter2_classification' or 'allenai/specter2')
        adapter_name: Local name for the adapter ('classification' or 'proximity')
        batch_size: Batch size for processing
    
    Returns:
        List of embeddings (order preserved matching input data)
    """
    print("\n" + "="*60)
    print(f"Generating {adapter_name} embeddings for {group} split")
    print("="*60)
    
    # Load model and tokenizer
    print("Loading SPECTER2 base model...")
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
    model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
    
    # Load and activate adapter
    print(f"Loading adapter: {adapter_source}")
    model.load_adapter(adapter_source, source="hf", load_as=adapter_name, set_active=True)
    
    # Move model to appropriate device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple M2 GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    model.to(device)
    model.eval()
    
    # Generate embeddings
    print(f"Generating embeddings (batch_size={batch_size})...")
    embeddings = []

    for i in tqdm(range(0, len(data), batch_size), desc=f"Processing {group}"):
        batch_end = min(i + batch_size, len(data))
        
        # Prepare batch texts
        batch_texts = []
        for idx in range(i, batch_end):
            item = data[idx]
            text = f"Title: {item['Title']} Abstract: {item['Abstract']}"
            batch_texts.append(text)
        
        # Tokenize batch
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                          return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings for batch
        with torch.no_grad():
            output = model(**inputs)
            batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Store results
        for embedding in batch_embeddings:
            embeddings.append(embedding)
    
    print(f"✓ Generated {len(embeddings)} embeddings")
    print(f"✓ Embedding dimension: {embeddings[0].shape[0]}")
    
    # Clean up to free memory
    del model
    del tokenizer

    # Clear GPU cache based on which device was actually used
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

    return embeddings

# Main execution
if __name__ == "__main__":
    # Load dataset 
    print("Loading dataset...")
    ds = load_dataset("JasonYan777/novelty-dataset")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Process both train and test group
    for group in ['train', 'test']:
        print("\n" + "="*80)
        print(f"PROCESSING {group.upper()} SPLIT")
        print("="*80)
        
        # Generate classification embeddings
        classification_embeddings = generate_embeddings_with_adapter(
            data=ds[group],
            group=group,
            adapter_source="allenai/specter2_classification",
            adapter_name="classification",
            batch_size=8
        )
        
        # Generate proximity embeddings
        proximity_embeddings = generate_embeddings_with_adapter(
            data=ds[group],
            group=group,
            adapter_source="allenai/specter2",
            adapter_name="proximity",
            batch_size=8
        )
        
        # Convert dataset to DataFrame and add both embeddings
        df_final = pd.DataFrame(ds[group])
        df_final['Classification_embedding'] = classification_embeddings
        df_final['Proximity_embedding'] = proximity_embeddings
        
        # Save the enhanced dataset
        print("\n" + "="*60)
        print(f"Saving {group} dataset with embeddings...")
        print("="*60)

        output_file = f'output/embeddings_{group}.pkl'
        df_final.to_pickle(output_file)
        print(f"✓ Saved {len(df_final)} samples to {output_file}")
        
        print(f"✓ Dataset columns: {list(df_final.columns)}")
        print(f"✓ Classification embedding dimension: {classification_embeddings[0].shape[0]}")
        print(f"✓ Proximity embedding dimension: {proximity_embeddings[0].shape[0]}")
        print("="*60)
    
    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETE!")
    print("="*80)
