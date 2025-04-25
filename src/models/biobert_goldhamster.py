import os
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast, TFBertModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
LABELS_DIR = Path("data/goldhamster/labels")
PAPERS_DIR = Path("data/goldhamster/papers")
OUTPUT_DIR = Path("data/goldhamster/predictions")
MODEL_PATH = Path("models/goldhamster/goldhamster_model.h5")

def setup_biobert():
    """Set up the BioBERT tokenizer and model."""
    # Max length of tokens
    max_length = 256
    # Load BioBERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('dmis-lab/biobert-v1.1')
    # Load the Transformers BERT model
    transformer_model = TFBertModel.from_pretrained('dmis-lab/biobert-v1.1', from_pt=True)
    return transformer_model, transformer_model.config, max_length, tokenizer

# Initialize BioBERT
transformer_model, transformer_config, max_length, tokenizer = setup_biobert()

def load_labels(labels_dir: Path) -> pd.DataFrame:
    """Load all label files into a pandas DataFrame."""
    data = []
    for file in labels_dir.glob("*.txt"):
        with open(file, "r") as f:
            for line in f:
                if line.strip():
                    pmid, labels = line.strip().split("\t")
                    data.append({"pmid": pmid, "labels": labels})
    return pd.DataFrame(data)

def load_papers(papers_dir: Path) -> pd.DataFrame:
    """Load all paper metadata into a pandas DataFrame."""
    data = []
    for file in papers_dir.glob("*.txt"):
        text = file.read_text()
        pmid = file.stem
        title = text.split("Title: ")[1].split("\n")[0]
        abstract = text.split("Abstract: ")[1].split("\n")[0]
        mesh_terms = text.split("MeSH Terms: ")[1].split("\n")[0]
        data.append({"pmid": pmid, "title": title, "abstract": abstract, "mesh_terms": mesh_terms})
    return pd.DataFrame(data)

def preprocess_data(papers: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Preprocess the data into the format required by the GoldHamster model."""
    inputs = []
    for _, row in papers.iterrows():
        title = row["title"]
        abstract = row["abstract"]
        if title != "N/A" and abstract != "N/A":
            inputs.append(f"{title} {abstract}")
    tokenized = tokenizer(
        text=inputs,
    	add_special_tokens=True,
	    max_length=max_length,
	    truncation=True,
	    padding=True, 
	    return_tensors='tf',
	    return_token_type_ids = False,
	    return_attention_mask = False,
	    verbose = True
    )
    return pad_sequences(tokenized["input_ids"], maxlen=max_length, padding="post", truncating="post")

def predict_with_model(model_path: Path, inputs: np.ndarray) -> List[Dict[str, float]]:
    """Load the GoldHamster model and make predictions."""
    model = load_model(model_path)
    predictions = model.predict(inputs)
    print(f"Predictions: {predictions}")
    return predictions

def save_predictions(predictions: Dict[str, np.ndarray], papers: pd.DataFrame, output_dir: Path) -> None:
    """Save predictions to a file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(5):
        pmid = papers.iloc[i]["pmid"]
        output_file = output_dir / f"{pmid}.txt"
        
        with open(output_file, "w") as f:
            f.write(f"PMID: {pmid}\n")
            f.write("Predictions:\n")
            
            for label, scores in predictions.items():
                arr_pred = predictions[label][1]
                print(arr_pred)
                if arr_pred[1]>arr_pred[0]:
                    f.write(f"  {label}\n")
                
        print(f"Saved predictions for PMID {pmid} to {output_file}")

def main():
    # Load data
    labels = load_labels(LABELS_DIR)
    papers = load_papers(PAPERS_DIR)
    
    # Preprocess data
    inputs = preprocess_data(papers)
    
    # Apply model
    print(f"Applying GoldHamster model to {len(inputs)} inputs...")
    predictions = predict_with_model(MODEL_PATH, inputs[:5])
    
    # Save predictions
    save_predictions(predictions, papers, OUTPUT_DIR)

if __name__ == "__main__":
    main()