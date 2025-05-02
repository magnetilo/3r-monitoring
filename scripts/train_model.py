from pathlib import Path
import time
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime

# Get the project root directory
project_root = Path(__file__).resolve().parent.parent

# Add the project directory to the system path
import sys
sys.path.append(str(project_root))

# Local imports
from src.data_loaders import GoldhamsterDataLoader
from src.model import BioBERTClassifier

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Configuration parameters
# Set these values to configure the training/prediction process
CONFIG = {
    # Model parameters
    "model_base_name": "goldhamster",  # Name of the model
    "model_pretrainded": "dmis-lab/biobert-v1.1",   # Pre-trained Hugging Face model name to use instead of BioBERT
    "model_learning_rate": 1e-4,      # Learning rate for training
    "model_batch_size": 32,           # Batch size for training
    "model_epochs": 10,                # Number of training epochs
    "model_max_length": 256,          # Maximum sequence length for BERT

    # Dataset parameters
    "data_name": "goldhamster",          # Type of dataset to use (goldhamster or custom)
    "data_split": "1",                     # Data split name to use (e.g., "_full", "0", "1", ..., "9")
    "data_use_titles": True,             # Whether to use paper titles
    "data_use_abstracts": True,          # Whether to use paper abstracts
    "data_use_mesh_terms": True,         # Whether to use MeSH terms
    "data_text_column": "TEXT",          # Name of the column containing text
    "data_id_column": "PMID",            # Name of the column containing document IDs
}


def main():
    """Main function to train or predict with BioBERT model."""
    # Get model name
    model_name = f"{CONFIG['model_base_name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model_path = project_root / "models" / CONFIG["data_name"] / f"{model_name}.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)

     # Set up MLflow experiment and run
    mlflow.set_experiment(CONFIG["model_base_name"])

    # Initialize data loader
    papers_dir = project_root / "data" / CONFIG["data_name"] / "docs"
    labels_dir = project_root / "data" / CONFIG["data_name"] / "labels"
    data_loader = GoldhamsterDataLoader(
        papers_dir, 
        labels_dir, 
        use_titles=CONFIG["data_use_titles"],
        use_abstracts=CONFIG["data_use_abstracts"],
        use_mesh_terms=CONFIG["data_use_mesh_terms"],
        text_column=CONFIG["data_text_column"],
        id_column=CONFIG["data_id_column"]
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=model_name):
        # Log configuration parameters
        for key, value in CONFIG.items():
            if value is not None:
                mlflow.log_param(key, value)

        # Initialize model
        print("Initializing model...")
        model = BioBERTClassifier(
            model_path=model_path,
            labels=data_loader.all_labels,
            max_length=CONFIG["model_max_length"],
            learning_rate=CONFIG["model_learning_rate"],
            batch_size=CONFIG["model_batch_size"],
            epochs=CONFIG["model_epochs"],
            load_model=False
        )

        split = CONFIG["data_split"]
        # Load data
        print(f"Loading data from split {split}...")
        train_df, train_skipped = data_loader.create_dataframe_from_split(f"train{split}.txt")
        dev_df, dev_skipped = data_loader.create_dataframe_from_split(f"dev{split}.txt")
                            
        print(f"Training model...")
        start_time = time.time()
        model.train(train_df, dev_df, text_column=CONFIG["data_text_column"])
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Log training time
        mlflow.log_metric("training_time_seconds", round(training_time, 2))
        
        # # Log the model
        # mlflow.log_artifact(model.model_path)
        # mlflow.set_tag("Model", model.model_path.stem)

        # Log only relative model path instead of full model
        mlflow.log_param("model_name", model_path.stem)
        mlflow.log_param("model_path", str(f"{model_path.parent.name}/{model_path.name}"))


if __name__ == "__main__":
    main()