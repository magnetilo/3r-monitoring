#!/usr/bin/env python3
"""
Script to train a BioBERT-based text classification model on the Goldhamster dataset
using MLflow for experiment tracking.
"""

from pathlib import Path
from datetime import datetime
import time
import mlflow
from r3_monitoring.data import GoldhamsterDataLoader
from r3_monitoring.models.pubmedbert_classifier import BioBERTClassifier


# Get the project root directory
project_root = Path(__file__).resolve().parent.parent

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Configuration parameters
# Set these values to configure the training/prediction process
CONFIG = {
    # Model parameters
    "experiment_name": "goldhamster-multilabel",  # MLflow experiment name
    "model_base_name": "PubMedBERT",  # Name of the model
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
    # Create temporary model path for training (MLflow will handle final storage)
    temp_model_dir = project_root / "temp_models"
    temp_model_dir.mkdir(exist_ok=True)
    model_path = temp_model_dir / f"{model_name}.pth"

     # Set up MLflow experiment and run
    mlflow.set_experiment(CONFIG["experiment_name"])

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
        history = model.train_model(train_df, dev_df, text_column=CONFIG["data_text_column"])
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Log training time
        mlflow.log_metric("training_time_seconds", round(training_time, 2))
        
        # Log training metrics if available
        if isinstance(history, dict) and 'val_accuracy' in history:
            final_val_accuracy = history['val_accuracy'][-1] if history['val_accuracy'] else 0
            mlflow.log_metric("final_val_accuracy", round(final_val_accuracy, 4))
            
            # Log training curves
            for epoch, (train_loss, val_loss, val_acc) in enumerate(zip(
                history.get('train_loss', []),
                history.get('val_loss', []),
                history.get('val_accuracy', [])
            )):
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc
                }, step=epoch)
        
        # Save and log the model with MLflow
        print("Saving and logging model with MLflow...")
        
        # First save the model to the temporary path
        model.save_model()
        
        # Log the model file as an artifact
        mlflow.log_artifact(str(model_path), artifact_path="models")
        
        # Log model metadata
        mlflow.log_dict({
            "labels": data_loader.all_labels,
            "model_pretrained": CONFIG['model_pretrainded'],
            "max_length": CONFIG['model_max_length'],
            "model_architecture": "BioBERTClassifier",
            "framework": "PyTorch"
        }, "model_metadata.json")
        
        mlflow.set_tag("Model", model_name)
        mlflow.set_tag("ModelFormat", "PyTorch")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_file", model_path.name)
        
        # Clean up temporary file
        if model_path.exists():
            model_path.unlink()
        
        print(f"Model '{model_name}' logged successfully with MLflow!")


if __name__ == "__main__":
    main()