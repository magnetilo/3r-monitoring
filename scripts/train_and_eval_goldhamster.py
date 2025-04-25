from pathlib import Path
import time
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime

# Add the src directory to the system path
import sys
sys.path.append(str(Path().resolve() / "src"))
print(str(Path().resolve() / "src"))

# Local imports
from data_loaders import GoldhamsterDataLoader, GenericTextDataLoader
from model import BioBERTClassifier
from evaluation import evaluate_multilabel, print_evaluation_results


# Configuration parameters
# Set these values to configure the training/prediction process
CONFIG = {
    # Mode
    "train": True,              # Whether to train the model
    "predict": True,            # Whether to make predictions
    "evaluate": True,           # Whether to evaluate predictions

    # Model parameters
    "model_dir": "goldhamster_custom",          # Directory to save the model
    # "model_name": "goldhamster_model.h5",         # Path to pretrained model (for prediction only)
    "model_name": "goldhamster_model_2025-04-25_13-48-48.h5",
    "learning_rate": 1e-4,      # Learning rate for training
    "batch_size": 32,           # Batch size for training
    "epochs": 10,               # Number of training epochs
    "max_length": 256,          # Maximum sequence length for BERT

    # Dataset parameters
    "dataset_type": "goldhamster",  # Type of dataset to use (goldhamster or custom)
    "split": 1,                     # Data split number to use
    "data_dir": None,               # Directory containing custom dataset files
    "labels": None,                 # Comma-separated list of labels for custom dataset
    "use_titles": True,             # Whether to use paper titles
    "use_abstracts": True,          # Whether to use paper abstracts
    "use_mesh_terms": True,         # Whether to use MeSH terms
    "text_column": "TEXT",          # Name of the column containing text
    "id_column": "PMID",              # Name of the column containing document IDs
    
    # BERT model selection
    "bert_model": None,             # Hugging Face model name to use instead of BioBERT

    # MLflow parameters
    "mlflow_tracking_uri": "http://127.0.0.1:5000",  # MLflow tracking server URI
    "experiment_name": "biobert_classification",     # MLflow experiment name
}


def main():
    """Main function to train or predict with BioBERT model."""
     # Set up MLflow
    mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
    mlflow.set_experiment(CONFIG["experiment_name"])
    
    # Generate a run name with timestamp
    run_name = f"{CONFIG['dataset_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
   # Define paths
    project_root = Path(__file__).resolve().parent.parent
    
    # Set up dataset-specific paths
    papers_dir = project_root / "data" / "goldhamster" / "papers"
    labels_dir = project_root / "data" / "goldhamster" / "labels"
    models_dir = project_root / "models" / CONFIG["model_dir"]
    predictions_dir = project_root / "data" / "goldhamster" / "predictions"
    
    # Ensure directories exists
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # File names based on split
    split = CONFIG["split"]
    train_file = f"train{split}.txt"
    dev_file = f"dev{split}.txt"
    test_file = f"test{split}.txt"
    
    # Initialize data loader
    data_loader = GoldhamsterDataLoader(
        papers_dir, 
        labels_dir, 
        use_titles=True, 
        use_abstracts=True, 
        use_mesh_terms=False
    )
    
    # Initialize model
    model = BioBERTClassifier(
        model_dir=models_dir,
        labels=data_loader.all_labels,
        model_name=CONFIG["dataset_type"],
        max_length=CONFIG["max_length"],
        learning_rate=CONFIG["learning_rate"],
        batch_size=CONFIG["batch_size"],
        epochs=CONFIG["epochs"]
    )
    
    # Use custom BERT model if specified
    if CONFIG["bert_model"]:
        model.set_custom_bert_model(CONFIG["bert_model"])
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log configuration parameters
        for key, value in CONFIG.items():
            if value is not None:
                if isinstance(value, Path):
                    mlflow.log_param(key, str(value))
                else:
                    mlflow.log_param(key, value)

        # Log dataset and model information as tags for easier filtering
        mlflow.set_tag("Dataset", f"{CONFIG['dataset_type']}_{CONFIG['split']}")
        mlflow.set_tag("Model", CONFIG["model_name"])

        # Train model
        if CONFIG["train"]:
            print(f"Loading data from split {split}...")
            data = data_loader.load_data(train_file, dev_file)
            
            train_data = data['train'][0]
            train_skipped = data['train'][1]
            dev_data = data['dev'][0]
            dev_skipped = data['dev'][1]
            
            print(f"Building model...")
            model.build_model(train_data)
            
            print(f"Training model...")
            start_time = time.time()
            model.train(train_data, dev_data, text_column=CONFIG["text_column"])
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")

            # Log training time
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Log the model
            mlflow.log_artifact(model.model_path)
            mlflow.set_tag("Model", model.model_path.stem)

        # Predict with model
        if CONFIG["predict"]:
            # Load model if path provided, otherwise use the model from training
            if CONFIG["model_name"] and not CONFIG["train"]:
                model_path = models_dir / CONFIG["model_name"]
                model.load_model(model_path)
                mlflow.log_param("loaded_model_path", str(model_path))
            elif not CONFIG["train"]:
                raise ValueError("model_path must be provided in CONFIG when train is False and predict is True")
            
            print(f"Loading test data from split {split}...")
            test_data = data_loader.load_data(None, None, test_file)['test']
            test_df = test_data[0]
            test_skipped = test_data[1]
            
            print(f"Making predictions...")
            predictions = model.predict(test_df, text_column=CONFIG["text_column"])
            
            output_file = predictions_dir / f"preds_{split}_{CONFIG['model_name']}.txt"
            print(f"Saving predictions to {output_file}...")
            model.save_predictions(
                predictions, 
                test_df, 
                test_skipped, 
                output_file,
                id_column=CONFIG["id_column"] if CONFIG["id_column"] else "PMID"
            )

            # Log predictions file
            mlflow.log_artifact(output_file)

        # Evaluate predictions
        if CONFIG["evaluate"] and CONFIG["predict"]:
            print(f"Evaluating predictions...")
            true_labels_path = labels_dir / test_file
            
            # Ensure the test file exists
            if not true_labels_path.exists():
                print(f"Warning: True labels file {true_labels_path} not found, skipping evaluation.")
            else:
                # Evaluate predictions
                eval_results = evaluate_multilabel(
                    true_labels_path=true_labels_path,
                    pred_labels_path=output_file,
                    all_labels=data_loader.all_labels
                )
                
                # Print evaluation results
                print_evaluation_results(eval_results)
                
                # Log evaluation metrics to MLflow
                for label, metrics in eval_results.items():
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)) and not pd.isna(metric_value):
                            mlflow.log_metric(f"{label}_{metric_name}", metric_value)
                
                # Save evaluation results to a JSON file
                eval_output_file = output_file.with_suffix('.eval.json')
                eval_output_file.write_text(pd.DataFrame(eval_results).to_json(indent=2))
                
                # Log the evaluation file
                mlflow.log_artifact(eval_output_file)
        
        print(f"MLflow run completed. Run ID: {mlflow.active_run().info.run_id}")
        print(f"View run details at: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{mlflow.active_run().info.run_id}")



if __name__ == "__main__":
    main()