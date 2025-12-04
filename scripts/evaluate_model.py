#!/usr/bin/env python3
"""
Script to evaluate a BioBERT model on the Goldhamster dataset using MLflow.
"""

from pathlib import Path
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


# Get the project root directory
project_root = Path(__file__).resolve().parent.parent

# Local imports
from r3_monitoring.data import GoldhamsterDataLoader
from r3_monitoring.models.pubmedbert_classifier import BioBERTClassifier
from r3_monitoring.evaluation import evaluate_multilabel, print_evaluation_results
from r3_monitoring.utils import load_mlflow_params_by_experiment_and_run, download_mlflow_model_artifact

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Configuration parameters
CONFIG = {
    "mlflow_experiment_name": "goldhamster-multilabel",         # MLflow experiment name
    "model_name": "PubMedBERT-20251204-153859",          # MLflow run name (only used if train is False)
    "predictions_dir": project_root / "data/goldhamster/predictions",  # Directory to save predictions
    # "model_path": "goldhamster/goldhamster_model.h5",  # Relative path to load the model (only if model not loged in MLflow)
    "evaluate": True,           # Whether to also evaluate or only predict
}

def main():
    """Main function to train or predict with BioBERT model."""
     # Set up MLflow
    mlflow.set_experiment(CONFIG["mlflow_experiment_name"])

    # Load parameters from MLflow run_name
    params = load_mlflow_params_by_experiment_and_run(
        experiment_name=CONFIG["mlflow_experiment_name"],
        run_name=CONFIG["model_name"]
    )
    if not params:
        params = {
            "data_name": "goldhamster",
            "data_split": 1,
            "data_use_titles": True,
            "data_use_abstracts": True,
            "data_use_mesh_terms": True,
            "data_text_column": "TEXT",
            "data_id_column": "PMID",
            "model_path": project_root / "models" / CONFIG["model_path"],
            "model_name": Path(CONFIG["model_path"]).stem,
        }
    
    # Initialize data loader
    papers_dir = project_root / "data" / params["data_name"] / "docs"
    labels_dir = project_root / "data" / params["data_name"] / "labels"
    data_loader = GoldhamsterDataLoader(
        papers_dir,
        labels_dir,
        use_titles=params["data_use_titles"],
        use_abstracts=params["data_use_abstracts"],
        use_mesh_terms=params["data_use_mesh_terms"]
    )

    # Initialize model
    print("Initializing model...")
    
    # Try to download model from MLflow first
    temp_model_path = project_root / "temp_models" / f"{CONFIG['model_name']}.pth"
    downloaded_model_path = download_mlflow_model_artifact(
        experiment_name=CONFIG["mlflow_experiment_name"],
        run_name=CONFIG["model_name"],
        local_path=temp_model_path
    )
    
    if downloaded_model_path:
        model_path = downloaded_model_path
        print(f"Using model downloaded from MLflow: {model_path}")
    else:
        # Fallback to traditional model path
        model_path = project_root / "models" / params.get("model_path", f"goldhamster/{CONFIG['model_name']}.pth")
        print(f"Using local model path: {model_path}")
    
    model = BioBERTClassifier(
        model_path=model_path,
        labels=data_loader.all_labels,
        load_model=True
    )

    # Calculate predictions
    split = params["data_split"]
    print(f"Loading test data from split {split}...")
    test_df, test_skipped = data_loader.create_dataframe_from_split(f"test{split}.txt")
    
    print(f"Making predictions...")
    predictions = model.predict(test_df, text_column=params["data_text_column"])

    pred_labels_path = Path(CONFIG["predictions_dir"]) / f"pred{split}_{params['model_name']}.txt"
    print(f"Saving predictions to {pred_labels_path}...")
    model.save_predictions(predictions, test_df, test_skipped, pred_labels_path, id_column=params["data_id_column"])


    if CONFIG["evaluate"]:
        # Start MLflow run with existing run ID if available
        client = MlflowClient()
        experiment = client.get_experiment_by_name(CONFIG["mlflow_experiment_name"])
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{CONFIG['model_name']}'"
        )
        if not runs:
            print(f"No run found with name '{CONFIG['model_name']}' in experiment '{CONFIG['mlflow_experiment_name']}'")
            print("Starting a new run...")
            mlflow.start_run(run_name=CONFIG["model_name"])
        else:
            run_id = runs[0].info.run_id
            print(f"Found existing run with ID: {run_id}")
            mlflow.start_run(run_id=run_id)

        # Evaluate predictions
        true_labels_path = labels_dir / f"test{split}.txt"
        eval_results = evaluate_multilabel(
            true_labels_path=true_labels_path,
            pred_labels_path=pred_labels_path,
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
        eval_output_file = pred_labels_path.with_suffix('.eval.json')
        eval_output_file.write_text(pd.DataFrame(eval_results).to_json(indent=2))
        
        # Log the evaluation file
        mlflow.log_artifact(eval_output_file)
        
        print(f"MLflow run completed. Run ID: {mlflow.active_run().info.run_id}")
        print(f"View run details at: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{mlflow.active_run().info.run_id}")

        # End the MLflow run
        mlflow.end_run()
        print("MLflow run ended.")


if __name__ == "__main__":
    main()