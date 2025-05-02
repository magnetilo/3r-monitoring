from pathlib import Path
import time
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
from mlflow.tracking import MlflowClient


# Get the project root directory
project_root = Path(__file__).resolve().parent.parent

# Add the project directory to the system path
import sys
sys.path.append(str(project_root))

# Local imports
from src.data_loaders import GoldhamsterDataLoader
from src.model import BioBERTClassifier
from src.evaluation import evaluate_multilabel, print_evaluation_results
from src.utils import load_mlflow_params_by_experiment_and_run

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Configuration parameters
CONFIG = {
    "mlflow_experiment_name": "goldhamster",         # MLflow experiment name
    "mlflow_run_name": "goldhamster-20250502-131034",          # MLflow run name (only used if train is False)
    "predictions_dir": project_root / "data/goldhamster/predictions",  # Directory to save predictions
    # "model_path": "goldhamster/goldhamster_model.h5",  # Relative path to load the model (only if model not loged in MLflow)
    "evaluate": True,           # Whether to evaluate predictions
}

def main():
    """Main function to train or predict with BioBERT model."""
     # Set up MLflow
    mlflow.set_experiment(CONFIG["mlflow_experiment_name"])

    # Load parameters from MLflow run_name
    params = load_mlflow_params_by_experiment_and_run(
        experiment_name=CONFIG["mlflow_experiment_name"],
        run_name=CONFIG["mlflow_run_name"]
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
    model_path = project_root / "models" / params["model_path"]
    model = BioBERTClassifier(
        model_path=model_path,
        labels=data_loader.all_labels,
        # max_length=CONFIG["model_max_length"],
        # learning_rate=CONFIG["model_learning_rate"],
        # batch_size=CONFIG["model_batch_size"],
        # epochs=CONFIG["model_epochs"],
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


    if evaluate:
        # Start MLflow run with existing run ID if available
        client = MlflowClient()
        experiment = client.get_experiment_by_name(CONFIG["mlflow_experiment_name"])
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{CONFIG['mlflow_run_name']}'"
        )
        if not runs:
            print(f"No run found with name '{CONFIG['mlflow_run_name']}' in experiment '{CONFIG['mlflow_experiment_name']}'")
            print("Starting a new run...")
            mlflow.start_run(run_name=CONFIG["mlflow_run_name"])
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