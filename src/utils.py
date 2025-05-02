import mlflow
from pathlib import Path
from typing import Dict, Any
from mlflow.tracking import MlflowClient

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

def load_mlflow_params_by_experiment_and_run(experiment_name: str, run_name: str) -> Dict[str, Any]:
    """
    Load all parameters stored in the MLflow server.
    Args:
        run_name: Name of the MLflow run to load parameters from.
    Returns:
        Dictionary with all parameters.
    """

    # Initialize the MLflow client
    client = MlflowClient()

    # Get the experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return {}

    # Search for the run by name within the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    if runs:
        run = runs[0]  # Assuming the first match is the desired run
        print(f"Found run_id: {run.info.run_id} for run_name: {run_name}")
        return run.data.params
    else:
        print(f"No run found with name: {run_name} in experiment: {experiment_name}")
        return {}
    
# Example usage
if __name__ == "__main__":
    experiment_name = "biobert_classification"
    run_name = "goldhamster_20250425_174827"
    params = load_mlflow_params_by_experiment_and_run(experiment_name, run_name)
    print("Loaded parameters:", params)