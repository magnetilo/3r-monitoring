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
        experiment_name: Name of the MLflow experiment.
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

def download_mlflow_model_artifact(experiment_name: str, run_name: str, local_path: Path) -> Path:
    """
    Download a model artifact from MLflow by experiment and run name.
    Uses caching to avoid re-downloading existing models.
    Args:
        experiment_name: Name of the MLflow experiment.
        run_name: Name of the MLflow run containing the model.
        local_path: Local path where the model should be saved.
    Returns:
        Path to the downloaded model file or None if not found.
    """
    # Check if model already exists locally
    if local_path.exists():
        print(f"Model already exists locally: {local_path}")
        return local_path
    
    # Initialize the MLflow client
    client = MlflowClient()

    # Get the experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return None

    # Search for the run by name within the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    
    if runs:
        run = runs[0]
        run_id = run.info.run_id
        print(f"Found run_id: {run_id} for run_name: {run_name}")
        
        # Download model artifacts
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # List artifacts to find model file
            artifacts = client.list_artifacts(run_id, "models")
            if artifacts:
                model_artifact = artifacts[0]  # Assume first artifact is the model
                
                print(f"Downloading model artifact: {model_artifact.path}")
                # Download directly to the desired location using MLflow's download
                downloaded_path = client.download_artifacts(run_id, model_artifact.path, dst_path=str(local_path.parent))
                
                # The downloaded file might have the original name, so we may need to rename it
                downloaded_file = Path(downloaded_path)
                if downloaded_file != local_path:
                    import shutil
                    shutil.move(str(downloaded_file), str(local_path))
                
                print(f"Successfully downloaded model to: {local_path}")
                return local_path
            else:
                print(f"No model artifacts found in run: {run_name}")
                return None
                
        except Exception as e:
            print(f"Error downloading model from MLflow: {str(e)}")
            return None
    else:
        print(f"No run found with name: {run_name} in experiment: {experiment_name}")
        return None
    
# Example usage
if __name__ == "__main__":
    experiment_name = "biobert_classification"
    run_name = "goldhamster_20250425_174827"
    params = load_mlflow_params_by_experiment_and_run(experiment_name, run_name)
    print("Loaded parameters:", params)