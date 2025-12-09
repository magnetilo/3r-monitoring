import mlflow
from pathlib import Path
from typing import Dict, Any, List, Optional
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
    
def load_biobert_from_mlflow(
    experiment_name: str,
    run_name: str,
    labels: List[str],
    temp_model_dir: Path,
    mlflow_tracking_uri: str = "http://127.0.0.1:5000",
    device: Optional[str] = None
):
    """
    Load a BioBERT model from MLflow experiment.
    
    Args:
        experiment_name: Name of the MLflow experiment
        run_name: Name of the MLflow run
        labels: List of labels for the model
        temp_model_dir: Directory to temporarily store downloaded model
        mlflow_tracking_uri: MLflow tracking server URI
        device: Device to use ('cuda', 'mps', 'cpu' or None for auto)
        
    Returns:
        Loaded BioBERTClassifier model
        
    Raises:
        ValueError: If model run is not found or model download fails
        
    Example:
        model = load_biobert_from_mlflow(
            experiment_name="goldhamster-multilabel",
            run_name="PubMedBERT-20251204-153859",
            labels=["replacement", "reduction", "refinement"],
            temp_model_dir=Path("temp_models")
        )
    """
    # Import here to avoid circular imports
    from r3_monitoring.models.pubmedbert_classifier.model import BioBERTClassifier
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Get MLflow client
    client = MlflowClient()
    
    # Find experiment
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
    except Exception as e:
        raise ValueError(f"Error finding experiment '{experiment_name}': {e}")
    
    # Find run by name
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    
    if not runs:
        raise ValueError(f"No run found with name '{run_name}' in experiment '{experiment_name}'")
    
    run = runs[0]
    run_id = run.info.run_id
    
    print(f"Found MLflow run: {run_name} (ID: {run_id})")
    
    # Setup model caching
    temp_model_dir.mkdir(parents=True, exist_ok=True)
    temp_model_path = temp_model_dir / f"{run_name}.pth"
    
    # Check if model already exists (caching mechanism)
    if temp_model_path.exists():
        print(f"Using cached model: {temp_model_path}")
    else:
        print(f"Model not found in cache, downloading from MLflow...")
        try:
            # Download the model file
            artifact_path = client.download_artifacts(run_id, "model.pth", str(temp_model_dir))
            
            # Rename to expected filename
            downloaded_path = Path(artifact_path)
            if downloaded_path != temp_model_path:
                downloaded_path.rename(temp_model_path)
            
            print(f"Model downloaded and cached to: {temp_model_path}")
            
        except Exception as e:
            raise ValueError(f"Error downloading model from MLflow: {e}")
    
    # Load model parameters from MLflow run
    params = run.data.params
    
    # Initialize and load the model
    model = BioBERTClassifier(
        labels=labels,
        model_path=temp_model_path,
        model_pretrainded=params.get("model_pretrained", "dmis-lab/biobert-v1.1"),
        max_length=int(params.get("model_max_length", "256")),
        learning_rate=float(params.get("model_learning_rate", "1e-4")),
        batch_size=int(params.get("model_batch_size", "32")),
        epochs=int(params.get("model_epochs", "10")),
        dropout_rate=float(params.get("model_dropout_rate", "0.1")),
        load_model=True,
        device=device
    )
    
    # Print the device being used by the model
    model_device = next(model.parameters()).device
    print("BioBERT model loaded successfully from MLflow")
    print(f"Model device: {model_device}")
    return model


# Example usage
if __name__ == "__main__":
    experiment_name = "biobert_classification"
    run_name = "goldhamster_20250425_174827"
    params = load_mlflow_params_by_experiment_and_run(experiment_name, run_name)
    print("Loaded parameters:", params)