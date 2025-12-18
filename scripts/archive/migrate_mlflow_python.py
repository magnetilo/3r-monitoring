#!/usr/bin/env python3
"""
MLflow migration script using the Python API.
This script copies experiments from local mlruns directory to a new MLflow server.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import yaml


# Configuration
NEW_MLFLOW_URI = "http://127.0.0.1:5000"
LOCAL_MLRUNS_PATH = project_root / "mlruns"

def read_experiment_meta(exp_path):
    """Read experiment metadata from meta.yaml"""
    meta_file = exp_path / "meta.yaml"
    if not meta_file.exists():
        return None
    
    with open(meta_file, 'r') as f:
        return yaml.safe_load(f)

def read_run_meta(run_path):
    """Read run metadata from meta.yaml"""
    meta_file = run_path / "meta.yaml"
    if not meta_file.exists():
        return None
    
    with open(meta_file, 'r') as f:
        return yaml.safe_load(f)

def read_params(run_path):
    """Read parameters from params directory"""
    params = {}
    params_dir = run_path / "params"
    if params_dir.exists():
        for param_file in params_dir.iterdir():
            if param_file.is_file():
                params[param_file.name] = param_file.read_text().strip()
    return params

def read_metrics(run_path):
    """Read metrics from metrics directory"""
    metrics = {}
    metrics_dir = run_path / "metrics"
    if metrics_dir.exists():
        for metric_file in metrics_dir.iterdir():
            if metric_file.is_file():
                metric_name = metric_file.name
                metric_values = []
                
                with open(metric_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(' ')
                        if len(parts) >= 2:
                            timestamp = int(parts[0])
                            value = float(parts[1])
                            step = int(parts[2]) if len(parts) > 2 else 0
                            metric_values.append((timestamp, value, step))
                
                metrics[metric_name] = metric_values
    return metrics

def read_tags(run_path):
    """Read tags from tags directory"""
    tags = {}
    tags_dir = run_path / "tags"
    if tags_dir.exists():
        for tag_file in tags_dir.iterdir():
            if tag_file.is_file():
                tags[tag_file.name] = tag_file.read_text().strip()
    return tags

def copy_artifacts(run_path, mlflow_run):
    """Copy artifacts to the new run"""
    artifacts_dir = run_path / "artifacts"
    if artifacts_dir.exists() and any(artifacts_dir.iterdir()):
        try:
            mlflow.log_artifacts(str(artifacts_dir))
            return True
        except Exception as e:
            print(f"      ⚠️  Warning: Failed to copy artifacts: {e}")
            return False
    return True

def migrate_run(run_path, new_exp_id, old_run_id):
    """Migrate a single run to the new MLflow server"""
    print(f"    📄 Migrating run: {old_run_id}")
    
    # Read run metadata
    run_meta = read_run_meta(run_path)
    if not run_meta:
        print(f"      ❌ No metadata found for run {old_run_id}")
        return False
    
    # Read run data
    params = read_params(run_path)
    metrics = read_metrics(run_path)
    tags = read_tags(run_path)
    
    # Generate run name
    run_name = tags.get('mlflow.runName', f"migrated_run_{old_run_id[:8]}")
    
    # Extract original timestamps from metadata
    original_start_time = run_meta.get('start_time', None)
    original_end_time = run_meta.get('end_time', None)
    original_status = run_meta.get('status', 'FINISHED')
    
    try:
        # Set tracking URI to new server
        mlflow.set_tracking_uri(NEW_MLFLOW_URI)
        client = MlflowClient(NEW_MLFLOW_URI)
        
        # Create new run with original timestamps using the client API
        run = client.create_run(
            experiment_id=new_exp_id,
            start_time=original_start_time,
            tags={
                'mlflow.runName': run_name,
                'mlflow.user': tags.get('mlflow.user', 'migrated'),
                'mlflow.source.type': tags.get('mlflow.source.type', 'LOCAL'),
                'mlflow.source.name': tags.get('mlflow.source.name', 'migrated')
            }
        )
        
        run_id = run.info.run_id
        print(f"      📝 Created run with original timestamps: {run_id}")
        
        # Set the active run context
        with mlflow.start_run(run_id=run_id):
            # Log parameters
            for param_name, param_value in params.items():
                try:
                    mlflow.log_param(param_name, param_value)
                except Exception as e:
                    print(f"      ⚠️  Warning: Failed to log param {param_name}: {e}")
            
            # Log metrics
            for metric_name, metric_values in metrics.items():
                try:
                    for timestamp, value, step in metric_values:
                        mlflow.log_metric(metric_name, value, step=step)
                except Exception as e:
                    print(f"      ⚠️  Warning: Failed to log metric {metric_name}: {e}")
            
            # Log additional tags (skip system tags that were already set)
            for tag_name, tag_value in tags.items():
                if not tag_name.startswith('mlflow.'):
                    try:
                        mlflow.set_tag(tag_name, tag_value)
                    except Exception as e:
                        print(f"      ⚠️  Warning: Failed to set tag {tag_name}: {e}")
            
            # Copy artifacts
            copy_artifacts(run_path, run)
        
        # Update run end time if available
        if original_end_time and original_status:
            try:
                # Map status from number to string format
                status_map = {
                    '1': 'RUNNING',
                    '2': 'SCHEDULED', 
                    '3': 'FINISHED',
                    '4': 'FAILED',
                    '5': 'KILLED'
                }
                status_string = status_map.get(str(original_status), 'FINISHED')
                client.set_terminated(run_id, status=status_string, end_time=original_end_time)
                print(f"      🕒 Set original end time and status: {status_string}")
            except Exception as e:
                print(f"      ⚠️  Warning: Failed to set end time: {e}")
        
        print(f"      ✅ Successfully migrated run {old_run_id} -> {run_id}")
        return True
            
    except Exception as e:
        print(f"      ❌ Failed to migrate run {old_run_id}: {e}")
        return False

def migrate_experiment(exp_path, exp_id):
    """Migrate a single experiment to the new MLflow server"""
    print(f"  📁 Migrating experiment ID: {exp_id}")
    
    # Read experiment metadata
    exp_meta = read_experiment_meta(exp_path)
    if not exp_meta:
        print(f"    ❌ No metadata found for experiment {exp_id}")
        return False
    
    exp_name = exp_meta.get('name', f'experiment_{exp_id}')
    print(f"    📝 Experiment name: {exp_name}")
    
    # Create experiment in new server
    client = MlflowClient(NEW_MLFLOW_URI)
    
    try:
        new_exp_id = client.create_experiment(exp_name)
        print(f"    ✅ Created experiment '{exp_name}' with new ID: {new_exp_id}")
    except Exception as e:
        if "already exists" in str(e):
            existing_exp = client.get_experiment_by_name(exp_name)
            new_exp_id = existing_exp.experiment_id
            print(f"    ℹ️  Experiment '{exp_name}' already exists with ID: {new_exp_id}")
        else:
            print(f"    ❌ Failed to create experiment: {e}")
            return False
    
    # Find all runs in the experiment
    run_dirs = [d for d in exp_path.iterdir() 
                if d.is_dir() and d.name not in ['tags', 'meta.yaml']]
    
    print(f"    📊 Found {len(run_dirs)} runs to migrate")
    
    success_count = 0
    for run_dir in run_dirs:
        if migrate_run(run_dir, new_exp_id, run_dir.name):
            success_count += 1
    
    print(f"    ✅ Successfully migrated {success_count}/{len(run_dirs)} runs")
    return success_count > 0

def main():
    """Main migration function"""
    print("🚀 MLflow Python API Migration Script")
    print("=" * 40)
    print(f"Source: {LOCAL_MLRUNS_PATH}")
    print(f"Target: {NEW_MLFLOW_URI}")
    print()
    
    # Check if new server is accessible
    try:
        client = MlflowClient(NEW_MLFLOW_URI)
        experiments = client.search_experiments()
        print(f"✅ MLflow server accessible (found {len(experiments)} existing experiments)")
    except Exception as e:
        print(f"❌ Cannot connect to MLflow server: {e}")
        print("Please make sure the server is running with: mlflow server --host 127.0.0.1 --port 5000")
        return False
    
    # Check local mlruns directory
    if not LOCAL_MLRUNS_PATH.exists():
        print(f"❌ MLruns directory not found: {LOCAL_MLRUNS_PATH}")
        return False
    
    # Find experiments to migrate
    experiments_to_migrate = []
    for exp_dir in LOCAL_MLRUNS_PATH.iterdir():
        if exp_dir.is_dir() and exp_dir.name.isdigit():
            experiments_to_migrate.append({
                'id': exp_dir.name,
                'path': exp_dir
            })
    
    print(f"✅ Found {len(experiments_to_migrate)} experiments to migrate")
    
    if not experiments_to_migrate:
        print("No experiments found to migrate.")
        return True
    
    # List experiments
    print("\n📋 Experiments to migrate:")
    for exp in experiments_to_migrate:
        print(f"   - Experiment ID: {exp['id']}")
    
    # Ask for confirmation
    response = input(f"\nDo you want to migrate all {len(experiments_to_migrate)} experiments? (y/n): ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        return False
    
    print("\n🔄 Starting migration...")
    
    # Migrate each experiment
    success_count = 0
    for i, exp_info in enumerate(experiments_to_migrate, 1):
        print(f"\n📦 Migrating experiment {i}/{len(experiments_to_migrate)}")
        
        if migrate_experiment(exp_info['path'], exp_info['id']):
            success_count += 1
    
    print(f"\n🎉 Migration completed!")
    print(f"✅ Successfully migrated {success_count}/{len(experiments_to_migrate)} experiments")
    print(f"🌐 View your experiments at: {NEW_MLFLOW_URI}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n👋 Migration cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)