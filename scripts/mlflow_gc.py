#!/usr/bin/env python3
"""
Simple script to clean up deleted MLflow runs using MLflow's built-in garbage collection.

This script uses MLflow's native 'gc' command to permanently delete runs that are marked 
as deleted. This is the recommended way to clean up deleted runs.

Usage:
    python scripts/mlflow_gc.py [--tracking-uri URI] [--dry-run]

Arguments:
    --tracking-uri: MLflow tracking URI (default: http://127.0.0.1:5000)
    --dry-run: Show what would be deleted without actually deleting
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def get_deleted_runs(tracking_uri: str):
    """Get deleted runs from MLflow server."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        from mlflow.entities import ViewType
        
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        
        # Get all experiments
        experiments = client.search_experiments(view_type=ViewType.ALL)
        
        deleted_runs = []
        deleted_experiments = []
        
        for exp in experiments:
            if exp.lifecycle_stage == 'deleted':
                deleted_experiments.append((exp.experiment_id, exp.name))
            
            # Get deleted runs from this experiment
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id], 
                run_view_type=ViewType.DELETED_ONLY
            )
            
            for run in runs:
                run_name = run.data.tags.get('mlflow.runName', 'Unnamed')
                deleted_runs.append((run.info.run_id, run_name, exp.experiment_id, exp.name))
        
        return deleted_runs, deleted_experiments
    
    except Exception as e:
        print(f"⚠️  Could not query deleted runs: {e}")
        return [], []


def run_mlflow_gc(tracking_uri: str, dry_run: bool = False) -> bool:
    """
    Run MLflow garbage collection to permanently delete runs marked as deleted.
    
    Args:
        tracking_uri: MLflow tracking server URI
        dry_run: If True, only show what would be deleted
        
    Returns:
        True if successful, False otherwise
    """
    print("🗑️  MLflow Garbage Collection")
    print("=" * 30)
    print(f"Tracking URI: {tracking_uri}")
    print(f"Mode: {'DRY RUN' if dry_run else 'ACTUAL DELETION'}")
    print()
    
    # Get deleted runs and experiments
    deleted_runs, deleted_experiments = get_deleted_runs(tracking_uri)
    
    print(f"🔍 Found {len(deleted_runs)} deleted runs")
    print(f"🔍 Found {len(deleted_experiments)} deleted experiments")
    print()
    
    if not deleted_runs and not deleted_experiments:
        print("✨ No deleted runs or experiments found. Nothing to clean up!")
        return True
    
    # Show what will be deleted
    if deleted_runs:
        print("📋 Deleted runs that will be permanently removed:")
        current_exp = None
        for run_id, run_name, exp_id, exp_name in deleted_runs:
            if exp_name != current_exp:
                current_exp = exp_name
                print(f"  📁 Experiment: {exp_name or 'Unnamed'} (ID: {exp_id})")
            print(f"    🏃 {run_name} ({run_id[:8]}...)")
        print()
    
    if deleted_experiments:
        print("📋 Deleted experiments that will be permanently removed:")
        for exp_id, exp_name in deleted_experiments:
            print(f"  📁 {exp_name or 'Unnamed'} (ID: {exp_id})")
        print()
    
    # Build the MLflow gc command
    cmd = ["uv", "run", "mlflow", "gc"]
    
    if dry_run:
        print("🔍 DRY RUN: No actual deletion will be performed.")
        print(f"Command that would be executed: MLFLOW_TRACKING_URI={tracking_uri} {' '.join(cmd)}")
        return True
    
    try:
        # Set environment variable for MLflow tracking URI
        import os
        env = os.environ.copy()
        env['MLFLOW_TRACKING_URI'] = tracking_uri
        
        print("🚀 Running MLflow garbage collection...")
        print(f"Command: {' '.join(cmd)}")
        print()
        
        # Run the command
        result = subprocess.run(
            cmd,
            env=env,
            cwd=project_root,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        
        print()
        print("✅ MLflow garbage collection completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ MLflow gc command failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def check_mlflow_server(tracking_uri: str) -> bool:
    """Check if MLflow server is accessible."""
    try:
        # Use MLflow client to check server
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        experiments = mlflow.search_experiments()
        print(f"✅ MLflow server accessible (found {len(experiments)} experiments)")
        return True
    except Exception as e:
        print(f"❌ Cannot connect to MLflow server: {e}")
        print("Please make sure the server is running.")
        return False


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Clean up deleted MLflow runs using MLflow's garbage collection"
    )
    parser.add_argument(
        "--tracking-uri", 
        default="http://127.0.0.1:5000",
        help="MLflow tracking server URI (default: http://127.0.0.1:5000)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    
    args = parser.parse_args()
    
    try:
        # Check if MLflow server is accessible
        if not check_mlflow_server(args.tracking_uri):
            sys.exit(1)
        
        print()
        
        # Run garbage collection
        success = run_mlflow_gc(args.tracking_uri, args.dry_run)
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n👋 Cleanup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()