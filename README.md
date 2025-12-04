# 3r-monitoring

Data pipeline for monitoring animal experiments in biomedical literature.

## Installation

Install uv: Use curl to download the script and execute it with sh:

````bash
curl -LsSf https://astral.sh/uv/install.sh | sh
````

Create venv and install deps with uv:

````bash
uv venv
uv sync
````

## BioPubMedBERT-GoldHamster pipeline

### Adaptations from Original Source

This implementation has been adapted from [mariananeves/goldhamster](https://github.com/mariananeves/goldhamster) with several key improvements:

- **Framework Migration**: Converted from TensorFlow/Keras to **PyTorch** for better performance and modern ML practices
- **Structured Architecture**: Implemented clean, modular classes in `src/model.py`:
  - `TextDataset`: Custom PyTorch Dataset class for efficient text preprocessing and batching
  - `BioBERTClassifier`: Complete PyTorch model with integrated training, validation, and inference
- **Enhanced Training**: Added early stopping with best model checkpointing to prevent overfitting
- **MLflow Integration**: Full experiment tracking with SQLite backend for reproducible ML workflows
- **Optimized Performance**: Leverages Apple Metal Performance Shaders (MPS) for M1/M2/M3/M4 acceleration

Links:

- Paper: https://link.springer.com/article/10.1186/s13326-023-00292-w
- Corpus: https://doi.org/10.5281/zenodo.7152295
- Original Source: https://github.com/mariananeves/goldhamster
- Model: https://huggingface.co/SMAFIRA/goldhamster

#### PubMed API

URLs:

- https://pmc.ncbi.nlm.nih.gov/tools/developers
- https://pmc.ncbi.nlm.nih.gov/tools/get-metadata
- Example API request: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=30955825

#### Run Scripts

Download data:

```` bash
# Download GoldHamster labels
python scripts/download_goldhamster_labels.py

# Download PubMed article metadata for GoldHamster labels
python scripts/download_pubmed_metadata.py
````

Train model:

- Start MLflow server with SQLite backend: ``mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db``
- In CONFIG section specify different training parameters.
- Run training script, this saves the trained model to ``models/``, and logs parameters to MLflow:

```` bash
uv run scripts/train_model.py
````

Predict and evaluate trained model:

- Start MLflow server with SQLite backend: ``mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db``
- In CONFIG section specify the name of the MLflow run from the model you want to use for prediction/evaluation.
- If ``evaluate = True``, the evaluation results are logged to the same MLflow experiment run.
- Run evaluation script:

```` bash
uv run scripts/evaluate_model.py
````

Inspect/compare results in MLflow:

- Start MLflow server with SQLite backend: ``mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db``
- Go to ``http://127.0.0.1:5000/`` in your browser.
- Select experiment and choose columns, e.g.:
  - Filter for ``f1`` columns and select all
  - Select parameters ``data_split`` and ``model_epochs``
  - Deselect 'Dataset', 'Source', and 'Models'
- Note: For the ``data_split == '_full'``, training and testset are overlapping, hence the results are "too good".

Clean up deleted runs:

- Permanently delete runs marked as deleted in MLflow UI:

```` bash
uv run scripts/mlflow_gc.py --dry-run  # Preview what will be deleted
uv run scripts/mlflow_gc.py            # Actually delete
````

#### Notebooks

- ``notebooks/01_stats_goldhamster_corpus.ipynb``: Shows statistics over downloaded labels and article metadata
