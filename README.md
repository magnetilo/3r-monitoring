# 3r-monitoring
Data pipeline for monitoring animal experiments in biomedical literature.


## Setup environment with mamba
- Install miniforge: https://github.com/conda-forge/miniforge
- After installation, you should be able to use mamba commands.
- More information on mamba package manager: https://mamba.readthedocs.io/
````
mamba create -y --name 3r-monitoring python=3.11
mamba activate 3r-monitoring
mamba install --yes --file requirements.txt
````


## BioPubMedBERT-GoldHamster pipeline
Links:
- Paper: https://link.springer.com/article/10.1186/s13326-023-00292-w
- Corpus: https://doi.org/10.5281/zenodo.7152295
- Source code: https://github.com/mariananeves/goldhamster
- Model: https://huggingface.co/SMAFIRA/goldhamster

#### PubMed API:
URLs:
- https://pmc.ncbi.nlm.nih.gov/tools/developers
- https://pmc.ncbi.nlm.nih.gov/tools/get-metadata
- Example API request: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=30955825


#### Run Scripts:
Download data:
````
# Download GoldHamster labels
python scripts/download_goldhamster_labels.py

# Download PubMed article metadata for GoldHamster labels
python scripts/download_pubmed_metadata.py
````

Train model:
- Start MLflow server in second terminal with ``mlflow ui``.
- In CONFIG section specify different training parameters.
- Run training script, this saves the trained model to ``models/``, and logs parameters to MLflow:
````
python scripts/train_model.py
````

Predict and evaluate trained model:
- Start MLflow server in second terminal with ``mlflow ui``.
- In CONFIG section specify the name of the MLflow run from the model you want to use for prediction/evaluation.
- If ``evaluate = True``, the evaluation results are logged to the same MLflow experiment run.
- Run evaluation script, and inspect/compare results in MLflow ``http://127.0.0.1:5000/``:
````
python scripts/evaluate_model.py
````

#### Notebooks:
- ``notebooks/01_stats_goldhamster_corpus.ipynb``: Shows statistics over downloaded labels and article metadata