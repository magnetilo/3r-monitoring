# 3r-monitoring
Data pipeline for monitoring animal experiments in biomedical literature.


## Set up your environment
Install uv:
- MacOS/Linux: ``curl -LsSf https://astral.sh/uv/install.sh | sh```
- Windows: ``powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```

Set up project from existing pyproject.toml:
````
uv sync
````

Set up project from scratch:
````
# Create a new directory for our project
uv init

# Create virtual environment and activate it
uv venv
source .venv/bin/activate

# Install dependencies
uv add requests ipykernel mlflow ...
````

## Setup environment with mamba
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

PubMed API:
- https://pmc.ncbi.nlm.nih.gov/tools/developers
- https://pmc.ncbi.nlm.nih.gov/tools/get-metadata

#### Download data:
````
# Download GoldHamster labels
uv run src/data/download_goldhamster_labels.py

# Download PubMed article metadata for GoldHamster labels
uv run src/data/download_pubmed_metadata.py
````

#### Notebooks:
- notebooks/01_stats_goldhamster_corpus.ipynb: Shows statistics over downloaded labels and article metadata