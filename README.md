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

Install the r3_monitoring package in development mode:

````bash
uv pip install -e .
````

## Package Structure

```text
r3_monitoring/
├── models/
│   └── pubmedbert_classifier/    # PyTorch BioBERT model implementation
│       ├── model.py              # BioBERTClassifier and TextDataset classes
│       └── model_tensorflow_backup.py
├── data/
│   ├── goldhamster_loader.py     # GoldHamster dataset loader
│   └── schemata.py              # Data schema definitions
├── evaluation/
│   └── metrics.py               # Multi-label evaluation functions
└── utils/
    └── mlflow_helpers.py        # MLflow integration utilities
```

## Data Directory Structure

```text
data/
├── goldhamster/                  # GoldHamster corpus data
│   ├── docs/                     # PubMed article metadata (JSON files)
│   ├── labels/                   # Manual annotations and labels
│   ├── papers_old/              # Legacy paper data
│   └── predictions/             # Model prediction outputs
└── pubmed_scraping/             # Country-specific PubMed data collection
    ├── pubmed_metadata/         # Shared article metadata (organized by month)
    │   ├── 2010-01/            # Articles published in January 2010
    │   │   ├── 12345678.json   # Individual article metadata
    │   │   └── ...
    │   ├── 2010-02/            # February 2010 articles
    │   ├── ...                 # Additional months
    │   └── unknown/            # Articles with unclear publication dates
    ├── search_metadata_switzerland/  # Switzerland search results
    │   ├── pmids_2010_2025.json      # Complete PMID list for date range
    │   ├── pmids_2020-01.json        # Monthly PMID lists
    │   ├── pmids_2020-02.json
    │   ├── ...
    │   └── summary_2010_2025.json    # Search and download summary
    ├── search_metadata_germany/      # Germany search results (when added)
    ├── search_metadata_france/       # France search results (when added)
    └── scraping_summary.json         # Overall scraping statistics
```

## PubMedBERT-GoldHamster pipeline

### Adaptations from Original Source

This implementation has been adapted from [mariananeves/goldhamster](https://github.com/mariananeves/goldhamster) with several key improvements:

- **Framework Migration**: Converted from TensorFlow/Keras to **PyTorch** for better performance and modern ML practices
- **Structured Architecture**: Implemented clean, modular package structure:
  - `r3_monitoring.models.pubmedbert_classifier`: PyTorch model with `BioBERTClassifier` and `TextDataset` classes
  - `r3_monitoring.data`: Data loaders including `GoldhamsterDataLoader` for corpus handling
  - `r3_monitoring.evaluation`: Metrics and evaluation functions for multi-label classification
  - `r3_monitoring.utils`: MLflow helpers and utility functions
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

Download Goldhamster training data:

```` bash
# Download GoldHamster labels
uv run scripts/download_goldhamster_labels.py

# Download PubMed article metadata for GoldHamster labels
uv run scripts/download_pubmed_metadata.py
````

Train and evaluate model:

```` bash
# Start MLflow server with SQLite backend
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Specify training parameters in CONFIG section, then run training script 
uv run scripts/train_model.py

# Specify name of the MLflow run to evaluate and parameters, then run ecaluation script
uv run scripts/evaluate_model.py

# Inspect/compare results in MLflow in you browser at:
http://127.0.0.1:5000

# Clean up deleted MLflow runs (permanently delete runs marked as deleted in MLflow UI):
uv run scripts/mlflow_gc.py --dry-run  # Preview what will be deleted
uv run scripts/mlflow_gc.py            # Actually delete
````

- Note: For the ``data_split == '_full'``, training and testset are overlapping, hence the results are "too good".

Process PubMed articles:

The `process_pubmed_articles.py` script processes PubMed metadata JSON files into a comprehensive analysis table in Parquet format. It includes Swiss affiliation detection, Goldhamster model predictions, and support for incremental updates.

```` bash
# Inspect all possible parameters
uv run scripts/process_pubmed_articles.py --help

# Example call
uv run scripts/process_pubmed_articles.py \
  --countries switzerland \
  --start-date 2009-01 \
  --end-date 2025-12 \
  --calculate-goldhamster \
  --goldhamster-model-name PubMedBERT-20251204-120044 \
  --output-file data/results/pubmed_analysis.parquet \
  --upsert              # Appends new data to existing output-file (upsert mode)
````

The script outputs a Parquet file containing:

- Article metadata (PMID, title, abstract, publication date, journal, etc.)
- Swiss affiliation detection (boolean flag and matched institution names)
- Goldhamster model predictions (multi-label classification scores)
- Processing metadata (timestamps, model version, etc.)

#### Notebooks

- ``notebooks/01_stats_goldhamster_corpus.ipynb``: Shows statistics over downloaded labels and article metadata
- ``notebooks/02_pubmed_analysis_statistics.ipynb``: Create result tables and plots from processed parquet file
