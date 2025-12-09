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

### Data Organization Notes

- **Monthly Organization**: JSON metadata files are organized by publication month (`YYYY-MM/`) to optimize file system performance (~2,500 files per directory instead of 30,000+ per year)
- **Shared Metadata**: All countries share the same `pubmed_metadata/` directory since articles can be relevant to multiple countries
- **Search Results**: Country-specific directories contain PMID lists, search parameters, and download statistics

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

Process PubMed articles:

The `process_pubmed_articles.py` script processes PubMed metadata JSON files into a comprehensive analysis table in Parquet format. It includes Swiss affiliation detection, Goldhamster model predictions, and support for incremental updates.

Basic usage - process all Swiss data:

```` bash
uv run scripts/process_pubmed_articles.py --countries switzerland
````

Process specific date range:

```` bash
uv run scripts/process_pubmed_articles.py --countries switzerland \
  --start-date 2009-01 \
  --end-date 2025-12 \
  --calculate-goldhamster \
  --goldhamster-model-name PubMedBERT-20251204-120044 \
  --output-file data/results/pubmed_analysis.parquet
````

Process with custom output file and model:

```` bash
uv run scripts/process_pubmed_articles.py --countries switzerland \
  --output-file data/swiss_articles_2024.parquet \
  --goldhamster-model-run-name "goldhamster-final-model"
````

Process with batch processing for memory efficiency:

```` bash
uv run scripts/process_pubmed_articles.py --countries switzerland \
  --batch-size 500 \
  --start-date 2022-01 \
  --end-date 2024-12
````

Enable Goldhamster predictions (disabled by default):

```` bash
uv run scripts/process_pubmed_articles.py --countries switzerland \
  --calculate-goldhamster-predictions \
  --start-date 2022-01 \
  --end-date 2024-12
````

Append new data to existing analysis (upsert mode):

```` bash
# First run creates the file
uv run scripts/process_pubmed_articles.py --countries switzerland --end-date 2023-12

# Second run adds new data without duplicates
uv run scripts/process_pubmed_articles.py --countries switzerland --start-date 2024-01 --end-date 2024-12 --upsert
````

The script outputs a Parquet file containing:

- Article metadata (PMID, title, abstract, publication date, journal, etc.)
- Swiss affiliation detection (boolean flag and matched institution names)
- Goldhamster model predictions (multi-label classification scores)
- Processing metadata (timestamps, model version, etc.)

#### Notebooks

- ``notebooks/01_stats_goldhamster_corpus.ipynb``: Shows statistics over downloaded labels and article metadata
