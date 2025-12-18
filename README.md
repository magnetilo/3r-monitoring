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

## Scripts

### Download Goldhamster training data

```` bash
# Download GoldHamster labels
uv run scripts/download_goldhamster_labels.py

# Download PubMed article metadata for GoldHamster labels
uv run scripts/download_pubmed_metadata.py
````

### Train and evaluate model

```` bash
# Start MLflow server with SQLite backend
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Specify training parameters in CONFIG section, then run training script 
uv run scripts/train_model.py

# Specify name of the MLflow run to evaluate and parameters, then run ecaluation script
uv run scripts/evaluate_model.py

# Inspect/compare results in MLflow in you browser at:
http://127.0.0.1:5000

# Clean up deleted MLflow runs (permanently delete runs marked as deleted in MLflow UI):
uv run mlflow gc --tracking-uri http://127.0.0.1:5000 --backend-store-uri sqlite:///mlflow.db
````

- Note: For the ``data_split == '_full'``, training and testset are overlapping, hence the results are "too good".

### Scrape PubMed articles

```` bash
uv run scripts/scrape_pubmed_articles.py
````

### Process PubMed articles

The `process_pubmed_articles.py` script processes PubMed metadata JSON files into a comprehensive analysis table in Parquet format. It includes Swiss affiliation detection, Goldhamster model predictions, and support for incremental updates.

```` bash
# Start MLflow server with SQLite backend (for loading the trained model)
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Inspect all possible parameters
uv run scripts/process_pubmed_articles.py --help

# Example call
uv run scripts/process_pubmed_articles.py \
  --countries switzerland \
  --start-date 2010-01 \
  --end-date 2025-12 \
  --calculate-goldhamster \
  --goldhamster-model-name PubMedBERT-20251204-120044 \
  --upsert              # Appends new data to existing output-file (upsert mode)
````

The script outputs a Parquet file containing:

- Article metadata (PMID, title, abstract, publication date, journal, etc.)
- Swiss affiliation detection (boolean flag and matched institution names)
- Goldhamster model predictions (multi-label classification scores)
- Processing metadata (timestamps, model version, etc.)

### Notebooks

- ``notebooks/01_stats_goldhamster_corpus.ipynb``: Shows statistics over downloaded labels and article metadata
- ``notebooks/02_pubmed_analysis_statistics.ipynb``: Create result tables and plots from processed parquet file

#### Statistical Analysis and Visualization

The `02_pubmed_analysis_statistics.ipynb` notebook generates comprehensive plots showing temporal trends in animal experimentation research with Swiss affiliations. The analysis implements two complementary approaches for counting and uncertainty estimation:

##### Approach 1: Binary Classification with Test Set Uncertainty

- **Counts**: Binary classification using threshold (typically 0.3 for better recall-precision balance)
- **Uncertainty**: Binomial standard deviation √(n × accuracy × (1 - accuracy)) using test set accuracy
- **Foundation**: Models each prediction as correct/incorrect based on empirical model performance
- **Advantages**: Uses real model performance metrics, accounts for systematic model errors

##### Approach 2: Probability Sum with Poisson Binomial Uncertainty

- **Counts**: Sum of predicted probabilities from the Goldhamster model
- **Uncertainty**: Poisson binomial standard deviation √Σp_i(1-p_i), where p_i are individual prediction probabilities
- **Foundation**: Follows Poisson binomial distribution (sum of independent Bernoulli trials with different success probabilities) which converges to Gaussian for large samples ([reference](https://math.stackexchange.com/questions/2375886/approximating-poisson-binomial-distribution-with-normal-distribution))
- **Advantages**: Uses full probability information, theoretically elegant for well-calibrated models

##### Key Insights from Comparison:

- Probability sum approach generally provides more accurate expected counts
- Binary classification with 0.5 threshold tends to undercount due to precision-recall imbalance (high precision, lower recall)
- Test set accuracy-based uncertainty estimates are more robust to model calibration issues
- Both approaches show temporal trends with statistically rigorous uncertainty quantification

The visualization displays both approaches side-by-side with transparent uncertainty bands, enabling direct comparison of counting methodologies and their associated confidence intervals.

## (Possible) next steps

- Quality checks for download filters: I didn't do much checking yet. But recognized, e.g., that some downloaded publication dates are outside the requested time period. Also, I didn't check "Switzerland API filter" too much yet.
- Quality checks on the classified Goldhamster-Labels would also be good: Ideally by a 3R-expert. Are we classifying correctly in general..?
- Thoughts on mapping the Goldhamster-Labels. I think, you are especially interested in "in vivo" label, correct?
  - see Paper, section "Definition of the annotation schema" for definitions
- Train an "in vivo yes/no" binary model as well an compare performance
- Mapping/clustering MeSH terms: I didn't do much on this so far. This requires some research and better understanding of MeSH terms. Also the "labels by MeSH term" excel in the attachment can be used to find MeSH terms with many "in vivo" labels and others with few.
  - Are there MeSH or Journals that we want to exclude a priori from processing?
  - Note, there are also many articles without MeSH terms, which probably need to be processed anyway.
- Thinking about visualizations and results
  - Do we already have all information we need?
  - Is something missing?
- Tidy up PubMed article scraping & processing: The code files are still a bit messy yet...
