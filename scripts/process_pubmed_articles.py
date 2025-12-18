#!/usr/bin/env python3
"""
Process PubMed metadata JSON files into a comprehensive analysis table.

This script loads PMIDs from search metadata, processes their metadata,
calculates additional features (Swiss affiliations, Goldhamster predictions),
and saves results to a Parquet file with upsert capability.
"""

from pathlib import Path
from typing import List, Dict, Optional, Set, Any
import re
from datetime import datetime
import typer
from typing_extensions import Annotated
import pandas as pd
from tqdm import tqdm
from r3_monitoring.data.pubmed_fetcher import load_pmids_from_search_metadata, load_metadata_for_pmids
from r3_monitoring.utils.mlflow_helpers import load_biobert_from_mlflow
from r3_monitoring.data import GoldhamsterDataLoader


app = typer.Typer(help="Process PubMed articles metadata into analysis table")


def analyze_swiss_affiliations(authors: List[Dict[str, Any]]) -> Dict[str, bool]:
    """
    Analyze authors for Swiss affiliations.
    
    Args:
        authors: List of author dictionaries with affiliations
        
    Returns:
        Dictionary with Swiss affiliation flags
    """
    if not authors or len(authors) == 0:
        return {
            "any_author_has_swiss_affiliation": False,
            "first_author_has_swiss_affiliation": False,
            "last_author_has_swiss_affiliation": False
        }
    
    # Patterns to identify Swiss affiliations
    swiss_patterns = [
        r'\bswitzerland\b',
        r'\bsuisse\b', 
        r'\bschweiz\b',
        r'\bsvizzera\b',
        r'\bch-\d{4}\b',  # Swiss postal codes
        r'\b\d{4}\s+\w+,?\s+switzerland\b',
        # Major Swiss cities
        r'\bzurich\b', r'\bz??rich\b', r'\bzurich\b',
        r'\bgeneva\b', r'\bgen??ve\b', r'\bgenf\b',
        r'\bbasel\b', r'\bb??le\b', r'\bbasilea\b',
        r'\bbern\b', r'\bberne\b', r'\bberna\b',
        r'\blausanne\b', r'\blucerne\b', r'\bluzern\b',
        # Swiss institutions
        r'\beth\b', r'\bepfl\b', r'\buniversit[yi??].*z??rich\b',
        r'\buniversit[yi??].*geneva\b', r'\buniversit[yi??].*basel\b',
        r'\buniversit[yi??].*bern\b', r'\buniversit[yi??].*lausanne\b'
    ]
    
    def has_swiss_affiliation(author: Dict[str, Any]) -> bool:
        """Check if author has Swiss affiliation."""
        affiliations = author.get('affiliations', [])
        if not affiliations:
            return False
            
        for affiliation in affiliations:
            if not affiliation:
                continue
            affiliation_lower = affiliation.lower()
            for pattern in swiss_patterns:
                if re.search(pattern, affiliation_lower):
                    return True
        return False
    
    # Check affiliations
    any_swiss = any(has_swiss_affiliation(author) for author in authors)
    first_swiss = has_swiss_affiliation(authors[0]) if authors else False
    last_swiss = has_swiss_affiliation(authors[-1]) if len(authors) > 0 else False
    
    return {
        "any_author_has_swiss_affiliation": any_swiss,
        "first_author_has_swiss_affiliation": first_swiss,
        "last_author_has_swiss_affiliation": last_swiss
    }


def predict_goldhamster_labels(
    texts: List[str],
    model,
    text_column: str = "text"
) -> List[Dict[str, float]]:
    """
    Predict Goldhamster labels for texts using the loaded model.
    
    Args:
        texts: List of text strings to classify
        model: Loaded BioBERT model
        text_column: Column name for text data
        
    Returns:
        List of dictionaries with label probabilities
    """
    if not texts:
        return []
    
    # Create DataFrame for model prediction
    df = pd.DataFrame({text_column: texts})
    
    # Get predictions from model
    # predictions = model.predict(df, text_column=text_column)
    probas = model.predict_proba(df, text_column=text_column)
        
    return probas


def process_metadata_batch(
    metadata_batch: List[Dict[str, Any]],
    goldhamster_model = None,
    calculate_goldhamster: bool = False
) -> pd.DataFrame:
    """
    Process a batch of metadata records and calculate additional features.
    
    Args:
        metadata_batch: List of metadata dictionaries
        goldhamster_model: Loaded Goldhamster model (optional)
        calculate_goldhamster: Whether to calculate Goldhamster predictions
        
    Returns:
        DataFrame with processed records and additional features
    """
    records = []
    
    for metadata in tqdm(metadata_batch, desc="Processing batch", leave=False):
        # Extract basic metadata
        record = {
            'pmid': metadata.get('pmid'),
            'title': metadata.get('title'),
            'abstract': metadata.get('abstract'),
            'publication_date': metadata.get('publication_date'),
            'journal_title': metadata.get('journal', {}).get('title'),
            'journal_iso_abbreviation': metadata.get('journal', {}).get('iso_abbreviation'),
            'journal_issn': metadata.get('journal', {}).get('issn'),
            'doi': metadata.get('doi'),
            'mesh_terms': metadata.get('mesh_terms', []),
            'keywords': metadata.get('keywords', []),
            'authors': metadata.get('authors', []),
            'download_timestamp': metadata.get('download_timestamp')
        }
        
        # Calculate Swiss affiliation flags
        swiss_analysis = analyze_swiss_affiliations(record['authors'])
        record.update(swiss_analysis)
        
        # Add derived fields
        record['author_count'] = len(record['authors']) if record['authors'] else 0
        record['mesh_term_count'] = len(record['mesh_terms']) if record['mesh_terms'] else 0
        record['keyword_count'] = len(record['keywords']) if record['keywords'] else 0
        record['has_abstract'] = bool(record['abstract'] and record['abstract'].strip())
        
        # Extract year and month from publication date
        try:
            if record['publication_date']:
                pub_date = pd.to_datetime(record['publication_date'])
                record['publication_year'] = pub_date.year
                record['publication_month'] = pub_date.month
            else:
                record['publication_year'] = None
                record['publication_month'] = None
        except:
            record['publication_year'] = None
            record['publication_month'] = None
        
        records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Calculate Goldhamster predictions if requested and model available
    if calculate_goldhamster and goldhamster_model and not df.empty:
        print("Calculating Goldhamster predictions...")
        
        # Prepare text for classification (combining title, abstract, mesh terms)
        texts = []
        for _, row in df.iterrows():
            text_parts = []
            if row['title']:
                text_parts.append(row['title'])
            if row['abstract']:
                text_parts.append(row['abstract'])
            if row['mesh_terms']:
                text_parts.append(' '.join(row['mesh_terms']))
            
            combined_text = ' '.join(text_parts) if text_parts else ''
            texts.append(combined_text)
        
        # Get predictions
        predictions = predict_goldhamster_labels(texts, goldhamster_model)
        
        # Add predictions to DataFrame
        for label, preds in predictions.items():
            df[f'goldhamster_{label}'] = preds
    
    return df


@app.command()
def main(
    countries: Annotated[List[str], typer.Option(help="Countries to process (e.g., switzerland,germany)")] = ["switzerland"],
    start_date: Annotated[str, typer.Option(help="Start date in YYYY-MM format")] = "2020-01",
    end_date: Annotated[str, typer.Option(help="End date in YYYY-MM format")] = "2024-12",
    output_file: Annotated[Optional[Path], typer.Option(help="Output parquet file path (auto-generated if not specified)")] = None,
    base_data_dir: Annotated[Path, typer.Option(help="Base data directory")] = Path("data/pubmed_scraping"),
    batch_size: Annotated[int, typer.Option(help="Processing batch size")] = 1000,
    calculate_goldhamster: Annotated[bool, typer.Option(help="Calculate Goldhamster predictions")] = False,
    goldhamster_experiment: Annotated[str, typer.Option(help="MLflow experiment name for Goldhamster model")] = "goldhamster-multilabel",
    goldhamster_model_name: Annotated[str, typer.Option(help="MLflow run name for Goldhamster model")] = "PubMedBERT-20251204-153859",
    mlflow_uri: Annotated[str, typer.Option(help="MLflow tracking URI")] = "http://127.0.0.1:5000",
    temp_model_dir: Annotated[Path, typer.Option(help="Temporary directory for downloaded models")] = Path("temp_models"),
    upsert: Annotated[bool, typer.Option(help="Upsert existing parquet file instead of overwriting")] = False
):
    """
    Process PubMed articles metadata into comprehensive analysis table.
    
    This script loads PMIDs from search metadata, processes their metadata,
    calculates additional features, and saves results to a Parquet file.
    """
    print(f"Processing PubMed articles for countries: {countries}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Set default output file if not specified
    if output_file is None:
        if calculate_goldhamster and goldhamster_model_name:
            output_file = Path(f"data/results/pubmed_analysis_{goldhamster_model_name}.parquet")
        else:
            output_file = Path("data/results/pubmed_analysis.parquet")
    
    print(f"Output file: {output_file}")
    
    # Resolve paths
    base_data_dir = project_root / base_data_dir if not base_data_dir.is_absolute() else base_data_dir
    output_file = project_root / output_file if not output_file.is_absolute() else output_file
    temp_model_dir = project_root / temp_model_dir if not temp_model_dir.is_absolute() else temp_model_dir
    
    # Load PMIDs from search metadata
    print("\n=== Loading PMIDs from search metadata ===")
    pmids_by_country = load_pmids_from_search_metadata(
        countries=countries,
        start_date=start_date,
        end_date=end_date,
        base_search_dir=base_data_dir
    )
    
    # Combine all PMIDs
    all_pmids = set()
    for country, pmids in pmids_by_country.items():
        all_pmids.update(pmids)
    
    print(f"\nTotal unique PMIDs to process: {len(all_pmids):,}")
    
    if len(all_pmids) == 0:
        print("No PMIDs found. Exiting.")
        return
    
    # Load Goldhamster model if requested
    goldhamster_model = None
    if calculate_goldhamster:
        print("\n=== Loading Goldhamster model ===")
        try:
            # Load labels from data loader
            papers_dir = project_root / "data/goldhamster/docs"
            labels_dir = project_root / "data/goldhamster/labels"
            data_loader = GoldhamsterDataLoader(papers_dir, labels_dir)
            
            goldhamster_model = load_biobert_from_mlflow(
                experiment_name=goldhamster_experiment,
                run_name=goldhamster_model_name,
                labels=data_loader.all_labels,
                temp_model_dir=temp_model_dir,
                mlflow_tracking_uri=mlflow_uri
            )
            print("Goldhamster model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load Goldhamster model: {e}")
            print("Continuing without Goldhamster predictions")
            calculate_goldhamster = False
    
    # Load existing data if upsert is requested
    existing_df = None
    if upsert and output_file.exists():
        print(f"\n=== Loading existing parquet file for upsert ===")
        try:
            existing_df = pd.read_parquet(output_file)
            existing_pmids = set(existing_df['pmid'].astype(str))
            # Filter out PMIDs that already exist
            all_pmids = all_pmids - existing_pmids
            print(f"Existing records: {len(existing_df):,}")
            print(f"PMIDs to process after filtering existing: {len(all_pmids):,}")
        except Exception as e:
            print(f"Warning: Could not load existing parquet file: {e}")
            print("Will create new file instead")
            existing_df = None
    
    if len(all_pmids) == 0:
        print("No new PMIDs to process. Exiting.")
        return
    
    # Load metadata for PMIDs
    print("\n=== Loading metadata for PMIDs ===")
    metadata_dir = base_data_dir / "pubmed_metadata"
    all_metadata = load_metadata_for_pmids(
        pmids=all_pmids,
        metadata_base_dir=metadata_dir,
        show_progress=True
    )
    
    if len(all_metadata) == 0:
        print("No metadata found. Exiting.")
        return
    
    # Process metadata in batches
    print(f"\n=== Processing metadata in batches of {batch_size} ===")
    all_processed_dfs = []
    
    for i in tqdm(range(0, len(all_metadata), batch_size), desc="Processing batches"):
        batch = all_metadata[i:i + batch_size]
        
        processed_df = process_metadata_batch(
            metadata_batch=batch,
            goldhamster_model=goldhamster_model,
            calculate_goldhamster=calculate_goldhamster
        )
        
        if not processed_df.empty:
            all_processed_dfs.append(processed_df)
    
    if not all_processed_dfs:
        print("No data was processed successfully. Exiting.")
        return
    
    # Combine all processed data
    print("\n=== Combining processed data ===")
    final_df = pd.concat(all_processed_dfs, ignore_index=True)
    
    # Combine with existing data if upserting
    if existing_df is not None and not existing_df.empty:
        print("Combining with existing data...")
        final_df = pd.concat([existing_df, final_df], ignore_index=True)
        # Remove duplicates based on PMID, keeping the latest
        final_df = final_df.drop_duplicates(subset=['pmid'], keep='last')
    
    # Add processing metadata
    final_df['processing_timestamp'] = datetime.now().isoformat()
    
    # Clean data types before saving to parquet
    print(f"\n=== Cleaning data types for parquet compatibility ===")
    
    # Convert publication_date to datetime if it's not already
    if 'publication_date' in final_df.columns:
        final_df['publication_date'] = pd.to_datetime(final_df['publication_date'], errors='coerce')
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['author_count', 'mesh_term_count', 'keyword_count', 'publication_year', 'publication_month']
    for col in numeric_columns:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
    
    # Ensure boolean columns are properly typed
    boolean_columns = ['any_author_has_swiss_affiliation', 'first_author_has_swiss_affiliation',
                       'last_author_has_swiss_affiliation', 'has_abstract']
    for col in boolean_columns:
        if col in final_df.columns:
            final_df[col] = final_df[col].astype('boolean')
    
    # Convert list columns to strings for parquet compatibility
    list_columns = ['mesh_terms', 'keywords', 'authors']
    for col in list_columns:
        if col in final_df.columns:
            final_df[col] = final_df[col].astype(str)
    
    # Ensure Goldhamster prediction columns are float
    goldhamster_cols = [col for col in final_df.columns if col.startswith('goldhamster_')]
    for col in goldhamster_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
    
    print(f"Data types cleaned. DataFrame shape: {final_df.shape}")
    
    # Save to parquet
    print(f"\n=== Saving to parquet file ===")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        final_df.to_parquet(output_file, index=False, engine='pyarrow')
        print(f"Successfully saved {len(final_df):,} records to {output_file}")
        
        # Print summary statistics
        print(f"\n=== Summary Statistics ===")
        print(f"Total records: {len(final_df):,}")
        
        # Safe date range calculation
        if 'publication_date' in final_df.columns:
            valid_dates = final_df['publication_date'].dropna()
            if len(valid_dates) > 0:
                print(f"Date range: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}")
            else:
                print("No valid publication dates found")
        
        # Safe year coverage calculation
        if 'publication_year' in final_df.columns:
            valid_years = final_df['publication_year'].dropna()
            if len(valid_years) > 0:
                print(f"Years covered: {sorted(valid_years.unique().astype(int))}")
        
        print(f"Records with Swiss affiliations: {final_df['any_author_has_swiss_affiliation'].sum():,}")
        
        if calculate_goldhamster and goldhamster_model and goldhamster_cols:
            # Show Goldhamster prediction statistics
            print(f"Goldhamster predictions calculated for {len(goldhamster_cols)} labels")
            for col in goldhamster_cols:
                mean_prob = final_df[col].mean()
                if pd.notna(mean_prob):
                    print(f"  {col.replace('goldhamster_', '')}: {mean_prob:.3f} mean probability")
        
        print(f"\nFile size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        print(f"Error saving parquet file: {e}")
        return
    
    print("\nProcessing completed successfully!")


if __name__ == "__main__":
    app()