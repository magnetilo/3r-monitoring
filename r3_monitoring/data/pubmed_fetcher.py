"""
PubMed data fetching and processing utilities.

This module provides reusable functionality for downloading and processing
PubMed article metadata for various use cases including dataset creation
and large-scale monitoring.
"""

import requests
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Set
from datetime import datetime
import json
from tqdm import tqdm
import re
from collections import defaultdict


class PubMedFetcher:
    """Core PubMed API client for fetching article metadata."""
    
    def __init__(self, email: str, tool_name: str = "3R-Monitoring"):
        """
        Initialize PubMed fetcher.
        
        Args:
            email: Required by NCBI for API identification
            tool_name: Tool identifier for API requests
        """
        self.email = email
        self.tool_name = tool_name
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': f'{tool_name} ({email})'
        })
        
        # API endpoints
        self.esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def search_pmids(self, query: str, max_results: int = None, 
                     batch_size: int = 9000) -> List[str]:
        """
        Search PubMed and return PMIDs matching the query.
        
        Note: PubMed ESearch has a limit of 9,999 records per query.
        For larger result sets, this method will return up to 9,999 PMIDs
        and print a warning.
        
        Args:
            query: PubMed search query
            max_results: Maximum number of results to return
            batch_size: Batch size for pagination
            
        Returns:
            List of PMIDs as strings (max 9,999 due to API limitation)
        """
        all_pmids = []
        retstart = 0
        
        while True:
            # Check if we're approaching the 9,999 limit
            if retstart >= 9999:
                print(f"Warning: PubMed ESearch limit reached. Retrieved {len(all_pmids)} PMIDs.")
                print("For more than 9,999 results, consider using more specific queries or EDirect.")
                break
            
            params = {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": min(batch_size, 9999 - retstart),  # Don't exceed the limit
                "retstart": retstart,
                "usehistory": "y"
            }
            
            response = self.session.get(self.esearch_url, params=params)
            time.sleep(0.1)  # Rate limiting
            
            if response.status_code != 200:
                print(f"Search failed with status {response.status_code}")
                # Check if it's the 9,999 limit error
                if response.status_code == 200:  # API returns 200 but with error in JSON
                    try:
                        error_data = response.json()
                        if "retstart" in error_data.get("esearchresult", {}).get("ERROR", ""):
                            print("Hit PubMed's 9,999 record limit")
                            break
                    except:
                        pass
                break
                
            data = response.json()
            search_result = data.get("esearchresult", {})
            
            # Check for API errors in the response
            if "ERROR" in search_result:
                print(f"PubMed API Error: {search_result['ERROR']}")
                if "retstart" in search_result["ERROR"]:
                    print("Hit PubMed's 9,999 record limit")
                break
            
            pmids = search_result.get("idlist", [])
            
            if not pmids:
                break
                
            all_pmids.extend(pmids)
            retstart += batch_size
            
            # Check limits
            if max_results and len(all_pmids) >= max_results:
                all_pmids = all_pmids[:max_results]
                break
                
            # Check if we've gotten all results
            count = int(search_result.get("count", 0))
            if retstart >= count:
                break
                
            if len(all_pmids) % 5000 == 0:
                print(f"Retrieved {len(all_pmids)} PMIDs so far...")
        
        # Print final statistics
        total_available = search_result.get("count", "unknown") if 'search_result' in locals() else "unknown"
        print(f"Search complete: Retrieved {len(all_pmids)} PMIDs (total available: {total_available})")
        
        return all_pmids
    
    def download_article_metadata(self, pmid: str, output_dir: Path) -> None:
        """
        Fetch and parse metadata for a single PMID and save to JSON file.
        
        Args:
            pmid: PubMed ID as string
            output_dir: Directory to save JSON file
        """
        self.download_batch_metadata([pmid], output_dir)
    
    def download_batch_metadata(self, pmids: List[str], output_dir: Path, batch_size: int = 200,
                              show_progress: bool = True) -> None:
        """
        Download and parse XML metadata for multiple PMIDs in batches, saving JSON files immediately.
        
        Args:
            pmids: List of PubMed IDs as strings
            output_dir: Directory to save JSON metadata files
            batch_size: Number of PMIDs per batch (max 500, recommended 200)
            show_progress: Whether to show progress bar
        """
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate number of batches
        num_batches = (len(pmids) + batch_size - 1) // batch_size
        
        # Process in batches with progress bar
        batch_iterator = range(0, len(pmids), batch_size)
        if show_progress:
            batch_iterator = tqdm(
                batch_iterator,
                desc="Downloading & saving JSON metadata",
                unit="batch",
                total=num_batches
            )
        
        for i in batch_iterator:
            batch_pmids = pmids[i:i+batch_size]
            batch_xml, _ = self._fetch_single_batch(batch_pmids)
            
            # Parse XML and save individual JSON files per PMID
            if batch_xml and not batch_xml.startswith("Error:"):
                try:
                    # Parse the XML to extract individual articles
                    root = ET.fromstring(batch_xml)
                    articles = root.findall(".//PubmedArticle")
                    
                    # Process each article and save as JSON
                    for article in articles:
                        pmid_element = article.find(".//PMID")
                        if pmid_element is not None:
                            pmid = pmid_element.text
                            
                            # Extract all metadata
                            metadata = {
                                "pmid": pmid,
                                "title": self._extract_title(article),
                                "abstract": self._extract_abstract(article),
                                "mesh_terms": self._extract_mesh_terms(article),
                                "keywords": self._extract_keywords(article),
                                "authors": self._extract_authors_with_affiliations(article),
                                "publication_date": self._extract_publication_date(article),
                                "journal": self._extract_journal_info(article),
                                "doi": self._extract_doi(article),
                                "download_timestamp": datetime.now().isoformat()
                            }
                            
                            # Determine year and month for directory organization
                            pub_date = metadata.get("publication_date")
                            if pub_date and len(pub_date) >= 7:
                                year = pub_date[:4]
                                month = pub_date[5:7]
                                date_dir = f"{year}-{month}"
                            elif pub_date and len(pub_date) >= 4:
                                year = pub_date[:4]
                                date_dir = f"{year}-unknown"
                            else:
                                date_dir = "unknown"
                            
                            # Create year-month subdirectory
                            month_dir = output_dir / date_dir
                            month_dir.mkdir(exist_ok=True)
                            
                            # Save to individual JSON file in year-month subdirectory
                            pmid_filename = month_dir / f"{pmid}.json"
                            with open(pmid_filename, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2)
                
                except ET.ParseError as e:
                    print(f"Error parsing batch XML: {e}")
                    continue
            
            # Rate limiting between batches
            time.sleep(0.5)

    def _fetch_single_batch(self, pmids: List[str]) -> Tuple[str, List[str]]:
        """Fetch raw XML for a single batch of PMIDs.
        
        Returns:
            Tuple of (xml_content, list_of_pmids_in_batch)
        """
        if not pmids:
            return "", []
        
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),  # Comma-separated PMIDs
            "retmode": "xml"
        }
        
        response = self.session.get(self.efetch_url, params=params)
        
        if response.status_code != 200:
            error_msg = f"Error: HTTP {response.status_code}"
            return error_msg, pmids
        
        return response.text, pmids
    
    def _extract_title(self, article) -> str:
        """Extract article title."""
        return article.findtext(".//ArticleTitle", default="N/A")
    
    def _extract_abstract(self, article) -> str:
        """Extract article abstract."""
        abstract_elements = article.findall(".//Abstract/AbstractText")
        if abstract_elements:
            return " ".join(
                "".join(el.itertext()).strip() for el in abstract_elements
            )
        return "N/A"
    
    def _extract_mesh_terms(self, article) -> List[str]:
        """Extract MeSH terms."""
        return [
            mesh_term.text for mesh_term in article.findall(".//MeshHeading/DescriptorName")
            if mesh_term.text
        ]
    
    def _extract_keywords(self, article) -> List[str]:
        """Extract author keywords."""
        keywords = []
        keyword_list = article.find(".//KeywordList")
        
        if keyword_list is not None:
            for keyword in keyword_list.findall("Keyword"):
                if keyword.text:
                    keywords.append(keyword.text)
        
        return keywords
    
    def _extract_authors_with_affiliations(self, article) -> List[Dict[str, Union[str, List[str]]]]:
        """Extract authors with their affiliations."""
        authors = []
        author_list = article.find(".//AuthorList")
        
        if author_list is not None:
            for author in author_list.findall("Author"):
                # Extract name
                last_name = author.findtext("LastName", "")
                first_name = author.findtext("ForeName", "")
                initials = author.findtext("Initials", "")
                
                # Extract affiliations
                affiliations = []
                for affiliation in author.findall("AffiliationInfo/Affiliation"):
                    if affiliation.text:
                        affiliations.append(affiliation.text)
                
                authors.append({
                    "last_name": last_name,
                    "first_name": first_name,
                    "initials": initials,
                    "affiliations": affiliations
                })
        
        return authors
    
    def _extract_publication_date(self, article) -> Optional[str]:
        """Extract publication date."""
        pub_date = article.find(".//PubDate")
        if pub_date is not None:
            year = pub_date.findtext("Year")
            month = pub_date.findtext("Month", "01")
            day = pub_date.findtext("Day", "01")
            
            # Handle month names
            month_map = {
                "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
            }
            if month in month_map:
                month = month_map[month]
            
            if year:
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        return None
    
    def _extract_journal_info(self, article) -> Dict[str, str]:
        """Extract journal information."""
        journal = article.find(".//Journal")
        if journal is not None:
            return {
                "title": journal.findtext(".//Title", "N/A"),
                "iso_abbreviation": journal.findtext(".//ISOAbbreviation", "N/A"),
                "issn": journal.findtext(".//ISSN", "N/A")
            }
        return {"title": "N/A", "iso_abbreviation": "N/A", "issn": "N/A"}
    
    def _extract_doi(self, article) -> Optional[str]:
        """Extract DOI if available."""
        article_ids = article.findall(".//ArticleId")
        for article_id in article_ids:
            if article_id.get("IdType") == "doi":
                return article_id.text
        return None


class CountryFilter:
    """Utility class for country-based filtering and detection."""
    
    # Country configurations
    COUNTRIES = {
        "switzerland": ["Switzerland", "Swiss", "Schweiz", "Suisse", "Svizzera"],
        "germany": ["Germany", "German", "Deutschland", "Allemagne", "Germania"], 
        "france": ["France", "French", "Francia", "Frankreich"]
    }
    
    @classmethod
    def build_country_query(cls, countries: List[str], year_start: int = None, 
                           year_end: int = None, additional_terms: str = None) -> str:
        """
        Build PubMed search query for specific countries.
        
        Args:
            countries: List of country keys from COUNTRIES dict
            year_start: Start year for date range
            year_end: End year for date range 
            additional_terms: Additional search terms
            
        Returns:
            PubMed search query string
        """
        # Build country affiliation terms
        country_queries = []
        for country in countries:
            if country in cls.COUNTRIES:
                terms = cls.COUNTRIES[country]
                country_query = " OR ".join([f'"{term}"[Affiliation]' for term in terms])
                country_queries.append(f"({country_query})")
        
        query = " OR ".join(country_queries)
        
        # Add date range if specified
        if year_start and year_end:
            query += f" AND ({year_start}[PDAT]:{year_end}[PDAT])"
        
        # Add additional terms if specified
        if additional_terms:
            query += f" AND ({additional_terms})"
            
        return query
    
    @classmethod
    def detect_countries_from_affiliations(cls, authors: List[Dict]) -> List[str]:
        """
        Detect countries based on author affiliations.
        
        Args:
            authors: List of author dictionaries with affiliations
            
        Returns:
            List of detected country keys
        """
        found_countries = set()
        
        for author in authors:
            for affiliation in author.get("affiliations", []):
                for country, terms in cls.COUNTRIES.items():
                    for term in terms:
                        if term.lower() in affiliation.lower():
                            found_countries.add(country)
        
        return list(found_countries)


class DatasetBuilder:
    """Utility class for building datasets from PubMed data."""
    
    def __init__(self, fetcher: PubMedFetcher, output_dir: Path):
        """
        Initialize dataset builder.
        
        Args:
            fetcher: PubMedFetcher instance
            output_dir: Directory to save dataset files
        """
        self.fetcher = fetcher
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_metadata(self, pmid: str, metadata: Dict, format: str = "json") -> None:
        """
        Save article metadata to file.
        
        Args:
            pmid: PubMed ID
            metadata: Article metadata dictionary
            format: Output format ("json" or "txt")
        """
        if format == "json":
            output_file = self.output_dir / f"{pmid}.json"
            with open(output_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        elif format == "txt":
            output_file = self.output_dir / f"{pmid}.txt"
            with open(output_file, 'w') as f:
                f.write(f"PMID: {metadata.get('pmid', 'N/A')}\n")
                f.write(f"Title: {metadata.get('title', 'N/A')}\n")
                f.write(f"Abstract: {metadata.get('abstract', 'N/A')}\n")
                f.write(f"MeSH Terms: {', '.join(metadata.get('mesh_terms', []))}\n")
    
    def build_from_pmid_list(self, pmids: List[str], skip_existing: bool = True, 
                           format: str = "json") -> None:
        """
        Build dataset from list of PMIDs using batch downloads.
        
        Args:
            pmids: List of PMIDs to process
            skip_existing: Skip PMIDs that already have saved files (checks for JSON files)
            format: Output format ("json" for metadata files)
        """
        # Use batch download which saves JSON metadata files automatically
        self.fetcher.download_batch_metadata(pmids, self.output_dir)


def load_pmids_from_search_metadata(
    countries: List[str],
    start_date: str,
    end_date: str,
    base_search_dir: Path
) -> Dict[str, Set[str]]:
    """
    Load PMIDs from search metadata files for given countries and date ranges.
    
    Args:
        countries: List of country names (e.g., ['switzerland', 'germany'])
        start_date: Start date in YYYY-MM format (e.g., '2020-01')
        end_date: End date in YYYY-MM format (e.g., '2024-12')
        base_search_dir: Base directory containing search_metadata_<country> folders
        
    Returns:
        Dictionary mapping country to set of PMIDs
        
    Example:
        pmids_by_country = load_pmids_from_search_metadata(
            countries=['switzerland'],
            start_date='2020-01',
            end_date='2024-12',
            base_search_dir=Path('data/pubmed_scraping')
        )
    """
    # Parse date range
    start_year, start_month = map(int, start_date.split('-'))
    end_year, end_month = map(int, end_date.split('-'))
    
    # Generate all year-month combinations in range
    target_months = []
    current_year, current_month = start_year, start_month
    
    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        target_months.append(f"{current_year}-{current_month:02d}")
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    
    # Load PMIDs for each country
    pmids_by_country = {}
    
    for country in countries:
        search_dir = base_search_dir / f"search_metadata_{country}"
        
        if not search_dir.exists():
            print(f"Warning: Search metadata directory not found: {search_dir}")
            pmids_by_country[country] = set()
            continue
        
        country_pmids = set()
        found_files = 0
        
        print(f"Loading PMIDs for {country} from {start_date} to {end_date}...")
        
        for month in target_months:
            pmid_file = search_dir / f"pmids_{month}.json"
            
            if pmid_file.exists():
                try:
                    with open(pmid_file, 'r') as f:
                        data = json.load(f)
                        pmids = data.get('pmids', [])
                        country_pmids.update(pmids)
                        found_files += 1
                        
                        if len(pmids) > 0:
                            print(f"  {month}: {len(pmids):,} PMIDs")
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Error reading {pmid_file}: {e}")
            else:
                print(f"  {month}: file not found")
        
        pmids_by_country[country] = country_pmids
        print(f"Total for {country}: {len(country_pmids):,} unique PMIDs from {found_files} files")
    
    return pmids_by_country


def load_metadata_for_pmids(
    pmids: Set[str], 
    metadata_base_dir: Path,
    show_progress: bool = True
) -> List[Dict]:
    """
    Load PubMed metadata JSON files for given PMIDs.
    
    Args:
        pmids: Set of PMIDs to load
        metadata_base_dir: Base directory containing year-month subdirectories with JSON files
        show_progress: Whether to show progress bar
        
    Returns:
        List of metadata dictionaries
        
    Example:
        metadata = load_metadata_for_pmids(
            pmids={'12345', '67890'},
            metadata_base_dir=Path('data/pubmed_scraping/pubmed_metadata')
        )
    """
    metadata_list = []
    found_count = 0
    missing_pmids = set(pmids.copy())
    
    # Get all subdirectories (year-month folders)
    subdirs = [d for d in metadata_base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if show_progress:
        subdirs = tqdm(subdirs, desc="Searching metadata directories")
    
    for subdir in subdirs:
        # Get all JSON files in this subdirectory
        json_files = list(subdir.glob("*.json"))
        
        for json_file in json_files:
            # Extract PMID from filename (e.g., "12345.json" -> "12345")
            pmid = json_file.stem
            
            if pmid in missing_pmids:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        metadata_list.append(metadata)
                        missing_pmids.remove(pmid)
                        found_count += 1
                        
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Warning: Error reading {json_file}: {e}")
        
        # Early exit if all PMIDs found
        if not missing_pmids:
            break
    
    if missing_pmids:
        print(f"Warning: Could not find metadata for {len(missing_pmids):,} PMIDs")
        if len(missing_pmids) <= 10:  # Show missing PMIDs if not too many
            print(f"Missing PMIDs: {sorted(list(missing_pmids))}")
    
    print(f"Loaded metadata for {found_count:,} out of {len(pmids):,} requested PMIDs")
    return metadata_list