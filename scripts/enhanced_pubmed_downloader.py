import os
import csv
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
from datetime import datetime
import re

# API endpoints
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Output directory
OUTPUT_DIR = Path("data/country_publications")

# Country configurations
COUNTRIES = {
    "switzerland": ["Switzerland", "Swiss", "Schweiz", "Suisse", "Svizzera"],
    "germany": ["Germany", "German", "Deutschland", "Allemagne", "Germania"],
    "france": ["France", "French", "Francia", "Frankreich"]
}

class PubMedCountryDownloader:
    def __init__(self, email: str, tool_name: str = "CountryPubMedAnalysis"):
        self.email = email
        self.tool_name = tool_name
        self.session = requests.Session()
        # Add retry logic and rate limiting
        self.session.headers.update({
            'User-Agent': f'{tool_name} ({email})'
        })
    
    def search_by_country(self, country_terms: List[str], year_start: int = 2000, 
                         year_end: int = 2025, batch_size: int = 10000) -> List[str]:
        """Search for PMIDs by country affiliation with batching for large results."""
        # Create search term with OR logic for country variants
        country_query = " OR ".join([f'"{term}"[Affiliation]' for term in country_terms])
        search_term = f"({country_query}) AND ({year_start}[PDAT]:{year_end}[PDAT])"
        
        all_pmids = []
        retstart = 0
        
        while True:
            params = {
                "db": "pubmed",
                "term": search_term,
                "retmode": "json",
                "retmax": batch_size,
                "retstart": retstart,
                "usehistory": "y"
            }
            
            response = self.session.get(ESEARCH_URL, params=params)
            time.sleep(0.1)  # Rate limiting
            
            if response.status_code != 200:
                print(f"Search failed with status {response.status_code}")
                break
                
            data = response.json()
            search_result = data.get("esearchresult", {})
            pmids = search_result.get("idlist", [])
            
            if not pmids:
                break
                
            all_pmids.extend(pmids)
            retstart += batch_size
            
            # Check if we've gotten all results
            count = int(search_result.get("count", 0))
            if retstart >= count:
                break
                
            print(f"Retrieved {len(all_pmids)} PMIDs so far...")
        
        return all_pmids
    
    def extract_publication_date(self, article) -> Optional[str]:
        """Extract publication date from XML."""
        # Try PubDate first
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
    
    def extract_authors_with_affiliations(self, article) -> List[Dict[str, str]]:
        """Extract authors and their affiliations."""
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
    
    def extract_keywords(self, article) -> List[str]:
        """Extract keywords from the article."""
        keywords = []
        keyword_list = article.find(".//KeywordList")
        
        if keyword_list is not None:
            for keyword in keyword_list.findall("Keyword"):
                if keyword.text:
                    keywords.append(keyword.text)
        
        return keywords
    
    def determine_country_from_affiliations(self, authors: List[Dict]) -> List[str]:
        """Determine countries based on author affiliations."""
        found_countries = set()
        
        for author in authors:
            for affiliation in author.get("affiliations", []):
                for country, terms in COUNTRIES.items():
                    for term in terms:
                        if term.lower() in affiliation.lower():
                            found_countries.add(country)
        
        return list(found_countries)
    
    def fetch_detailed_metadata(self, pmid: str) -> Dict:
        """Fetch comprehensive metadata for a single PMID."""
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        }
        
        response = self.session.get(EFETCH_URL, params=params)
        time.sleep(0.1)  # Rate limiting
        
        if response.status_code != 200:
            return {"pmid": pmid, "error": f"HTTP {response.status_code}"}
        
        try:
            root = ET.fromstring(response.content)
            article = root.find(".//PubmedArticle")
            
            if article is None:
                return {"pmid": pmid, "error": "No article found"}
            
            # Extract title
            title = article.findtext(".//ArticleTitle", default="N/A")
            
            # Extract abstract
            abstract_elements = article.findall(".//Abstract/AbstractText")
            abstract = " ".join(
                "".join(el.itertext()).strip() for el in abstract_elements
            ) if abstract_elements else "N/A"
            
            # Extract MeSH terms
            mesh_terms = [
                mesh_term.text for mesh_term in article.findall(".//MeshHeading/DescriptorName")
                if mesh_term.text
            ]
            
            # Extract other metadata
            keywords = self.extract_keywords(article)
            authors = self.extract_authors_with_affiliations(article)
            pub_date = self.extract_publication_date(article)
            countries = self.determine_country_from_affiliations(authors)
            
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "mesh_terms": mesh_terms,
                "keywords": keywords,
                "authors": authors,
                "publication_date": pub_date,
                "detected_countries": countries,
                "download_timestamp": datetime.now().isoformat()
            }
            
        except ET.ParseError as e:
            return {"pmid": pmid, "error": f"XML parse error: {e}"}
        except Exception as e:
            return {"pmid": pmid, "error": f"Unexpected error: {e}"}
    
    def process_country(self, country: str, year_start: int = 2000, year_end: int = 2025):
        """Process all publications for a specific country."""
        print(f"\n=== Processing {country.upper()} ===")
        
        # Create country-specific output directory
        country_dir = OUTPUT_DIR / country
        country_dir.mkdir(parents=True, exist_ok=True)
        
        # Search for PMIDs
        print(f"Searching for {country} publications ({year_start}-{year_end})...")
        pmids = self.search_by_country(COUNTRIES[country], year_start, year_end)
        print(f"Found {len(pmids)} PMIDs for {country}")
        
        # Save PMID list
        pmid_file = country_dir / f"{country}_pmids_{year_start}_{year_end}.json"
        with open(pmid_file, 'w') as f:
            json.dump(pmids, f, indent=2)
        
        # Download metadata in batches
        print(f"Downloading metadata for {len(pmids)} publications...")
        
        with ThreadPoolExecutor(max_workers=3) as executor:  # Conservative threading
            futures = {
                executor.submit(self.fetch_detailed_metadata, pmid): pmid 
                for pmid in pmids[:1000]  # Limit for testing
            }
            
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc=f"Downloading {country} data"):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Save individual files as backup
                    pmid = result.get("pmid")
                    if pmid:
                        individual_file = country_dir / "individual" / f"{pmid}.json"
                        individual_file.parent.mkdir(exist_ok=True)
                        with open(individual_file, 'w') as f:
                            json.dump(result, f, indent=2)
                            
                except Exception as e:
                    pmid = futures[future]
                    print(f"Error processing PMID {pmid}: {e}")
                    results.append({"pmid": pmid, "error": str(e)})
        
        # Save consolidated results
        results_file = country_dir / f"{country}_metadata_{year_start}_{year_end}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Completed {country}: {len(results)} records processed")
        return results

def main():
    # Initialize downloader
    downloader = PubMedCountryDownloader(
        email="your.email@example.com",  # REPLACE WITH YOUR EMAIL
        tool_name="3R-Monitoring-Analysis"
    )
    
    # Process each country
    for country in COUNTRIES.keys():
        try:
            downloader.process_country(country, year_start=2020, year_end=2025)
        except Exception as e:
            print(f"Failed to process {country}: {e}")
            continue

if __name__ == "__main__":
    main()