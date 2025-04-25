import os
import csv
import requests
from pathlib import Path
from typing import List, Dict
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json

# Directory containing the TSV files
LABELS_DIR = Path("data/goldhamster/labels")
OUTPUT_DIR = Path("data/goldhamster/papers")
PUBMED_API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def read_tsv_files(directory: Path) -> List[str]:
    """Read all TSV files in the directory and extract PubMed IDs."""
    pmids = []
    for file in directory.glob("*.txt"):
        with open(file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if row:  # Ensure the row is not empty
                    pmids.append(row[0])  # First column is the PMID
    return pmids

def fetch_pubmed_data(pmid: str) -> Dict[str, str]:
    """Fetch title, abstract, and MeSH terms for a given PubMed ID using the PubMed API."""
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }
    response = requests.get(PUBMED_API_URL, params=params)
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        article = root.find(".//PubmedArticle")
        if article is not None:
            # Extract title
            title = article.findtext(".//ArticleTitle", default="N/A")
            
            # Extract abstract, handling nested tags
            abstract_elements = article.findall(".//Abstract/AbstractText")
            abstract = " ".join(
                "".join(el.itertext()).strip() for el in abstract_elements
            ) if abstract_elements else "N/A"
            
            # Extract MeSH terms
            mesh_terms = [
                mesh_term.text for mesh_term in article.findall(".//MeshHeading/DescriptorName")
            ]

            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "mesh_terms": ", ".join(mesh_terms) if mesh_terms else "N/A"
            }
    print(f"Failed to fetch data for PMID {pmid} (Status code: {response.status_code})")
    return {"title": "N/A", "abstract": "N/A", "mesh_terms": "N/A"}

def save_paper_data(pmid: str, data: Dict[str, str], output_dir: Path) -> None:
    """Save the title, abstract, and MeSH terms of a paper to a file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{pmid}.json"
    # Save the data in JSON format
    output_file.write_text(json.dumps(data, indent=2))

    # with open(output_file, "w") as f:
    #     f.write(f"PMID: {pmid}\n")
    #     f.write(f"Title: {data['title']}\n")
    #     f.write(f"Abstract: {data['abstract']}\n")
    #     f.write(f"MeSH Terms: {data['mesh_terms']}\n")
    # print(f"Saved data for PMID {pmid} to {output_file}")

def process_pmid(pmid: str) -> None:
    """Fetch and save data for a single PMID if the file does not already exist."""
    output_file = OUTPUT_DIR / f"{pmid}.txt"
    # Return directly if the file already exists and is valid
    if output_file.exists():
        text = output_file.read_text()
        title = text.split("Title: ")[1].split("\n")[0]
        if "N/A" not in title:
            # print(f"File for PMID {pmid} already exists and is valid. Skipping download.")
            return
    # Fetch and save the data for files that do not exist or are invalid
    paper_data = fetch_pubmed_data(pmid)
    save_paper_data(pmid, paper_data, OUTPUT_DIR)

def main():
    pmids = read_tsv_files(LABELS_DIR)
    pmids = list(set(pmids))  # Remove duplicates
    print(f"Found {len(pmids)} PMIDs to process.")
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_pmid, pmid): pmid for pmid in pmids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading PubMed data"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing PMID {futures[future]}: {e}")

if __name__ == "__main__":
    main()