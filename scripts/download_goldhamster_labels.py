import requests
from pathlib import Path

BASE_URL = 'https://zenodo.org/records/7152295/files'
RANGE_VALUES = range(10)
DEV_FILENAMES = [f"dev{split}.txt" for split in RANGE_VALUES]
TRAIN_FILENAMES = [f"train{split}.txt" for split in RANGE_VALUES]
TEST_FILENAMES = [f"test{split}.txt" for split in RANGE_VALUES]

def download_file(url: str, dest_folder: Path) -> None:
    """Download a file from a URL to a destination folder."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    filename = dest_folder / url.split('/')[-1]
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {url} (Status code: {response.status_code})")

def download_goldhamster_labels() -> None:
    """Download GoldHamster label datasets (dev, train, test splits)."""
    dest_folder = Path('data/goldhamster/labels')
    all_filenames = DEV_FILENAMES + TRAIN_FILENAMES + TEST_FILENAMES
    
    for filename in all_filenames:
        url = f"{BASE_URL}/{filename}"
        download_file(url, dest_folder)

def merge_goldhamster_labels() -> None:
    """Merge all splits into single files for dev, train, and test, ensuring no duplicate PMIDs."""
    dest_folder = Path('data/goldhamster/labels')
    merged_files = {
        "dev_full.txt": DEV_FILENAMES,
        "train_full.txt": TRAIN_FILENAMES,
        "test_full.txt": TEST_FILENAMES,
    }

    for merged_filename, split_filenames in merged_files.items():
        merged_file_path = dest_folder / merged_filename
        seen_pmids = set()  # Track unique PMIDs
        with open(merged_file_path, 'w') as merged_file:
            for split_filename in split_filenames:
                split_file_path = dest_folder / split_filename
                if split_file_path.exists():
                    with open(split_file_path, 'r') as split_file:
                        for line in split_file:
                            pmid = line.split('\t')[0]
                            if pmid not in seen_pmids:
                                seen_pmids.add(pmid)
                                merged_file.write(line)
        print(f"Merged into: {merged_file_path}")

if __name__ == "__main__":
    #download_goldhamster_labels()
    merge_goldhamster_labels()
    print("All files downloaded and merged successfully.")