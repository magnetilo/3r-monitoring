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

if __name__ == "__main__":
    download_goldhamster_labels()