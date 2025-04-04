#!/usr/bin/env python3
import os
import urllib.request
import zipfile
import sys
from pathlib import Path

def download_file(url, target_path):
    """Download a file from URL to target path with progress indicator"""
    print(f"Downloading {url} to {target_path}...")

    def progress_callback(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading: {percent}% complete")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, target_path, progress_callback)
    print("\nDownload complete!")

def main():
    # Create data directory if it doesn't exist
    data_dir = Path("sample_data")
    os.makedirs(data_dir, exist_ok=True)

    # Sample data options
    sample_files = {
        "1": {
            "name": "Stanford Bunny",
            "url": "https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz",
            "filename": "bunny.tar.gz",
            "is_zip": True
        },
        "2": {
            "name": "Stanford Dragon",
            "url": "https://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz",
            "filename": "dragon_recon.tar.gz",
            "is_zip": True
        },
        "3": {
            "name": "Open3D Sample (Rabbit)",
            "url": "https://raw.githubusercontent.com/isl-org/Open3D/master/examples/test_data/Rabbit.ply",
            "filename": "Rabbit.ply",
            "is_zip": False
        }
    }

    # Display options
    print("Available sample point cloud files:")
    for key, info in sample_files.items():
        print(f"{key}. {info['name']}")

    # Get user selection
    selection = input("\nSelect a file to download (1-3): ")

    if selection in sample_files:
        selected = sample_files[selection]
        target_path = data_dir / selected['filename']

        # Download the file
        download_file(selected['url'], target_path)

        # Extract if it's a compressed file
        if selected['is_zip']:
            print(f"Extracting {target_path}...")
            if target_path.suffix == '.zip':
                with zipfile.ZipFile(target_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
            elif target_path.suffix == '.gz':
                import tarfile
                with tarfile.open(target_path, "r:gz") as tar:
                    tar.extractall(path=data_dir)
            print("Extraction complete!")

        print(f"\nSample data downloaded to: {data_dir}")
        print("You can now open this file in the Point Cloud Processing application.")
    else:
        print("Invalid selection!")

if __name__ == "__main__":
    main()
