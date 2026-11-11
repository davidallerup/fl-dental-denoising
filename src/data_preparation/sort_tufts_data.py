"""Python script to take the raw data from the DENTEX challenge
https://dentex.grand-challenge.org/data/
and sort the images depending on the client that they're taken from.
"""

from src.paths import DATA_ROOT
import os
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional

def get_new_image_name(img_path: Path) -> str:
    """Rename image by appending parent folder before 'xrays', replacing dashes with underscores."""
    parts = img_path.parts
    if 'xrays' in parts:
        xrays_idx = parts.index('xrays')
        if xrays_idx > 0:
            folder_name = parts[xrays_idx - 1].replace('-', '_')
            base_name = img_path.stem
            ext = img_path.suffix
            return f"{base_name}_{folder_name}{ext}"
    return img_path.name

def process_images(img_paths: List[Path], output_root: str) -> None:
    """Process a list of image paths and sort them by client."""
    for img_path in img_paths:
        # The DENTEX images are stored as RGB for some reason, but they're 8bit grayscale, so
        # we just load them as grayscale directly (and save them as this!)
        img = Image.open(img_path).convert('L')

        # Create destination directory
        dest_dir = os.path.join(output_root)
        os.makedirs(dest_dir, exist_ok=True)

        output_path = os.path.join(dest_dir, get_new_image_name(img_path))
        img.save(output_path)
        print(f"Processed image: {get_new_image_name(img_path)})")

def create_federated_dataset(img_paths: List[str],
                             output_root: str) -> None:
    """Create federated dataset from DENTEX data.

    Args:
        img_paths: List of paths containing images
        output_root: Root directory for the organized dataset
    """
    # Pre-process data
    images = []
    for path in img_paths:
        images.extend(list(Path(path).rglob('*.JPG')))
    process_images(images, output_root)

if __name__ == "__main__":

    # Local macOS paths for TUFTS
    img_paths = [
        f'{DATA_ROOT}/raw_tufts_jpg'
    ]

    #output_root = f'{DATA_ROOT}/raw_sorted_dentex'
    output_root = f'{DATA_ROOT}/raw_tufts'

    create_federated_dataset(
        img_paths=img_paths,
        output_root=output_root
    )
