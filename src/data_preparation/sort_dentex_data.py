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

def preprocess_image(img: Image.Image, client: str) -> Image.Image:
    """Remove black bars from the images

    Args:
        img: PIL Image to process
        client: client identifier (client0, client1, client2)

    Returns:
        Preprocessed PIL Image
    """
    if client == 'client0':
        # Remove 45px from bottom, left, right and 15px from top
        width, height = img.size
        return img.crop((45, 15, width - 45, height - 45))

    elif client == 'client1':
        # Remove 150px from left, 30px from bottom and right
        width, height = img.size
        return img.crop((150, 0, width - 30, height - 30))

    elif client == 'client2':
        # Remove black padded areas and white "R"
        img_array = np.array(img)

        # Mask out black (near 0) and white (near 255) pixels
        mask = (img_array > 5) & (img_array < 250)

        # Find rows and columns containing actual image content
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]

        # Get boundaries of non-black/non-white region
        if len(rows) > 0 and len(cols) > 0:
            top, bottom = rows[0], rows[-1]
            left, right = cols[0], cols[-1]
            return img.crop((left, top, right, bottom))

    return img

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
        _, height = img.size

        # Determine client based on image height
        if height == 1316:
            client = 'client0'
        elif height == 1504:
            client = 'client1'
        elif height == 976:
            client = 'client2'
        else:
            client = 'unknown'

        # Preprocess the image based on client
        if client != 'unknown':
            img = preprocess_image(img, client)

        # Create destination directory
        dest_dir = os.path.join(output_root, client)
        os.makedirs(dest_dir, exist_ok=True)

        # Resize image to 512x512 before saving
        # img_resized = img.resize((512, 512), Image.LANCZOS)
        # We no longer want to resize. Just sort the images into the clients and then patch them

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
        images.extend(list(Path(path).rglob('*.png')))
    process_images(images, output_root)

if __name__ == "__main__":

    # Local windows paths
    """
    img_paths = [
        f'{DATA_ROOT}/raw/dentex/training_data/training_data/unlabelled/xrays',
        f'{DATA_ROOT}/raw/dentex/training_data/training_data/quadrant_enumeration/xrays',
        f'{DATA_ROOT}/raw/dentex/training_data/training_data/quadrant/xrays',
        f'{DATA_ROOT}/raw/dentex/validation_data/validation_data/quadrant_enumeration_disease/xrays',
        f'{DATA_ROOT}/raw/dentex/training_data/training_data/quadrant-enumeration-disease/xrays'
    ]

    output_root = f'{DATA_ROOT}/temp_dentex'
    """

    # Local macOS paths for DENTEX
    img_paths = [
        f'{DATA_ROOT}/raw/dentex/training_data/training_data/unlabelled/xrays',
        f'{DATA_ROOT}/raw/dentex/training_data/training_data/quadrant_enumeration/xrays',
        f'{DATA_ROOT}/raw/dentex/training_data/training_data/quadrant/xrays',
        f'{DATA_ROOT}/raw/dentex/validation_data/validation_data/quadrant_enumeration_disease/xrays',
        f'{DATA_ROOT}/raw/dentex/training_data/training_data/quadrant-enumeration-disease/xrays'
    ]

    output_root = f'{DATA_ROOT}/raw_sorted_dentex'

    create_federated_dataset(
        img_paths=img_paths,
        output_root=output_root
    )
