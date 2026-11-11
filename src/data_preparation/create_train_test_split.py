from __future__ import annotations
from src.paths import DATA_ROOT

import shutil
from pathlib import Path
from typing import List, Tuple

try:
    from sklearn.model_selection import train_test_split  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required. Install with: poetry add scikit-learn"
    ) from e

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    # Sort by filename (case-sensitive) so the seeded splits are identical on every operating system.
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=lambda p: p.name)

"""
def split_institution0(images: List[Path], seed: int) -> Tuple[List[Path], List[Path]]:
    #Split institution0 image list into two equal (or near equal) halves.
    #We use train_test_split with test_size=0.5 to preserve determinism with random_state.

    if not images:
        return [], []
    part_a, part_b = train_test_split(images, test_size=0.5, random_state=seed, shuffle=True)
    return part_a, part_b
"""

"""
def adjust_image(image_path: Path, brightness_factor: float = 0.8, contrast_factor: float = 0.8) -> Image.Image:
    # Adjust brightness and contrast of an image.
    try:
        with Image.open(image_path) as img:
            # Convert to RGB to ensure compatibility
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Create a copy to avoid modifying the original
            img_copy = img.copy()

            enhancer = ImageEnhance.Brightness(img_copy)
            img_copy = enhancer.enhance(brightness_factor)

            enhancer = ImageEnhance.Contrast(img_copy)
            img_copy = enhancer.enhance(contrast_factor)

            return img_copy
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None
"""

def copy_images(
    images: List[Path],
    dest: Path,
    dry_run: bool = False,
    modify: bool = False
) -> None:
    """Copy images with progress tracking and error handling."""
    dest.mkdir(parents=True, exist_ok=True)
    total = len(images)

    for idx, src in enumerate(images, 1):
        try:
            target = dest / src.name
            if dry_run:
                print(f"[DRY] Would {'modify and copy' if modify else 'copy'} {src} -> {target}")
            else:
                shutil.copy2(src, target)

                # Print progress every 10%
                if idx % max(1, total // 10) == 0:
                    print(f"Progress: {idx}/{total} ({idx / total * 100:.1f}%)")

        except Exception as e:
            print(f"Error processing {src}: {str(e)}")
            continue


def split_data(images: List[Path], seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split images into train (70%), validation (15%) and test (15%) sets."""
    if not images:
        return [], [], []

    # First split: 70% train, 30% temp
    train, temp = train_test_split(images, test_size=0.3, random_state=seed, shuffle=True)

    # Second split: Split temp into two equal parts (15% each of original)
    val, test = train_test_split(temp, test_size=0.5, random_state=seed, shuffle=True)

    return train, val, test


def train_val_test_split_copy(
    source_root: str,
    output_root: str,
    client_name: str,
    seed: int = 42,
    dry_run: bool = False,
    discard_half: bool = False
) -> None:

    images = list_images(Path(source_root))

    if not images:
        raise RuntimeError(f"No images found in {source_root}")

    # If we only want to use half the images
    if discard_half:
        images, _ = train_test_split(images, test_size=0.5, random_state=seed, shuffle=True)

    # Create splits for each client
    client_train, client_val, client_test = split_data(images, seed)

    # Define output directories
    output_base = Path(output_root)

    # Copy files for each client and split
    print(f"\nClient 0 splits: train={len(client_train)}, val={len(client_val)}, test={len(client_test)}")
    copy_images(client_train, output_base / "train" / client_name , dry_run=dry_run)
    copy_images(client_val, output_base / "val" / client_name, dry_run=dry_run)
    copy_images(client_test, output_base / "test" / client_name, dry_run=dry_run)

    print(f"Done creating train/val/test splits for {source_root}")


if __name__ == "__main__":

    #temp_root = f'{DATA_ROOT}/temp_dentex'

    # Common parameters
    #seed = 42
    #dry_run = False
    #output_root = f'{DATA_ROOT}/dental_train_val_test_splits_case1/'
    #output_root = f'{DATA_ROOT}/dental_train_val_test_splits_case1'

    ############ Create splits for the different clients ############

    ############# DENTEX ############
    # There's over double the amount of data for this client, so we just discard half.
    """
    source_root0 = f'{DATA_ROOT}/raw_sorted_dentex/client0'
    train_val_test_split_copy(
        source_root=source_root0,
        output_root=output_root,
        seed=seed,
        client_name='client0',
        dry_run=dry_run,
        discard_half=True
    )

    source_root1 = f'{DATA_ROOT}/raw_sorted_dentex/client1'
    train_val_test_split_copy(
        source_root=source_root1,
        output_root=output_root,
        seed=seed,
        client_name='client1',
        dry_run=dry_run,
        discard_half=False
    )

    ############# A Dual-Labeled Dataset ############
    source_root2 = f'{DATA_ROOT}/raw_sorted_dual_labeled/dl-client1'
    train_val_test_split_copy(
        source_root=source_root2,
        output_root=output_root,
        seed=seed,
        client_name='client2',
        dry_run=dry_run,
        discard_half=False
    )

    source_root3 = f'{DATA_ROOT}/raw_sorted_dual_labeled/dl-client2'
    train_val_test_split_copy(
        source_root=source_root3,
        output_root=output_root,
        seed=seed,
        client_name='client3',
        dry_run=dry_run,
        discard_half=False
    )


    ############# Tufts ############
    source_root4 = f'{DATA_ROOT}/raw_tufts'
    train_val_test_split_copy(
        source_root=source_root4,
        output_root=output_root,
        seed=seed,
        client_name='client4',
        dry_run=dry_run,
        discard_half=False
    )
    """


    ###################### Train/val/test split for Case Study 2 ######################
    # Common parameters
    seed = 42
    dry_run = False
    output_root = f'{DATA_ROOT}/dental_train_val_test_splits_case2'

    ############ Create splits for the different clients ############

    ############# DENTEX ############
    # There's over double the amount of data for this client, so we just discard half.
    source_root0 = f'{DATA_ROOT}/raw_sorted_dentex/client0'
    train_val_test_split_copy(
        source_root=source_root0,
        output_root=output_root,
        seed=seed,
        client_name='client0',
        dry_run=dry_run,
        discard_half=False
    )

    source_root1 = f'{DATA_ROOT}/raw_sorted_dentex/client1'
    train_val_test_split_copy(
        source_root=source_root1,
        output_root=output_root,
        seed=seed,
        client_name='client1',
        dry_run=dry_run,
        discard_half=False
    )

    # Extra client with little data (Dataset 3), added from Case Study 2
    source_root5 = f'{DATA_ROOT}/raw_sorted_dentex/client2'
    train_val_test_split_copy(
        source_root=source_root5,
        output_root=output_root,
        seed=seed,
        client_name='client5',
        dry_run=dry_run,
        discard_half=False
    )

    ############# A Dual-Labeled Dataset ############
    source_root2 = f'{DATA_ROOT}/raw_sorted_dual_labeled/dl-client1'
    train_val_test_split_copy(
        source_root=source_root2,
        output_root=output_root,
        seed=seed,
        client_name='client2',
        dry_run=dry_run,
        discard_half=False
    )

    source_root3 = f'{DATA_ROOT}/raw_sorted_dual_labeled/dl-client2'
    train_val_test_split_copy(
        source_root=source_root3,
        output_root=output_root,
        seed=seed,
        client_name='client3',
        dry_run=dry_run,
        discard_half=False
    )

    ############# Tufts ############
    source_root4 = f'{DATA_ROOT}/raw_tufts'
    train_val_test_split_copy(
        source_root=source_root4,
        output_root=output_root,
        seed=seed,
        client_name='client4',
        dry_run=dry_run,
        discard_half=False
    )

