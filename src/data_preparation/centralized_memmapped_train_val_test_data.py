from src.paths import DATA_ROOT
import numpy as np
import torch
import os

from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict

from src.data_preparation.add_noise import add_noise

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])

def calculate_patch_positions(image_shape, patch_size=256, max_overlap=24):
    step_size = patch_size - max_overlap
    image_height, image_width = image_shape

    # Calculating the number of patches ensuring they don't exceed the image size
    x_positions = list(range(0, image_width - max_overlap - 1, step_size))
    y_positions = list(range(0, image_height - max_overlap - 1, step_size))

    # Ensure the last patch doesn't exceed the image size by adding positions for any remaining area
    if x_positions[-1] + patch_size > image_width:
        x_positions[-1] = image_width - patch_size
    if y_positions[-1] + patch_size > image_height:
        y_positions[-1] = image_height - patch_size

    # Final patch positions
    patch_positions = [(x, y) for y in y_positions for x in x_positions]

    return patch_positions

def img_stats(image_list) -> Tuple[Dict[Tuple[int, int], int], List[str]]:

    dim_count = {}
    dtype_list = []

    for p in image_list:
        img = Image.open(p)
        img_array = np.array(img)

        dim = img_array.shape
        img_dtype = str(img_array.dtype)

        if dim in dim_count:
            dim_count[dim] += 1
        else:
            dim_count[dim] = 1

        if img_dtype not in dtype_list:
            dtype_list.append(img_dtype)

    return dim_count, dtype_list

def create_combined_memmapped_data(client_dirs: Dict[int, str],
                                   clean_memmap_file: str,
                                   noisy_memmap_file: str,
                                   max_patch_overlap: int = 20,
                                   patch_size: int = 256) -> Dict[int, Tuple[int, int]]:
    """
    Create a single memmap containing data from all clients.
    
    Returns:
        Dict mapping client_id to (start_idx, end_idx) in the memmap
    """
    
    # First pass: calculate total patches needed and client boundaries
    client_boundaries = {}
    total_patches = 0
    
    for client_id, image_dir in client_dirs.items():
        print(f"Calculating patches for client {client_id}...")
        
        # Get image list and stats for this client
        image_list = list_images(Path(image_dir))
        dim_count, dtype_list = img_stats(image_list=image_list)
        
        if len(dtype_list) > 1:
            raise ValueError(f"Client {client_id}: Multiple datatypes found in images: {set(dtype_list)}")
        
        # Calculate number of patches for this client
        client_patches = 0
        for dim, cnt in dim_count.items():
            client_patches += len(calculate_patch_positions(dim, patch_size=patch_size, max_overlap=max_patch_overlap)) * cnt
        
        # Store boundaries
        start_idx = total_patches
        end_idx = total_patches + client_patches
        client_boundaries[client_id] = (start_idx, end_idx)
        total_patches += client_patches
        
        print(f"Client {client_id}: {client_patches} patches (indices {start_idx}-{end_idx-1})")
    
    print(f"Total patches across all clients: {total_patches}")
    
    # Initialize combined memory maps
    clean_memmap_file = clean_memmap_file + f"_len{total_patches}_size{patch_size}x{patch_size}.npy"
    noisy_memmap_file = noisy_memmap_file + f"_len{total_patches}_size{patch_size}x{patch_size}.npy"
    memmap_shape = (total_patches, patch_size, patch_size)
    
    clean_memmap = np.memmap(filename=clean_memmap_file,
                             dtype=np.float32,
                             mode='w+',
                             shape=memmap_shape)
    
    noisy_memmap = np.memmap(filename=noisy_memmap_file,
                             dtype=np.float32,
                             mode='w+',
                             shape=memmap_shape)
    
    # Second pass: populate the memmaps
    global_memmap_idx = 0
    
    for client_id, image_dir in client_dirs.items():
        print(f"Processing images for client {client_id}...")
        
        image_list = list_images(Path(image_dir))
        
        for p in image_list:
            img = Image.open(p)
            img_array = np.array(img)
            
            # Normalize to [0, 1]
            img_dtype = str(img_array.dtype)
            img_shape = img_array.shape
            
            if img_dtype == 'uint8':
                img_array = img_array.astype(np.float32) / 255.0
            elif img_dtype == 'uint16':
                img_array = img_array.astype(np.float32) / 65535.0
            else:
                raise ValueError(f"Unknown image dtype: {img_dtype}")
            
            # Sanity checks
            if img_array.min() < 0.0:
                raise ValueError(f"Image {p} contains negative value")
            if img_array.max() > 1.0:
                raise ValueError(f"Image {p} contains value greater than 1")
            
            # Add noise
            noisy_img_array = add_noise(image=img_array)
            
            # Calculate patches
            patch_positions = calculate_patch_positions(image_shape=img_shape,
                                                        patch_size=patch_size,
                                                        max_overlap=max_patch_overlap)
            
            # Add patches to combined memmaps
            for x, y in patch_positions:
                clean_memmap[global_memmap_idx] = img_array[y : y + patch_size, x : x + patch_size]
                noisy_memmap[global_memmap_idx] = noisy_img_array[y : y + patch_size, x : x + patch_size]
                global_memmap_idx += 1
    
    print(f'Combined memmap creation complete! Total patches: {global_memmap_idx}')
    print(f"Clean memmap saved to: {clean_memmap_file}")
    print(f"Noisy memmap saved to: {noisy_memmap_file}")
    
    return client_boundaries

if __name__ == '__main__':
    
    """
    # Define all client directories
    train_dirs = {
        0: f'{DATA_ROOT}/dental_train_val_test_splits_case1/train/client0',
        1: f'{DATA_ROOT}/dental_train_val_test_splits_case1/train/client1',
        2: f'{DATA_ROOT}/dental_train_val_test_splits_case1/train/client2',
        3: f'{DATA_ROOT}/dental_train_val_test_splits_case1/train/client3',
        4: f'{DATA_ROOT}/dental_train_val_test_splits_case1/train/client4'
    }
    
    val_dirs = {
        0: f'{DATA_ROOT}/dental_train_val_test_splits_case1/val/client0',
        1: f'{DATA_ROOT}/dental_train_val_test_splits_case1/val/client1',
        2: f'{DATA_ROOT}/dental_train_val_test_splits_case1/val/client2',
        3: f'{DATA_ROOT}/dental_train_val_test_splits_case1/val/client3',
        4: f'{DATA_ROOT}/dental_train_val_test_splits_case1/val/client4'
    }
    
    # Create combined train memmap
    print("Creating combined train memmap...")
    train_boundaries = create_combined_memmapped_data(
        client_dirs=train_dirs,
        clean_memmap_file=f'{DATA_ROOT}/memmap_centralized_case1/combined_train_clean',
        noisy_memmap_file=f'{DATA_ROOT}/memmap_centralized_case1/combined_train_noisy'
    )
    
    # Create combined validation memmap
    print("\nCreating combined validation memmap...")
    val_boundaries = create_combined_memmapped_data(
        client_dirs=val_dirs,
        clean_memmap_file=f'{DATA_ROOT}/memmap_centralized_case1/combined_val_clean',
        noisy_memmap_file=f'{DATA_ROOT}/memmap_centralized_case1/combined_val_noisy'
    )
    """

    """
    train_dirs = {
        0: f'{DATA_ROOT}/dental_train_val_test_splits_case2/train/client0',
        1: f'{DATA_ROOT}/dental_train_val_test_splits_case2/train/client1',
        2: f'{DATA_ROOT}/dental_train_val_test_splits_case2/train/client2',
        3: f'{DATA_ROOT}/dental_train_val_test_splits_case2/train/client3',
        4: f'{DATA_ROOT}/dental_train_val_test_splits_case2/train/client4',
        5: f'{DATA_ROOT}/dental_train_val_test_splits_case2/train/client5'
    }
    
    val_dirs = {
        0: f'{DATA_ROOT}/dental_train_val_test_splits_case2/val/client0',
        1: f'{DATA_ROOT}/dental_train_val_test_splits_case2/val/client1',
        2: f'{DATA_ROOT}/dental_train_val_test_splits_case2/val/client2',
        3: f'{DATA_ROOT}/dental_train_val_test_splits_case2/val/client3',
        4: f'{DATA_ROOT}/dental_train_val_test_splits_case2/val/client4',
        5: f'{DATA_ROOT}/dental_train_val_test_splits_case2/val/client5'
    }
    
    # Create combined train memmap
    print("Creating combined train memmap...")
    train_boundaries = create_combined_memmapped_data(
        client_dirs=train_dirs,
        clean_memmap_file=f'{DATA_ROOT}/memmap_centralized_case2/combined_train_clean',
        noisy_memmap_file=f'{DATA_ROOT}/memmap_centralized_case2/combined_train_noisy'
    )
    
    # Create combined validation memmap
    print("\nCreating combined validation memmap...")
    val_boundaries = create_combined_memmapped_data(
        client_dirs=val_dirs,
        clean_memmap_file=f'{DATA_ROOT}/memmap_centralized_case2/combined_val_clean',
        noisy_memmap_file=f'{DATA_ROOT}/memmap_centralized_case2/combined_val_noisy'
    )
    """

    train_dirs = {
        0: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client0',
        1: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client1',
        2: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client2',
        3: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client3',
        4: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client4',
        5: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client5',
        6: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client6',
        7: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client7',
        8: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client8',
        9: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client9',
        10: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client10'
    }
    
    val_dirs = {
        0: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client0',
        1: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client1',
        2: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client2',
        3: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client3',
        4: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client4',
        5: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client5',
        6: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client6',
        7: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client7',
        8: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client8',
        9: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client9',
        10: f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client10'
    }
    
    

    # Create combined train memmap
    print("Creating combined train memmap...")
    train_boundaries = create_combined_memmapped_data(
        client_dirs=train_dirs,
        clean_memmap_file=f'{DATA_ROOT}/memmap_centralized_case3b/combined_train_clean',
        noisy_memmap_file=f'{DATA_ROOT}/memmap_centralized_case3b/combined_train_noisy'
    )
    
    # Create combined validation memmap
    print("\nCreating combined validation memmap...")
    val_boundaries = create_combined_memmapped_data(
        client_dirs=val_dirs,
        clean_memmap_file=f'{DATA_ROOT}/memmap_centralized_case3b/combined_val_clean',
        noisy_memmap_file=f'{DATA_ROOT}/memmap_centralized_case3b/combined_val_noisy'
    )
    