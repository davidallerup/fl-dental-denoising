from src.paths import DATA_ROOT
import numpy as np
import torch
import os
import random

from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict

#from src.data_preparation.add_noise import add_noise

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def rescale_pixel_range(image, min_val=0.1,max_val=0.9):
    x1 = 0
    x2 = 1
    y1 = min_val
    y2 = max_val
    scale = (y2-y1)/(x2-x1)
    image = scale*(image - x1) + y1

    return image

def _add_gaussian_noise(image, mean=0, sigma=0.01):
    """Add Gaussian noise to an image."""
    # sigma = var ** 0.5
    noisy_image = image + np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image.astype(np.float32)

def _add_salt_and_pepper_noise(image, prob):
    """ Add salt and pepper noise to an image. """
    # Create a copy of the image to avoid modifying the original one
    noisy_image = np.copy(image)

    # Salt mode
    num_salt = np.ceil(prob * image.size * 0.5)
    coordx = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coordx[0], coordx[1]] = np.max(noisy_image)

    # Pepper mode
    num_pepper = np.ceil(prob * image.size * 0.5)
    coordy = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coordy[0], coordy[1]] = np.min(noisy_image)

    return np.array(noisy_image).astype(np.float32)

def _add_speckle_noise(image, sigma_v):
    noisy_image = image + image * (np.random.normal(0, sigma_v, image.shape))
    noisy_image = np.clip(noisy_image, 0, 1)
    return np.array(noisy_image).astype(np.float32)

def _add_quantum_noise(image, sigma_scaling):
    
    noisy_image = np.random.normal(image, image*sigma_scaling, image.shape)
    noisy_image = np.clip(noisy_image, 0, 1)

    return noisy_image.astype(np.float32)

def add_noise0(image: np.ndarray) -> np.ndarray:

    rescaled_image = rescale_pixel_range(image, min_val=0.1, max_val=0.9)

    quantum_noise_sigma_scaling = random.uniform(0.075, 0.125)
    noisy_image = _add_quantum_noise(rescaled_image, quantum_noise_sigma_scaling)

    sigma = random.uniform(0.015, 0.025)
    noisy_image = _add_gaussian_noise(noisy_image, mean=0, sigma=sigma)

    #sigma_v = random.uniform(0, 0.005)
    #noisy_image = _add_speckle_noise(noisy_image, sigma_v)

    #prob = random.uniform(0, 0.01)
    #noisy_image = _add_salt_and_pepper_noise(noisy_image, prob)

    return noisy_image


def add_noise1(image: np.ndarray) -> np.ndarray:

    rescaled_image = rescale_pixel_range(image, min_val=0.1, max_val=0.9)

    quantum_noise_sigma_scaling = random.uniform(0.075, 0.125)
    noisy_image = _add_quantum_noise(rescaled_image, quantum_noise_sigma_scaling)

    sigma = random.uniform(0.015, 0.025)
    noisy_image = _add_gaussian_noise(noisy_image, mean=0, sigma=sigma)

    sigma_v = random.uniform(0, 0.01)
    noisy_image = _add_speckle_noise(noisy_image, sigma_v)

    prob = random.uniform(0, 0.02)
    noisy_image = _add_salt_and_pepper_noise(noisy_image, prob)

    return noisy_image

def add_noise2(image: np.ndarray) -> np.ndarray:

    rescaled_image = rescale_pixel_range(image, min_val=0.1, max_val=0.9)

    quantum_noise_sigma_scaling = random.uniform(0.075*2, 0.125*2)
    noisy_image = _add_quantum_noise(rescaled_image, quantum_noise_sigma_scaling)

    sigma = random.uniform(0.015*2, 0.025*2)
    noisy_image = _add_gaussian_noise(noisy_image, mean=0, sigma=sigma)

    #sigma_v = random.uniform(0, 0.01)
    #noisy_image = _add_speckle_noise(noisy_image, sigma_v)

    #prob = random.uniform(0, 0.02)
    #noisy_image = _add_salt_and_pepper_noise(noisy_image, prob)

    return noisy_image

def add_noise3(image: np.ndarray) -> np.ndarray:

    rescaled_image = rescale_pixel_range(image, min_val=0.1, max_val=0.9)

    quantum_noise_sigma_scaling = random.uniform(0.075, 0.125)
    noisy_image = _add_quantum_noise(rescaled_image, quantum_noise_sigma_scaling)

    sigma = random.uniform(0.015*2, 0.025*2)
    noisy_image = _add_gaussian_noise(noisy_image, mean=0, sigma=sigma)

    sigma_v = random.uniform(0, 0.01)
    noisy_image = _add_speckle_noise(noisy_image, sigma_v)

    prob = random.uniform(0, 0.02)
    noisy_image = _add_salt_and_pepper_noise(noisy_image, prob)

    return noisy_image


def add_noise4(image: np.ndarray) -> np.ndarray:

    rescaled_image = rescale_pixel_range(image, min_val=0.1, max_val=0.9)

    quantum_noise_sigma_scaling = random.uniform(0.075*0.5, 0.125*0.5)
    noisy_image = _add_quantum_noise(rescaled_image, quantum_noise_sigma_scaling)

    sigma = random.uniform(0.015*0.5, 0.025*0.5)
    noisy_image = _add_gaussian_noise(noisy_image, mean=0, sigma=sigma)

    #sigma_v = random.uniform(0, 0.01)
    #noisy_image = _add_speckle_noise(noisy_image, sigma_v)

    #prob = random.uniform(0, 0.02)
    #noisy_image = _add_salt_and_pepper_noise(noisy_image, prob)

    return noisy_image


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
            #noisy_img_array = add_noise(image=img_array)
            if client_id == 0:
                noisy_img_array = add_noise0(image=img_array)
            elif client_id == 1:
                noisy_img_array = add_noise1(image=img_array)
            elif client_id == 2:
                noisy_img_array = add_noise2(image=img_array)
            elif client_id == 3:
                noisy_img_array = add_noise3(image=img_array)
            elif client_id == 4:
                noisy_img_array = add_noise4(image=img_array)
            
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
        clean_memmap_file=f'{DATA_ROOT}/memmap_centralized_case4/combined_train_clean',
        noisy_memmap_file=f'{DATA_ROOT}/memmap_centralized_case4/combined_train_noisy'
    )
    
    # Create combined validation memmap
    print("\nCreating combined validation memmap...")
    val_boundaries = create_combined_memmapped_data(
        client_dirs=val_dirs,
        clean_memmap_file=f'{DATA_ROOT}/memmap_centralized_case4/combined_val_clean',
        noisy_memmap_file=f'{DATA_ROOT}/memmap_centralized_case4/combined_val_noisy'
    )