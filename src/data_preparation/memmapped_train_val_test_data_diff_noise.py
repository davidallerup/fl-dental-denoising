from src.paths import DATA_ROOT
import numpy as np
import torch
import os
import random

from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict


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


def create_memmapped_data(image_dir: str,
                          clean_memmap_file: str,
                          noisy_memmap_file: str,
                          client_id: int,
                          max_patch_overlap: int = 20,
                          patch_size: int = 256) -> None:

    # Get a list of paths for each of the images in a given split.
    image_list = list_images(Path(image_dir))

    # Since images are not necessarily the same dimensions, we need to calculate the number of occurrences.
    dim_count, dtype_list = img_stats(image_list=image_list)

    if len(dtype_list) > 1:
        raise ValueError(f"Multiple datatypes found in images: {set(dtype_list)}. All images must have the same datatype.")

    # Since there can be multiple image dimensions, we need to calculate the number of patches for each of the dims
    num_patches = 0
    for dim, cnt in dim_count.items():
        num_patches += len(calculate_patch_positions(dim, patch_size=patch_size, max_overlap=max_patch_overlap)) * cnt

    # Initialize the memory maps
    clean_memmap_file = clean_memmap_file + f"_len{num_patches}_size{patch_size}x{patch_size}.npy"
    noisy_memmap_file = noisy_memmap_file + f"_len{num_patches}_size{patch_size}x{patch_size}.npy"
    memmap_shape = (num_patches, patch_size, patch_size)
    clean_memmap = np.memmap(filename=clean_memmap_file,
                             dtype=np.float32,
                             mode='w+',
                             shape=memmap_shape)

    noisy_memmap = np.memmap(filename=noisy_memmap_file,
                             dtype=np.float32,
                             mode='w+',
                             shape=memmap_shape)

    # Iterate through images and patches and add to memory map
    memmap_idx = 0
    for p in image_list:
        img = Image.open(p)
        img_array = np.array(img)

        # Normalize to [0, 1]
        img_dtype = str(img_array.dtype)
        img_shape = img_array.shape

        if img_dtype == 'uint8':
            img_array = img_array.astype(np.float32) / 255.0 # 2^8 - 1
        elif img_dtype == 'uint16':
            img_array = img_array.astype(np.float32) / 65535.0 # 2^16 - 1
        else:
            raise ValueError(f"Unknown image dtype: {img_dtype}")

        # Add another sanity check
        if img_array.min() < 0.0:
            raise ValueError(f"Image {p} contains negative value: {p}")
        if img_array.max() > 1.0:
            raise ValueError(f"Image {p} contains value greater than 1: {p}")

        # Add noise
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

        # Iterate patch positions and add patches to memmaps
        for x, y in patch_positions:

            # Save patches
            clean_memmap[memmap_idx] = img_array[y : y + patch_size, x : x + patch_size]
            noisy_memmap[memmap_idx] = noisy_img_array[y : y + patch_size, x : x + patch_size]

            memmap_idx += 1

    print('Done!')

if __name__ == '__main__':

    # Train splits
    train0_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case1/train/client0'
    train1_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case1/train/client1'
    train2_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case1/train/client2'
    train3_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case1/train/client3'
    train4_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case1/train/client4'

    # Validation splits
    val0_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case1/val/client0'
    val1_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case1/val/client1'
    val2_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case1/val/client2'
    val3_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case1/val/client3'
    val4_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case1/val/client4'

    # train 0
    clean_memmap_file = f'{DATA_ROOT}/memmap_case4/train_clean_client0'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case4/train_noisy_client0'
    create_memmapped_data(image_dir=train0_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file,
                          client_id=0)
    
    # train 1
    clean_memmap_file = f'{DATA_ROOT}/memmap_case4/train_clean_client1'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case4/train_noisy_client1'
    create_memmapped_data(image_dir=train1_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file,
                          client_id=1)

    # train 2
    clean_memmap_file = f'{DATA_ROOT}/memmap_case4/train_clean_client2'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case4/train_noisy_client2'
    create_memmapped_data(image_dir=train2_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file,
                          client_id=2)
    
    # train 3
    clean_memmap_file = f'{DATA_ROOT}/memmap_case4/train_clean_client3'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case4/train_noisy_client3'
    create_memmapped_data(image_dir=train3_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file,
                          client_id=3)
    
    # train 4
    clean_memmap_file = f'{DATA_ROOT}/memmap_case4/train_clean_client4'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case4/train_noisy_client4'
    create_memmapped_data(image_dir=train4_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file,
                          client_id=4)
    
    # val 0
    clean_memmap_file = f'{DATA_ROOT}/memmap_case4/val_clean_client0'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case4/val_noisy_client0'
    create_memmapped_data(image_dir=val0_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file,
                          client_id=0)
    
    # val 1
    clean_memmap_file = f'{DATA_ROOT}/memmap_case4/val_clean_client1'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case4/val_noisy_client1'
    create_memmapped_data(image_dir=val1_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file,
                          client_id=1)
    
    # val 2
    clean_memmap_file = f'{DATA_ROOT}/memmap_case4/val_clean_client2'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case4/val_noisy_client2'
    create_memmapped_data(image_dir=val2_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file,
                          client_id=2)
    
    # val 3
    clean_memmap_file = f'{DATA_ROOT}/memmap_case4/val_clean_client3'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case4/val_noisy_client3'
    create_memmapped_data(image_dir=val3_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file,
                          client_id=3)
    
    # val 4
    clean_memmap_file = f'{DATA_ROOT}/memmap_case4/val_clean_client4'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case4/val_noisy_client4'
    create_memmapped_data(image_dir=val4_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file,
                          client_id=4)

    