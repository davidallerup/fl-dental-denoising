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


def create_memmapped_data(image_dir: str,
                          clean_memmap_file: str,
                          noisy_memmap_file: str,
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
        noisy_img_array = add_noise(image=img_array)

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

    """
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
    clean_memmap_file = f'{DATA_ROOT}/memmap_case1/train_clean_client0'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case1/train_noisy_client0'
    create_memmapped_data(image_dir=train0_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 1
    clean_memmap_file = f'{DATA_ROOT}/memmap_case1/train_clean_client1'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case1/train_noisy_client1'
    create_memmapped_data(image_dir=train1_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)

    # train 2
    clean_memmap_file = f'{DATA_ROOT}/memmap_case1/train_clean_client2'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case1/train_noisy_client2'
    create_memmapped_data(image_dir=train2_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 3
    clean_memmap_file = f'{DATA_ROOT}/memmap_case1/train_clean_client3'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case1/train_noisy_client3'
    create_memmapped_data(image_dir=train3_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 4
    clean_memmap_file = f'{DATA_ROOT}/memmap_case1/train_clean_client4'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case1/train_noisy_client4'
    create_memmapped_data(image_dir=train4_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 0
    clean_memmap_file = f'{DATA_ROOT}/memmap_case1/val_clean_client0'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case1/val_noisy_client0'
    create_memmapped_data(image_dir=val0_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 1
    clean_memmap_file = f'{DATA_ROOT}/memmap_case1/val_clean_client1'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case1/val_noisy_client1'
    create_memmapped_data(image_dir=val1_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 2
    clean_memmap_file = f'{DATA_ROOT}/memmap_case1/val_clean_client2'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case1/val_noisy_client2'
    create_memmapped_data(image_dir=val2_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 3
    clean_memmap_file = f'{DATA_ROOT}/memmap_case1/val_clean_client3'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case1/val_noisy_client3'
    create_memmapped_data(image_dir=val3_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 4
    clean_memmap_file = f'{DATA_ROOT}/memmap_case1/val_clean_client4'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case1/val_noisy_client4'
    create_memmapped_data(image_dir=val4_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    """

    """
    # Train splits
    train0_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case2/train/client0'
    train1_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case2/train/client1'
    train2_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case2/train/client2'
    train3_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case2/train/client3'
    train4_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case2/train/client4'
    train5_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case2/train/client5'

    # Validation splits
    val0_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case2/val/client0'
    val1_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case2/val/client1'
    val2_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case2/val/client2'
    val3_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case2/val/client3'
    val4_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case2/val/client4'
    val5_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case2/val/client5'

    # train 0
    clean_memmap_file = f'{DATA_ROOT}/memmap_case2/train_clean_client0'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case2/train_noisy_client0'
    create_memmapped_data(image_dir=train0_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 1
    clean_memmap_file = f'{DATA_ROOT}/memmap_case2/train_clean_client1'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case2/train_noisy_client1'
    create_memmapped_data(image_dir=train1_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)

    # train 2
    clean_memmap_file = f'{DATA_ROOT}/memmap_case2/train_clean_client2'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case2/train_noisy_client2'
    create_memmapped_data(image_dir=train2_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 3
    clean_memmap_file = f'{DATA_ROOT}/memmap_case2/train_clean_client3'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case2/train_noisy_client3'
    create_memmapped_data(image_dir=train3_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 4
    clean_memmap_file = f'{DATA_ROOT}/memmap_case2/train_clean_client4'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case2/train_noisy_client4'
    create_memmapped_data(image_dir=train4_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 5
    clean_memmap_file = f'{DATA_ROOT}/memmap_case2/train_clean_client5'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case2/train_noisy_client5'
    create_memmapped_data(image_dir=train5_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 0
    clean_memmap_file = f'{DATA_ROOT}/memmap_case2/val_clean_client0'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case2/val_noisy_client0'
    create_memmapped_data(image_dir=val0_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 1
    clean_memmap_file = f'{DATA_ROOT}/memmap_case2/val_clean_client1'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case2/val_noisy_client1'
    create_memmapped_data(image_dir=val1_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 2
    clean_memmap_file = f'{DATA_ROOT}/memmap_case2/val_clean_client2'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case2/val_noisy_client2'
    create_memmapped_data(image_dir=val2_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 3
    clean_memmap_file = f'{DATA_ROOT}/memmap_case2/val_clean_client3'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case2/val_noisy_client3'
    create_memmapped_data(image_dir=val3_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 4
    clean_memmap_file = f'{DATA_ROOT}/memmap_case2/val_clean_client4'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case2/val_noisy_client4'
    create_memmapped_data(image_dir=val4_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 5
    clean_memmap_file = f'{DATA_ROOT}/memmap_case2/val_clean_client5'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case2/val_noisy_client5'
    create_memmapped_data(image_dir=val5_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    """
    """
    # Train splits
    train0_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/train/client0'
    train1_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/train/client1'
    train2_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/train/client2'
    train3_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/train/client3'
    train4_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/train/client4'
    train5_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/train/client5'
    train6_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/train/client6'
    train7_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/train/client7'
    train8_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/train/client8'
    train9_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/train/client9'
    train10_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/train/client10'

    # Validation splits
    val0_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/val/client0'
    val1_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/val/client1'
    val2_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/val/client2'
    val3_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/val/client3'
    val4_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/val/client4'
    val5_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/val/client5'
    val6_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/val/client6'
    val7_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/val/client7'
    val8_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/val/client8'
    val9_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/val/client9'
    val10_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3a/val/client10'

    # train 0
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_clean_client0'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_noisy_client0'
    create_memmapped_data(image_dir=train0_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 1
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_clean_client1'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_noisy_client1'
    create_memmapped_data(image_dir=train1_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)

    # train 2
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_clean_client2'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_noisy_client2'
    create_memmapped_data(image_dir=train2_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 3
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_clean_client3'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_noisy_client3'
    create_memmapped_data(image_dir=train3_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 4
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_clean_client4'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_noisy_client4'
    create_memmapped_data(image_dir=train4_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 5
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_clean_client5'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_noisy_client5'
    create_memmapped_data(image_dir=train5_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 6
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_clean_client6'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_noisy_client6'
    create_memmapped_data(image_dir=train6_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 7
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_clean_client7'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_noisy_client7'
    create_memmapped_data(image_dir=train7_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 8
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_clean_client8'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_noisy_client8'
    create_memmapped_data(image_dir=train8_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 9
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_clean_client9'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_noisy_client9'
    create_memmapped_data(image_dir=train9_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 10
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_clean_client10'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/train_noisy_client10'
    create_memmapped_data(image_dir=train10_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 0
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_clean_client0'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_noisy_client0'
    create_memmapped_data(image_dir=val0_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 1
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_clean_client1'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_noisy_client1'
    create_memmapped_data(image_dir=val1_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 2
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_clean_client2'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_noisy_client2'
    create_memmapped_data(image_dir=val2_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 3
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_clean_client3'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_noisy_client3'
    create_memmapped_data(image_dir=val3_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 4
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_clean_client4'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_noisy_client4'
    create_memmapped_data(image_dir=val4_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 5
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_clean_client5'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_noisy_client5'
    create_memmapped_data(image_dir=val5_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 6
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_clean_client6'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_noisy_client6'
    create_memmapped_data(image_dir=val6_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 7
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_clean_client7'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_noisy_client7'
    create_memmapped_data(image_dir=val7_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 8
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_clean_client8'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_noisy_client8'
    create_memmapped_data(image_dir=val8_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 9
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_clean_client9'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_noisy_client9'
    create_memmapped_data(image_dir=val9_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 10
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_clean_client10'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3a/val_noisy_client10'
    create_memmapped_data(image_dir=val10_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    """

    """
    # Train splits
    train0_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client0'
    train1_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client1'
    train2_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client2'
    train3_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client3'
    train4_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client4'
    train5_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client5'
    train6_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client6'
    train7_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client7'
    train8_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client8'
    train9_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client9'
    train10_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/train/client10'

    # Validation splits
    val0_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client0'
    val1_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client1'
    val2_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client2'
    val3_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client3'
    val4_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client4'
    val5_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client5'
    val6_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client6'
    val7_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client7'
    val8_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client8'
    val9_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client9'
    val10_dir = f'{DATA_ROOT}/dental_train_val_test_splits_case3b/val/client10'

    # train 0
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_clean_client0'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_noisy_client0'
    create_memmapped_data(image_dir=train0_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 1
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_clean_client1'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_noisy_client1'
    create_memmapped_data(image_dir=train1_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)

    # train 2
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_clean_client2'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_noisy_client2'
    create_memmapped_data(image_dir=train2_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 3
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_clean_client3'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_noisy_client3'
    create_memmapped_data(image_dir=train3_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 4
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_clean_client4'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_noisy_client4'
    create_memmapped_data(image_dir=train4_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 5
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_clean_client5'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_noisy_client5'
    create_memmapped_data(image_dir=train5_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 6
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_clean_client6'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_noisy_client6'
    create_memmapped_data(image_dir=train6_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 7
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_clean_client7'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_noisy_client7'
    create_memmapped_data(image_dir=train7_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 8
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_clean_client8'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_noisy_client8'
    create_memmapped_data(image_dir=train8_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 9
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_clean_client9'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_noisy_client9'
    create_memmapped_data(image_dir=train9_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # train 10
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_clean_client10'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/train_noisy_client10'
    create_memmapped_data(image_dir=train10_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 0
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_clean_client0'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_noisy_client0'
    create_memmapped_data(image_dir=val0_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 1
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_clean_client1'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_noisy_client1'
    create_memmapped_data(image_dir=val1_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 2
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_clean_client2'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_noisy_client2'
    create_memmapped_data(image_dir=val2_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 3
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_clean_client3'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_noisy_client3'
    create_memmapped_data(image_dir=val3_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 4
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_clean_client4'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_noisy_client4'
    create_memmapped_data(image_dir=val4_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 5
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_clean_client5'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_noisy_client5'
    create_memmapped_data(image_dir=val5_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 6
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_clean_client6'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_noisy_client6'
    create_memmapped_data(image_dir=val6_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 7
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_clean_client7'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_noisy_client7'
    create_memmapped_data(image_dir=val7_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 8
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_clean_client8'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_noisy_client8'
    create_memmapped_data(image_dir=val8_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 9
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_clean_client9'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_noisy_client9'
    create_memmapped_data(image_dir=val9_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    # val 10
    clean_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_clean_client10'
    noisy_memmap_file = f'{DATA_ROOT}/memmap_case3b/val_noisy_client10'
    create_memmapped_data(image_dir=val10_dir,
                          clean_memmap_file=clean_memmap_file,
                          noisy_memmap_file=noisy_memmap_file)
    
    """
