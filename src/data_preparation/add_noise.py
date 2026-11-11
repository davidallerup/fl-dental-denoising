import numpy as np
import random


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

def _add_poisson_noise(image):
    """Add Poisson noise to an image."""
    # Scale the image to the range [0, 255] for Poisson noise
    noisy_image = np.random.poisson(image * 255.0) / (255.0)
    return np.clip(noisy_image, 0, 1).astype(np.float32)

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

def add_noise(image: np.ndarray) -> np.ndarray:

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
