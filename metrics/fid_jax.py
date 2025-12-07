"""
FID (Frechet Inception Distance) computation using JAX InceptionV3.
Adapted from: https://github.com/matthias-wright/jax-fid
"""
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from typing import Tuple
import functools

def compute_statistics(features):
    """
    Compute mean and covariance statistics from features.

    Args:
        features: numpy array of shape (N, 2048) - InceptionV3 features

    Returns:
        mu: Mean vector of shape (2048,)
        sigma: Covariance matrix of shape (2048, 2048)
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def fid_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute FID score from precomputed statistics.

    Args:
        mu1: Mean of real images features, shape (2048,)
        sigma1: Covariance of real images features, shape (2048, 2048)
        mu2: Mean of generated images features, shape (2048,)
        sigma2: Covariance of generated images features, shape (2048, 2048)
        eps: Small constant for numerical stability

    Returns:
        fid: FID score (lower is better, 0 = identical distributions)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    offset = np.eye(sigma1.shape[0]) * eps
    covmean, _ = scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset), disp=False)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def get_fid_fn():
    """
    Get InceptionV3 function for FID computation.
    Note: This is a placeholder that will import from external JAX InceptionV3 implementation.

    For now, returns None and should be implemented by importing from:
    - jax-diffusion-transformer/utils/fid.py, or
    - shortcut-models/utils/fid.py

    Returns:
        apply_fn: Function that takes images in [-1, 1] and returns 2048-d features
    """
    try:
        # Try to import from sibling directories
        import sys
        import os

        # Add paths for potential InceptionV3 implementations
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Try shortcut-models first
        shortcut_path = os.path.join(os.path.dirname(parent_dir), 'shortcut-models', 'utils')
        if os.path.exists(shortcut_path):
            sys.path.insert(0, shortcut_path)
            from fid import get_fid_network
            return get_fid_network()

        # Try jax-diffusion-transformer
        jax_dit_path = os.path.join(os.path.dirname(parent_dir), 'jax-diffusion-transformer', 'utils')
        if os.path.exists(jax_dit_path):
            sys.path.insert(0, jax_dit_path)
            from fid import get_fid_network
            return get_fid_network()

        raise ImportError("Could not find JAX InceptionV3 implementation")

    except Exception as e:
        print(f"Warning: Could not load InceptionV3: {e}")
        print("FID computation will not be available.")
        return None


def extract_features_batch(inception_fn, images, batch_size=64):
    """
    Extract InceptionV3 features from a batch of images.

    Args:
        inception_fn: InceptionV3 apply function
        images: numpy array of shape (N, H, W, C) in range [-1, 1]
        batch_size: Batch size for processing

    Returns:
        features: numpy array of shape (N, 2048)
    """
    if inception_fn is None:
        raise ValueError("InceptionV3 function not available")

    n_images = images.shape[0]
    features_list = []

    for i in range(0, n_images, batch_size):
        batch = images[i:i+batch_size]

        # Resize to 299x299 if needed (InceptionV3 input size)
        if batch.shape[1] != 299 or batch.shape[2] != 299:
            batch = jax.image.resize(batch, (batch.shape[0], 299, 299, 3), method='bilinear')

        # Extract features
        batch_features = inception_fn(batch)  # Shape: (B, 1, 1, 2048)
        batch_features = batch_features.reshape(batch_features.shape[0], -1)  # (B, 2048)

        features_list.append(np.array(batch_features))

    features = np.concatenate(features_list, axis=0)
    return features


def compute_fid(real_images, fake_images, inception_fn=None, batch_size=64):
    """
    Compute FID between real and fake images.

    Args:
        real_images: numpy array of shape (N, H, W, C) in range [-1, 1]
        fake_images: numpy array of shape (M, H, W, C) in range [-1, 1]
        inception_fn: InceptionV3 apply function (if None, will try to load)
        batch_size: Batch size for feature extraction

    Returns:
        fid: FID score
    """
    if inception_fn is None:
        inception_fn = get_fid_fn()

    print(f"Extracting features from {len(real_images)} real images...")
    real_features = extract_features_batch(inception_fn, real_images, batch_size)

    print(f"Extracting features from {len(fake_images)} fake images...")
    fake_features = extract_features_batch(inception_fn, fake_images, batch_size)

    print("Computing statistics...")
    mu1, sigma1 = compute_statistics(real_features)
    mu2, sigma2 = compute_statistics(fake_features)

    print("Computing FID score...")
    fid = fid_from_stats(mu1, sigma1, mu2, sigma2)

    return fid
