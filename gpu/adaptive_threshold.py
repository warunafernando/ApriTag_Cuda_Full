"""
GPU adaptive thresholding implementation using CuPy/CUDA.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cupy as cp
except Exception as exc:  # pragma: no cover
    cp = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None


def gpu_adaptive_threshold(
    gray_gpu: cp.ndarray,
    block_size: int = 11,
    C: float = 2.0,
    method: str = "mean",
) -> cp.ndarray:
    """
    GPU adaptive threshold using CuPy/CUDA.

    Args:
        gray_gpu: 2D uint8 or float32 CuPy array on device
        block_size: odd window size (e.g. 11, 15)
        C: constant subtracted from local mean
        method: 'mean' for box filter mean (can extend to 'gaussian' later)

    Returns:
        2D uint8 binary image (0 or 255) on device
    """
    if cp is None:
        raise RuntimeError(f"CuPy not available for GPU adaptive threshold: {_gpu_import_error}")

    if gray_gpu.ndim != 2:
        raise ValueError("gpu_adaptive_threshold expects 2D grayscale image")

    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1

    # Convert to float32 for computation
    gray_f32 = gray_gpu.astype(cp.float32)

    if method == "mean":
        # Box filter using uniform kernel for local mean
        # Create uniform kernel
        kernel_size = block_size
        kernel = cp.ones((kernel_size, kernel_size), dtype=cp.float32) / (kernel_size * kernel_size)

        # Convolve to get local mean
        # Use scipy.signal.convolve2d equivalent in CuPy
        # For now, use a simple approach: pad and use element-wise operations
        # More efficient: use CuPy's convolution or custom CUDA kernel
        mean_gpu = _box_filter_cupy(gray_f32, kernel_size)
    elif method == "gaussian":
        # For future: implement Gaussian filter
        raise NotImplementedError("Gaussian method not yet implemented for GPU")
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply threshold: binary = (gray >= (mean - C)) ? 255 : 0
    # OpenCV uses >= (inclusive), not >
    threshold = mean_gpu - C
    binary = (gray_f32 >= threshold).astype(cp.uint8) * 255

    return binary


def gpu_box_filter_mean(gray_gpu: cp.ndarray, k: int) -> cp.ndarray:
    """
    Compute local mean using a separable box filter on GPU.
    Pure GPU implementation using CuPy vectorized operations.

    Args:
        gray_gpu: 2D float32 CuPy array
        k: odd window size (e.g. 11)

    Returns:
        2D float32 CuPy array of local means
    """
    h, w = gray_gpu.shape
    pad = k // 2

    # Pad image with edge replication (BORDER_REPLICATE to match OpenCV)
    img_padded = cp.pad(gray_gpu, ((pad, pad), (pad, pad)), mode="edge")
    hp, wp = img_padded.shape

    # Horizontal pass: vectorized row-wise moving averages
    # Compute cumulative sum along rows
    row_cumsum = cp.cumsum(img_padded, axis=1, dtype=cp.float32)
    # Pad with zeros for indexing
    row_cumsum_padded = cp.pad(row_cumsum, ((0, 0), (1, 0)), mode="constant", constant_values=0)
    # Moving sum: sum[i:i+k] = cumsum[i+k] - cumsum[i]
    # For each output position, we need cumsum[i+k] - cumsum[i]
    row_sums = row_cumsum_padded[:, k:] - row_cumsum_padded[:, :-k]
    # Extract valid region and normalize
    mean_h = row_sums[:, :w] / k

    # Vertical pass: vectorized column-wise moving averages
    # Compute cumulative sum along columns
    col_cumsum = cp.cumsum(mean_h, axis=0, dtype=cp.float32)
    # Pad with zeros
    col_cumsum_padded = cp.pad(col_cumsum, ((1, 0), (0, 0)), mode="constant", constant_values=0)
    # Moving sum
    col_sums = col_cumsum_padded[k:, :] - col_cumsum_padded[:-k, :]
    # Extract valid region and normalize
    mean = col_sums[:h, :] / k

    return mean


def _box_filter_cupy(img: cp.ndarray, kernel_size: int) -> cp.ndarray:
    """
    Pure GPU box filter using separable filters.
    Replaces the CPU cv2.boxFilter call.
    """
    return gpu_box_filter_mean(img, kernel_size)

