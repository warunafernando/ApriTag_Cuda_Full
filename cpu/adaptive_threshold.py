"""
CPU reference implementation for adaptive thresholding.
"""

from __future__ import annotations

import cv2
import numpy as np


def cpu_adaptive_threshold(
    gray: np.ndarray,
    block_size: int = 11,
    C: int = 2,
    method: str = "mean",
) -> np.ndarray:
    """
    CPU reference adaptive threshold using OpenCV.

    Args:
        gray: 2D uint8 image (0-255)
        block_size: odd window size (e.g. 11, 15)
        C: constant subtracted from local mean
        method: 'mean' for ADAPTIVE_THRESH_MEAN_C (can extend to 'gaussian' later)

    Returns:
        2D uint8 binary image (0 or 255)
    """
    if gray.ndim != 2:
        raise ValueError("cpu_adaptive_threshold expects 2D grayscale image")

    if method == "mean":
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    elif method == "gaussian":
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1

    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=adaptive_method,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=C,
    )

    return binary

