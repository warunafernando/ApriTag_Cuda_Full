"""
CPU reference implementation for Canny edge detection.
"""

from __future__ import annotations

import cv2
import numpy as np


def cpu_canny(
    gray: np.ndarray,
    threshold1: float = 50.0,
    threshold2: float = 150.0,
    aperture_size: int = 3,
    L2gradient: bool = False,
) -> np.ndarray:
    """
    CPU reference Canny edge detection using OpenCV.

    Args:
        gray: 2D uint8 image (0-255)
        threshold1: First threshold for hysteresis procedure (lower)
        threshold2: Second threshold for hysteresis procedure (upper)
        aperture_size: Aperture size for Sobel operator (3, 5, or 7)
        L2gradient: If True, use L2 norm for gradient magnitude, else L1

    Returns:
        2D uint8 binary image (0 or 255) - edges are 255, non-edges are 0
    """
    if gray.ndim != 2:
        raise ValueError("cpu_canny expects 2D grayscale image")

    edges = cv2.Canny(
        gray,
        threshold1=int(threshold1),
        threshold2=int(threshold2),
        apertureSize=aperture_size,
        L2gradient=L2gradient,
    )

    return edges

