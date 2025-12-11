"""
CPU reference implementation for Canny edge detection.
"""

from __future__ import annotations

import cv2
import numpy as np


def cpu_canny_edges(
    gray: np.ndarray,
    low_thresh: int = 35,
    high_thresh: int = 110,
    aperture_size: int = 3,
    use_l2_gradient: bool = True,
) -> np.ndarray:
    """
    CPU reference Canny edge detector using OpenCV.

    Args:
        gray: 2D uint8 image (0-255)
        low_thresh: Lower threshold for hysteresis procedure
        high_thresh: Upper threshold for hysteresis procedure
        aperture_size: Aperture size for Sobel operator (3, 5, or 7)
        use_l2_gradient: If True, use L2 norm for gradient magnitude, else L1

    Returns:
        2D uint8 edge image (0 or 255) - edges are 255, non-edges are 0
    """
    if gray.ndim != 2:
        raise ValueError("cpu_canny_edges expects 2D grayscale image")

    edges = cv2.Canny(
        gray,
        threshold1=low_thresh,
        threshold2=high_thresh,
        apertureSize=aperture_size,
        L2gradient=use_l2_gradient,
    )

    return edges

