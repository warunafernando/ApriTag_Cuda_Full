"""
CPU reference implementation for corner refinement (subpixel).
"""

from __future__ import annotations

import cv2
import numpy as np


def cpu_refine_corners(
    gray: np.ndarray,
    corners_in: np.ndarray,
    window_size: int = 5,
    max_iters: int = 10,
    epsilon: float = 0.01,
) -> np.ndarray:
    """
    CPU reference corner refinement (subpixel).

    Args:
        gray: (H, W) uint8 grayscale image (CPU, numpy).
        corners_in: (N, 4, 2) float32 array of initial quad corners.
        window_size: radius or half window size (e.g. 5 → 11x11 patch).
        max_iters: max number of iterations.
        epsilon: convergence threshold.

    Returns:
        corners_refined: (N, 4, 2) float32 refined corners.
    """
    if gray.ndim != 2:
        raise ValueError("cpu_refine_corners expects 2D grayscale image")

    if corners_in.ndim != 3 or corners_in.shape[1] != 4 or corners_in.shape[2] != 2:
        raise ValueError(f"corners_in must be (N, 4, 2), got {corners_in.shape}")

    corners_refined = corners_in.copy().astype(np.float32)
    win_size = (2 * window_size + 1, 2 * window_size + 1)
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iters, epsilon)

    # Refine each quad's corners
    for i in range(corners_refined.shape[0]):
        quad_corners = corners_refined[i].reshape(-1, 1, 2)
        cv2.cornerSubPix(gray, quad_corners, winSize=win_size, zeroZone=(-1, -1), criteria=term_crit)
        corners_refined[i] = quad_corners.reshape(4, 2)

    return corners_refined

