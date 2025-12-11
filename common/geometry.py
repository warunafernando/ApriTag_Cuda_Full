from __future__ import annotations

import numpy as np


def canonicalize_corners(corners: np.ndarray) -> np.ndarray:
    """
    Order four points as (TL, TR, BR, BL).

    Args:
        corners: Array-like with shape (4, 2).

    Returns:
        np.ndarray with shape (4, 2) ordered TL, TR, BR, BL.
    """
    pts = np.asarray(corners, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected (4,2) corners, got {pts.shape}")

    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])

    # Map angles to quadrants: TL (-pi, -pi/2), TR (-pi/2, 0), BR (0, pi/2), BL (pi/2, pi)
    order = np.argsort(
        [
            _angle_score(a)
            for a in angles
        ]
    )
    return pts[order]


def _angle_score(angle: float) -> float:
    """
    Convert angle to consistent ordering TL, TR, BR, BL.
    """
    # Normalize angle to [-pi, pi]
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle <= -np.pi:
        angle += 2 * np.pi

    # TL: (-pi, -pi/2], TR: (-pi/2, 0], BR: (0, pi/2], BL: (pi/2, pi]
    if -np.pi < angle <= -np.pi / 2:
        return 0  # TL
    if -np.pi / 2 < angle <= 0:
        return 1  # TR
    if 0 < angle <= np.pi / 2:
        return 2  # BR
    return 3  # BL

