"""
CPU reference implementation for quad candidate extraction from edges.
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from common.geometry import canonicalize_corners


def cpu_quad_candidates_from_edges(edges_cpu: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    CPU reference quad candidate generator using OpenCV contours.

    Args:
        edges_cpu: numpy uint8 2D array (0 or 255) - binary edge image
        cfg: config dict with 'quads' block containing thresholds

    Returns:
        quads_cpu: np.ndarray, shape (M, 4, 2), float32
            Each quad is 4 (x,y) corner points in image pixel coordinates.
    """
    if edges_cpu.ndim != 2:
        raise ValueError("cpu_quad_candidates_from_edges expects 2D edge image")

    quads_cfg = cfg.get("quads", {})
    min_area = float(quads_cfg.get("min_area", 400.0))
    max_area = float(quads_cfg.get("max_area", 20000.0))
    min_aspect = float(quads_cfg.get("min_aspect", 0.5))
    max_aspect = float(quads_cfg.get("max_aspect", 2.0))
    max_rect_error_deg = float(quads_cfg.get("max_rect_error_deg", 25.0))
    max_quads = int(quads_cfg.get("max_quads_per_frame", 10))

    # Find contours
    contours, _ = cv2.findContours(edges_cpu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    quads = []

    for contour in contours:
        # Approximate polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Must have 4 vertices
        if len(approx) != 4:
            continue

        # Convert to float32 array
        quad = approx.reshape(4, 2).astype(np.float32)

        # Filter by area
        area = cv2.contourArea(quad)
        if area < min_area or area > max_area:
            continue

        # Filter by aspect ratio
        rect = cv2.minAreaRect(quad)
        width, height = rect[1]
        if width == 0 or height == 0:
            continue
        aspect = max(width, height) / min(width, height)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        # Filter by rectangularity (corner angles)
        # Compute angles at each corner
        angles_ok = True
        for i in range(4):
            p0 = quad[i]
            p1 = quad[(i + 1) % 4]
            p2 = quad[(i + 2) % 4]

            v1 = p1 - p0
            v2 = p2 - p1
            angle = np.arccos(
                np.clip(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8),
                    -1.0,
                    1.0,
                )
            )
            angle_deg = np.degrees(angle)
            # Check if angle is close to 90 degrees
            if abs(angle_deg - 90.0) > max_rect_error_deg:
                angles_ok = False
                break

        if not angles_ok:
            continue

        # Canonicalize corner order
        quad_canon = canonicalize_corners(quad)
        quads.append(quad_canon)

        if len(quads) >= max_quads:
            break

    if len(quads) == 0:
        return np.empty((0, 4, 2), dtype=np.float32)

    return np.array(quads, dtype=np.float32)

