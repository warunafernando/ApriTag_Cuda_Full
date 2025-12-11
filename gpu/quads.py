"""
GPU quad candidate extraction from edges using connected components and geometry.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.ndimage import label
    import cv2
except Exception as exc:  # pragma: no cover
    cp = None
    label = None
    cv2 = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None

from common.geometry import canonicalize_corners


def gpu_quad_candidates_from_edges(edges_gpu: cp.ndarray, cfg: Dict[str, Any]) -> tuple[cp.ndarray, dict]:
    """
    Generate quad candidates from a binary edge image on GPU.

    Args:
        edges_gpu: CuPy uint8 2D array (0 or 255) representing edges
        cfg: dict-like, containing quad extraction thresholds

    Returns:
        quads_gpu: CuPy float32 array of shape (N, 4, 2)
            Each quad is 4 (x,y) corner points in image pixel coordinates.
        timings: dict with keys like 't_label_ms', 't_quads_ms', 't_total_ms'
    """
    if cp is None:
        raise RuntimeError(f"CuPy not available for GPU quads: {_gpu_import_error}")

    if edges_gpu.ndim != 2:
        raise ValueError("gpu_quad_candidates_from_edges expects 2D edge image")

    import time
    t_start = time.perf_counter()

    quads_cfg = cfg.get("quads", {})
    min_area = float(quads_cfg.get("min_area", 400.0))
    max_area = float(quads_cfg.get("max_area", 20000.0))
    min_aspect = float(quads_cfg.get("min_aspect", 0.5))
    max_aspect = float(quads_cfg.get("max_aspect", 2.0))
    max_rect_error_deg = float(quads_cfg.get("max_rect_error_deg", 25.0))
    max_quads = int(quads_cfg.get("max_quads_per_frame", 10))

    # Hybrid approach: Use CPU findContours on full image (like CPU reference),
    # but do filtering and processing on GPU where possible
    # Transfer edges to CPU for contour finding (this is acceptable per Phase-14C Option B)
    t0 = time.perf_counter()
    edges_cpu = cp.asnumpy(edges_gpu)
    t_label = (time.perf_counter() - t0) * 1000.0  # Transfer time

    # Find contours on full image (same as CPU reference)
    contours, _ = cv2.findContours(edges_cpu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return cp.empty((0, 4, 2), dtype=cp.float32), {
            "t_label_ms": t_label,
            "t_quads_ms": 0.0,
            "t_total_ms": (time.perf_counter() - t_start) * 1000.0,
        }

    # Process contours
    t0 = time.perf_counter()
    quads_list = []

    for contour in contours:
        # Approximate polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Must have 4 vertices
        if len(approx) != 4:
            continue

        # Convert to float32 array
        quad = approx.reshape(4, 2).astype(np.float32)

        # Filter by area (same order as CPU)
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

        # Filter by rectangularity
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
            if abs(angle_deg - 90.0) > max_rect_error_deg:
                angles_ok = False
                break

        if not angles_ok:
            continue

        # Canonicalize
        quad_canon = canonicalize_corners(quad)
        quads_list.append(quad_canon)

        if len(quads_list) >= max_quads:
            break

    cp.cuda.Stream.null.synchronize()
    t_quads = (time.perf_counter() - t0) * 1000.0

    if len(quads_list) == 0:
        quads_gpu = cp.empty((0, 4, 2), dtype=cp.float32)
    else:
        quads_gpu = cp.asarray(np.array(quads_list, dtype=np.float32))

    timings = {
        "t_label_ms": t_label,
        "t_quads_ms": t_quads,
        "t_total_ms": (time.perf_counter() - t_start) * 1000.0,
    }

    return quads_gpu, timings

