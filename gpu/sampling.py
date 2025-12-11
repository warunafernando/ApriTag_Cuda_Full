"""
GPU sampling implementation using CuPy.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import cv2

try:
    import cupy as cp
except Exception as exc:  # pragma: no cover - handled at runtime
    cp = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None


def sample_gpu(
    gray: np.ndarray,
    corners_tl_tr_br_bl: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """
    Warp-sample the tag region on GPU using homography and bilinear sampling.
    Returns a sampling grid including border cells (same layout as CPU).
    """
    if cp is None:
        raise RuntimeError(f"CuPy not available for GPU sampling: {_gpu_import_error}")

    sampling_cfg = cfg.get("sampling", {})
    gpu_cfg = cfg.get("gpu", {})
    if gpu_cfg.get("force_cpu_exact_sampling", True):
        return _sample_cpu_exact(gray, corners_tl_tr_br_bl, sampling_cfg)

    warp_size = int(sampling_cfg.get("warp_size", 96))
    inner_cells = int(sampling_cfg.get("cells", 8))
    border_cells = int(sampling_cfg.get("border_cells", 1))
    grid_size = inner_cells + 2 * border_cells
    cell_size = warp_size / grid_size

    corners = np.asarray(corners_tl_tr_br_bl, dtype=np.float32)
    dst = np.array(
        [
            [0, 0],
            [warp_size - 1, 0],
            [warp_size - 1, warp_size - 1],
            [0, warp_size - 1],
        ],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(corners, dst)
    H_inv = np.linalg.inv(H).astype(np.float32)
    H_inv_cp = cp.asarray(H_inv)

    cp_gray = cp.asarray(gray, dtype=cp.float32)
    h, w = gray.shape[:2]

    # Build full warp image (warp_size x warp_size) via inverse homography so
    # sampling matches CPU warpPerspective + bilinear sample.
    coords = cp.arange(warp_size, dtype=cp.float32)
    yy_warp, xx_warp = cp.meshgrid(coords, coords, indexing="ij")
    x_flat = xx_warp.ravel()
    y_flat = yy_warp.ravel()

    denom = H_inv_cp[2, 0] * x_flat + H_inv_cp[2, 1] * y_flat + H_inv_cp[2, 2]
    x_src = (H_inv_cp[0, 0] * x_flat + H_inv_cp[0, 1] * y_flat + H_inv_cp[0, 2]) / denom
    y_src = (H_inv_cp[1, 0] * x_flat + H_inv_cp[1, 1] * y_flat + H_inv_cp[1, 2]) / denom

    x0 = cp.floor(x_src).astype(cp.int32)
    y0 = cp.floor(y_src).astype(cp.int32)
    x0 = cp.clip(x0, 0, w - 2)
    y0 = cp.clip(y0, 0, h - 2)

    x1 = x0 + 1
    y1 = y0 + 1
    dx = x_src - x0
    dy = y_src - y0

    v00 = cp_gray[y0, x0]
    v10 = cp_gray[y0, x1]
    v01 = cp_gray[y1, x0]
    v11 = cp_gray[y1, x1]

    warp_flat = (
        (1 - dx) * (1 - dy) * v00
        + dx * (1 - dy) * v10
        + (1 - dx) * dy * v01
        + dx * dy * v11
    )
    warp_img = warp_flat.reshape(warp_size, warp_size)

    # Now sample grid cell centers from the warped image (same as CPU path).
    rs = cp.arange(grid_size, dtype=cp.float32)
    cs = cp.arange(grid_size, dtype=cp.float32)
    yy, xx = cp.meshgrid(rs, cs, indexing="ij")
    y_cell = (yy + 0.5) * cell_size
    x_cell = (xx + 0.5) * cell_size

    sample_grid = _bilinear_sample_grid(warp_img, x_cell, y_cell)
    return cp.asnumpy(sample_grid)


def _sample_cpu_exact(gray: np.ndarray, corners: np.ndarray, sampling_cfg: Dict[str, Any]) -> np.ndarray:
    """
    Exact CPU-parity sampling using cv2.warpPerspective to match the CPU pipeline.
    """
    warp_size = int(sampling_cfg.get("warp_size", 96))
    inner_cells = int(sampling_cfg.get("cells", 8))
    border_cells = int(sampling_cfg.get("border_cells", 1))
    grid_size = inner_cells + 2 * border_cells
    cell_size = warp_size / grid_size

    dst = np.array(
        [
            [0, 0],
            [warp_size - 1, 0],
            [warp_size - 1, warp_size - 1],
            [0, warp_size - 1],
        ],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(gray, H, (warp_size, warp_size))

    sample_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r in range(grid_size):
        for c in range(grid_size):
            y = (r + 0.5) * cell_size
            x = (c + 0.5) * cell_size
            sample_grid[r, c] = _bilinear_sample_numpy(warped, x, y)
    return sample_grid


def _bilinear_sample_numpy(img: np.ndarray, x: float, y: float) -> float:
    h, w = img.shape[:2]
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return 0.0
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1
    dx, dy = x - x0, y - y0

    return float(
        (1 - dx) * (1 - dy) * img[y0, x0]
        + dx * (1 - dy) * img[y0, x1]
        + (1 - dx) * dy * img[y1, x0]
        + dx * dy * img[y1, x1]
    )


def _bilinear_sample_grid(img: cp.ndarray, x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
    h, w = img.shape[:2]
    x0 = cp.floor(x).astype(cp.int32)
    y0 = cp.floor(y).astype(cp.int32)
    x0 = cp.clip(x0, 0, w - 2)
    y0 = cp.clip(y0, 0, h - 2)

    x1 = x0 + 1
    y1 = y0 + 1
    dx = x - x0
    dy = y - y0

    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]

    return (
        (1 - dx) * (1 - dy) * v00
        + dx * (1 - dy) * v10
        + (1 - dx) * dy * v01
        + dx * dy * v11
    )

