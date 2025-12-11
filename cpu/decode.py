from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DecodeResultCPU:
    id: Optional[int]
    hamming: Optional[int]
    bits: Optional[list[int]]
    sample_grid: Optional[np.ndarray]


def decode_apriltag(
    gray: np.ndarray,
    corners_tl_tr_br_bl: np.ndarray,
    cfg: Dict[str, Any],
) -> DecodeResultCPU:
    """
    CPU decode pipeline: warp, sample, threshold, and codebook lookup.
    """
    if gray.ndim != 2:
        raise ValueError("decode_apriltag expects a single-channel grayscale image")

    sampling_cfg = cfg.get("sampling", {})
    warp_size = int(sampling_cfg.get("warp_size", 96))
    inner_cells_cfg = int(sampling_cfg.get("cells", 8))
    border_cells = int(sampling_cfg.get("border_cells", 1))

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    inner_cells_dict = int(dictionary.markerSize)  # Expected to be 6 for tag36h11
    # Use the dictionary's inner size for decoding; the config drives sampling density.
    inner_cells = inner_cells_dict

    sample_grid, grid_for_decode = _warp_and_sample(
        gray, corners_tl_tr_br_bl, warp_size, inner_cells_cfg, border_cells
    )
    bits_matrix, threshold = _threshold_bits(grid_for_decode)

    best_id, best_hamming = _match_codebook_bytes(bits_matrix, dictionary, cfg)

    bits_flat = bits_matrix.astype(int).flatten().tolist()
    return DecodeResultCPU(
        id=best_id,
        hamming=best_hamming,
        bits=bits_flat,
        sample_grid=sample_grid,
    )


def _warp_and_sample(
    gray: np.ndarray,
    corners: np.ndarray,
    warp_size: int,
    inner_cells_cfg: int,
    border_cells: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp the tag to a square and sample per-cell intensities.

    Returns:
        full_sample_grid: (grid_size, grid_size) including border cells.
        inner_for_decode: cropped to the dictionary inner size region.
    """
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

    grid_size = inner_cells_cfg + 2 * border_cells
    cell_size = warp_size / grid_size

    sample_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r in range(grid_size):
        for c in range(grid_size):
            y = (r + 0.5) * cell_size
            x = (c + 0.5) * cell_size
            sample_grid[r, c] = _bilinear_sample(warped, x, y)

    # For decode, always use dictionary inner size (6) centered with the given border.
    inner_cells_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11).markerSize
    start = border_cells
    end = start + inner_cells_dict
    if sample_grid.shape[0] < end or sample_grid.shape[1] < end:
        raise ValueError(
            f"Sampling grid too small for inner decode region: grid {sample_grid.shape}, "
            f"requires {(end, end)}"
        )
    inner_for_decode = sample_grid[start:end, start:end]
    return sample_grid, inner_for_decode


def _bilinear_sample(img: np.ndarray, x: float, y: float) -> float:
    h, w = img.shape[:2]
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return 0.0
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1
    dx, dy = x - x0, y - y0

    val = (
        (1 - dx) * (1 - dy) * img[y0, x0]
        + dx * (1 - dy) * img[y0, x1]
        + (1 - dx) * dy * img[y1, x0]
        + dx * dy * img[y1, x1]
    )
    return float(val)


def _threshold_bits(grid_inner: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Threshold an inner grid to bits using mean intensity.
    """
    threshold = float(grid_inner.mean())
    bits = (grid_inner > threshold).astype(np.uint8)
    return bits, threshold


def _match_codebook_bytes(
    bits_matrix: np.ndarray, dictionary, cfg: Dict[str, Any]
) -> Tuple[Optional[int], Optional[int]]:
    """
    Match observed bits to the AprilTag codebook using byte-list comparison.
    """
    max_hamming = int(cfg.get("decode", {}).get("max_hamming", 4))
    bits_uint8 = bits_matrix.astype(np.uint8)

    # Convert observed bits to the same byte-list representation as the dictionary.
    obs_bytes = cv2.aruco.Dictionary_getByteListFromBits(bits_uint8)[0].astype(np.uint8)
    dict_bytes = dictionary.bytesList.astype(np.uint8)

    xor = np.bitwise_xor(dict_bytes, obs_bytes)
    bit_counts = np.unpackbits(xor, axis=-1).sum(axis=(1, 2))

    best_idx = int(np.argmin(bit_counts))
    best_hamming = int(bit_counts[best_idx])

    if best_hamming <= max_hamming:
        return best_idx, best_hamming
    return None, None

