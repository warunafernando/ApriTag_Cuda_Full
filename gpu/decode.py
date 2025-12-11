"""
GPU bit extraction and decode using CuPy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2

try:
    import cupy as cp
except Exception as exc:  # pragma: no cover
    cp = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None


_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
_DICT_BYTES_GPU = None


def decode_gpu_bits(sample_grid: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Threshold the inner grid to bits on GPU (mean threshold).
    """
    if cp is None:
        raise RuntimeError(f"CuPy not available for GPU bit extraction: {_gpu_import_error}")

    border_cells = int(cfg.get("sampling", {}).get("border_cells", 1))
    inner_size = int(_DICTIONARY.markerSize)  # tag36h11 -> 6

    sample_cp = cp.asarray(sample_grid, dtype=cp.float32)
    start = border_cells
    end = start + inner_size
    if sample_cp.shape[0] < end or sample_cp.shape[1] < end:
        raise ValueError(
            f"Sampling grid too small for decode: {sample_cp.shape}, needs {(end, end)}"
        )

    inner = sample_cp[start:end, start:end]
    threshold = cp.mean(inner)
    bits = (inner > threshold).astype(cp.uint8)
    return cp.asnumpy(bits)


def decode_gpu_codebook(bits: np.ndarray, cfg: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """
    GPU codebook lookup against tag36h11 dictionary using byte-list comparison.
    """
    if cp is None:
        raise RuntimeError(f"CuPy not available for GPU codebook lookup: {_gpu_import_error}")

    max_hamming = int(cfg.get("decode", {}).get("max_hamming", 4))
    bits_cp = cp.asarray(bits, dtype=cp.uint8)

    # Convert bits to byte-list on CPU (small) then move to GPU.
    bits_np = cp.asnumpy(bits_cp)
    obs_bytes_np = cv2.aruco.Dictionary_getByteListFromBits(bits_np)[0].astype(np.uint8)
    obs_bytes = cp.asarray(obs_bytes_np, dtype=cp.uint8)

    dict_bytes = _dict_bytes_gpu()

    xor = cp.bitwise_xor(dict_bytes, obs_bytes)
    xor_flat = xor.reshape(xor.shape[0], -1)
    unpacked = cp.unpackbits(xor_flat)
    unpacked = unpacked.reshape(xor.shape[0], -1)
    bit_counts = unpacked.sum(axis=1)

    best_idx = int(cp.argmin(bit_counts).get())
    best_hamming = int(bit_counts[best_idx].get())

    if best_hamming <= max_hamming:
        return best_idx, best_hamming
    return None, None


def _dict_bytes_gpu():
    global _DICT_BYTES_GPU
    if _DICT_BYTES_GPU is None:
        _DICT_BYTES_GPU = cp.asarray(_DICTIONARY.bytesList.astype(np.uint8), dtype=cp.uint8)
    return _DICT_BYTES_GPU

