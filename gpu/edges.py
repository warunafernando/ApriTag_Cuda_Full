"""
GPU Canny edge detection implementation using CuPy/CUDA.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except Exception as exc:  # pragma: no cover
    cp = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None


def gpu_canny_edges(
    gray_gpu: cp.ndarray,
    low_thresh: float = 35.0,
    high_thresh: float = 110.0,
    aperture_size: int = 3,
    use_l2_gradient: bool = True,
) -> cp.ndarray:
    """
    GPU Canny edge detector using CuPy/CUDA.

    Args:
        gray_gpu: 2D uint8 or float32 CuPy array on device
        low_thresh: Lower threshold for hysteresis
        high_thresh: Upper threshold for hysteresis
        aperture_size: Sobel kernel size (3, 5, or 7)
        use_l2_gradient: If True, use L2 norm for gradient magnitude

    Returns:
        2D uint8 CuPy array (0 or 255) on device
    """
    if cp is None:
        raise RuntimeError(f"CuPy not available for GPU Canny: {_gpu_import_error}")

    if gray_gpu.ndim != 2:
        raise ValueError("gpu_canny_edges expects 2D grayscale image")

    # Convert to float32 for computation
    gray_f32 = gray_gpu.astype(cp.float32)

    # Stage 1: Gaussian blur (separable)
    blurred = _gaussian_blur(gray_f32, aperture_size)

    # Stage 2: Gradient computation (Sobel)
    gx, gy = _sobel_gradients(blurred, aperture_size)
    if use_l2_gradient:
        magnitude = cp.sqrt(gx * gx + gy * gy)
    else:
        magnitude = cp.abs(gx) + cp.abs(gy)

    # Stage 3: Non-maximum suppression
    nms = _non_maximum_suppression(magnitude, gx, gy)

    # Stage 4: Double-threshold and hysteresis
    edges = _double_threshold_hysteresis(nms, low_thresh, high_thresh)

    # Convert to uint8 (0 or 255)
    return edges.astype(cp.uint8)


def _gaussian_blur(img: cp.ndarray, kernel_size: int) -> cp.ndarray:
    """
    Gaussian blur using separable 1D filters (vectorized).
    For aperture_size=3, use 3x3 Gaussian kernel.
    """
    # Simple Gaussian kernel for small sizes
    if kernel_size == 3:
        # 3x3 Gaussian: [1, 2, 1] / 4
        kernel_1d = cp.array([1.0, 2.0, 1.0], dtype=cp.float32) / 4.0
    elif kernel_size == 5:
        # 5x5 Gaussian: [1, 4, 6, 4, 1] / 16
        kernel_1d = cp.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=cp.float32) / 16.0
    else:
        # Default: simple 3x3
        kernel_1d = cp.array([1.0, 2.0, 1.0], dtype=cp.float32) / 4.0

    # Use scipy.ndimage equivalent if available, otherwise use simple approach
    # For now, use a simplified vectorized approach
    h, w = img.shape
    pad = len(kernel_1d) // 2
    img_padded = cp.pad(img, ((0, 0), (pad, pad)), mode="edge")
    
    # Horizontal pass: fully vectorized (no row loops)
    k = len(kernel_1d)
    blurred_h = cp.zeros_like(img, dtype=cp.float32)
    # Vectorized: for each kernel position, add weighted slice
    for i in range(k):
        blurred_h += img_padded[:, i:i+w] * kernel_1d[i]

    # Vertical pass: fully vectorized (no column loops)
    blurred_h_padded = cp.pad(blurred_h, ((pad, pad), (0, 0)), mode="edge")
    blurred = cp.zeros_like(img, dtype=cp.float32)
    # Vectorized: for each kernel position, add weighted slice
    for i in range(k):
        blurred += blurred_h_padded[i:i+h, :] * kernel_1d[i]

    return blurred


def _sobel_gradients(img: cp.ndarray, aperture_size: int) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Compute Sobel gradients Gx and Gy (vectorized).
    """
    if aperture_size == 3:
        # 3x3 Sobel kernels
        sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
        sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)
    else:
        # Default to 3x3
        sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
        sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)

    h, w = img.shape
    pad = 1
    img_padded = cp.pad(img, ((pad, pad), (pad, pad)), mode="edge")

    # Vectorized convolution using array slicing
    gx = cp.zeros_like(img, dtype=cp.float32)
    gy = cp.zeros_like(img, dtype=cp.float32)

    # For 3x3 kernel, compute using vectorized operations
    for i in range(3):
        for j in range(3):
            gx += img_padded[i:i+h, j:j+w] * sobel_x[i, j]
            gy += img_padded[i:i+h, j:j+w] * sobel_y[i, j]

    return gx, gy


def _non_maximum_suppression(magnitude: cp.ndarray, gx: cp.ndarray, gy: cp.ndarray) -> cp.ndarray:
    """
    Vectorized non-maximum suppression: keep only local maxima along gradient direction.
    No Python loops - fully vectorized using CuPy operations.
    """
    # Compute gradient direction (quantized to 4 bins: 0, 45, 90, 135 degrees)
    # Avoid division by zero
    eps = 1e-8
    angle = cp.arctan2(gy, gx + eps) * 180.0 / cp.pi
    angle = (angle + 180.0) % 180.0  # Normalize to [0, 180)

    # Quantize to 4 directions
    # 0°: horizontal, 45°: diagonal, 90°: vertical, 135°: anti-diagonal
    dir_quant = cp.zeros_like(angle, dtype=cp.int32)
    dir_quant[(angle >= 0) & (angle < 22.5)] = 0
    dir_quant[(angle >= 22.5) & (angle < 67.5)] = 1  # 45°
    dir_quant[(angle >= 67.5) & (angle < 112.5)] = 2  # 90°
    dir_quant[(angle >= 112.5) & (angle < 157.5)] = 3  # 135°
    dir_quant[(angle >= 157.5)] = 0

    mag = magnitude

    # Create shifted versions for each direction
    # 0° (horizontal): compare left/right
    mag_left = cp.pad(mag, ((0, 0), (1, 0)), mode="constant", constant_values=0)[:, :-1]
    mag_right = cp.pad(mag, ((0, 0), (0, 1)), mode="constant", constant_values=0)[:, 1:]

    # 90° (vertical): compare up/down
    mag_up = cp.pad(mag, ((1, 0), (0, 0)), mode="constant", constant_values=0)[:-1, :]
    mag_down = cp.pad(mag, ((0, 1), (0, 0)), mode="constant", constant_values=0)[1:, :]

    # 45° (diagonal ↘): compare up-right / down-left
    mag_ur = cp.pad(mag, ((1, 0), (0, 1)), mode="constant", constant_values=0)[:-1, 1:]  # up-right
    mag_dl = cp.pad(mag, ((0, 1), (1, 0)), mode="constant", constant_values=0)[1:, :-1]  # down-left

    # 135° (diagonal ↙): compare up-left / down-right
    mag_ul = cp.pad(mag, ((1, 0), (1, 0)), mode="constant", constant_values=0)[:-1, :-1]  # up-left
    mag_dr = cp.pad(mag, ((0, 1), (0, 1)), mode="constant", constant_values=0)[1:, 1:]  # down-right

    # Build keep masks for each direction
    keep_0 = (dir_quant == 0) & (mag >= mag_left) & (mag >= mag_right)
    keep_90 = (dir_quant == 2) & (mag >= mag_up) & (mag >= mag_down)
    keep_45 = (dir_quant == 1) & (mag >= mag_ur) & (mag >= mag_dl)
    keep_135 = (dir_quant == 3) & (mag >= mag_ul) & (mag >= mag_dr)

    # Combine all keep masks
    keep = keep_0 | keep_45 | keep_90 | keep_135

    # Apply suppression: keep magnitude where mask is True, else 0
    nms = cp.where(keep, mag, cp.zeros_like(mag))

    return nms


def _double_threshold_hysteresis(nms: cp.ndarray, low_thresh: float, high_thresh: float) -> cp.ndarray:
    """
    Vectorized double-threshold and hysteresis using binary dilation.
    No Python loops - fully vectorized using CuPy/cupyx operations.
    """
    # Classify pixels
    strong = nms >= high_thresh
    weak = (nms >= low_thresh) & (nms < high_thresh)

    # Hysteresis: promote weak edges connected to strong edges (8-connected)
    # Use binary dilation to approximate hysteresis
    try:
        from cupyx.scipy.ndimage import binary_dilation
    except ImportError:
        # Fallback: use simple approach if cupyx not available
        # This is less accurate but still vectorized
        edges = strong.astype(cp.float32) * 255.0
        return edges

    # 8-connected neighborhood structure
    structure = cp.ones((3, 3), dtype=bool)

    # Dilate strong edges to find weak edges that are 8-connected
    strong_dilated = binary_dilation(strong, structure=structure)

    # Promote weak edges that are connected to strong edges
    promoted_weak = weak & strong_dilated

    # Final edges: strong + promoted weak
    final_edges = strong | promoted_weak

    # Convert to uint8 (0 or 255)
    edges_u8 = final_edges.astype(cp.uint8) * 255

    return edges_u8.astype(cp.float32)  # Return as float32 for consistency

