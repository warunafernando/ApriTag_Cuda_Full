"""
Fast GPU edge detector using fused Sobel+threshold kernel.
Optimized for speed, not strict Canny parity.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    from cupy import RawKernel
except Exception as exc:  # pragma: no cover
    cp = None
    RawKernel = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None


# CUDA kernel for fused Sobel + threshold
_SOBEL_THRESHOLD_KERNEL = """
extern "C" __global__
void sobel_threshold(
    const float* gray,
    unsigned char* edges,
    int height,
    int width,
    float low_thresh,
    float high_thresh
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Skip border pixels (1-pixel border)
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        if (x < width && y < height) {
            edges[y * width + x] = 0;
        }
        return;
    }
    
    // Load 3x3 neighborhood (using shared memory would be better, but keep simple for now)
    float g00 = gray[(y-1) * width + (x-1)];
    float g01 = gray[(y-1) * width + x];
    float g02 = gray[(y-1) * width + (x+1)];
    float g10 = gray[y * width + (x-1)];
    float g11 = gray[y * width + x];
    float g12 = gray[y * width + (x+1)];
    float g20 = gray[(y+1) * width + (x-1)];
    float g21 = gray[(y+1) * width + x];
    float g22 = gray[(y+1) * width + (x+1)];
    
    // Sobel gradients
    float gx = -g00 + g02 - 2.0f * g10 + 2.0f * g12 - g20 + g22;
    float gy = -g00 - 2.0f * g01 - g02 + g20 + 2.0f * g21 + g22;
    
    // Gradient magnitude (L2)
    float mag = sqrtf(gx * gx + gy * gy);
    
    // Threshold
    unsigned char edge_val = 0;
    if (mag >= high_thresh) {
        edge_val = 255;
    } else if (mag >= low_thresh) {
        edge_val = 255;  // Simple: both thresholds map to edge
    }
    
    edges[y * width + x] = edge_val;
}
"""


def gpu_fast_edges(
    gray_gpu: cp.ndarray,
    low_thresh: float = 30.0,
    high_thresh: float = 90.0,
) -> cp.ndarray:
    """
    Fast GPU edge detector for quad detection.
    Uses fused Sobel+threshold kernel (no NMS, no hysteresis).

    Args:
        gray_gpu: 2D CuPy array (uint8 or float32) on device
        low_thresh: Lower threshold for edge detection
        high_thresh: Upper threshold for edge detection

    Returns:
        2D uint8 CuPy array (0 or 255) on device
    """
    if cp is None:
        raise RuntimeError(f"CuPy not available for GPU fast edges: {_gpu_import_error}")

    if gray_gpu.ndim != 2:
        raise ValueError("gpu_fast_edges expects 2D grayscale image")

    # Convert to float32 if needed
    gray_f32 = gray_gpu.astype(cp.float32)

    h, w = gray_f32.shape
    edges = cp.zeros((h, w), dtype=cp.uint8)

    # Compile kernel (cache it)
    if not hasattr(gpu_fast_edges, '_kernel'):
        gpu_fast_edges._kernel = RawKernel(_SOBEL_THRESHOLD_KERNEL, 'sobel_threshold')

    kernel = gpu_fast_edges._kernel

    # Launch kernel
    block_size = (16, 16)
    grid_size = (
        (w + block_size[0] - 1) // block_size[0],
        (h + block_size[1] - 1) // block_size[1],
    )

    kernel(
        grid_size,
        block_size,
        (
            gray_f32,
            edges,
            np.int32(h),
            np.int32(w),
            np.float32(low_thresh),
            np.float32(high_thresh),
        ),
    )

    return edges

