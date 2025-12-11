"""
GPU corner refinement (subpixel) using batched processing and RawKernels.
"""

from __future__ import annotations

from typing import Any, Dict

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


# RawKernel for batched corner refinement with improved robustness
_refine_kernel_code = """
extern "C" __global__
void refine_corners_kernel(
    const float* gray, int h, int w,
    const float* Ix, const float* Iy,
    float* x, float* y,
    int window_size, int max_iters, float epsilon,
    float step_max, float det_min, float grad_thresh,
    int n_corners
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_corners) return;

    float x0 = x[idx];
    float y0 = y[idx];

    // Clamp initial position
    x0 = fmaxf(window_size + 1.0f, fminf(w - 2.0f - window_size, x0));
    y0 = fmaxf(window_size + 1.0f, fminf(h - 2.0f - window_size, y0));

    int converged_count = 0;  // Track consecutive converged iterations

    for (int iter = 0; iter < max_iters; iter++) {
        int x_int = (int)x0;
        int y_int = (int)y0;

        int x_min = x_int - window_size;
        int x_max = x_int + window_size + 1;
        int y_min = y_int - window_size;
        int y_max = y_int + window_size + 1;

        if (x_min < 0 || x_max > w || y_min < 0 || y_max > h) break;

        // Compute center offset within patch
        float center_x = x0 - x_min;
        float center_y = y0 - y_min;

        // Bilinear sample intensity at center
        int x_floor = (int)floorf(center_x);
        int y_floor = (int)floorf(center_y);
        int x_ceil = x_floor + 1;
        int y_ceil = y_floor + 1;

        x_floor = max(0, min(x_max - x_min - 1, x_floor));
        y_floor = max(0, min(y_max - y_min - 1, y_floor));
        x_ceil = max(0, min(x_max - x_min - 1, x_ceil));
        y_ceil = max(0, min(y_max - y_min - 1, y_ceil));

        float fx = center_x - x_floor;
        float fy = center_y - y_floor;

        // Sample intensity at center
        int patch_w = x_max - x_min;
        int patch_h = y_max - y_min;
        float I00 = gray[(y_min + y_floor) * w + (x_min + x_floor)];
        float I10 = gray[(y_min + y_floor) * w + (x_min + x_ceil)];
        float I01 = gray[(y_min + y_ceil) * w + (x_min + x_floor)];
        float I11 = gray[(y_min + y_ceil) * w + (x_min + x_ceil)];

        float I_center = I00 * (1.0f - fx) * (1.0f - fy) +
                        I10 * fx * (1.0f - fy) +
                        I01 * (1.0f - fx) * fy +
                        I11 * fx * fy;

        // Build structure tensor and error vector
        float Jxx = 0.0f, Jxy = 0.0f, Jyy = 0.0f;
        float Jx_err = 0.0f, Jy_err = 0.0f;
        float grad_mag_sum = 0.0f;
        int patch_pixels = 0;

        float sigma = window_size / 2.0f;
        float sigma_sq_2 = 2.0f * sigma * sigma;

        for (int py = 0; py < patch_h; py++) {
            for (int px = 0; px < patch_w; px++) {
                int gx_idx = (y_min + py) * w + (x_min + px);
                int gy_idx = (y_min + py) * w + (x_min + px);
                float Ix_val = Ix[gx_idx];
                float Iy_val = Iy[gy_idx];
                float I_val = gray[(y_min + py) * w + (x_min + px)];

                // Weight by distance from center (Gaussian-like)
                float dx_px = px - center_x;
                float dy_py = py - center_y;
                float dist_sq = dx_px * dx_px + dy_py * dy_py;
                float weight = expf(-dist_sq / sigma_sq_2);

                // Accumulate structure tensor
                Jxx += weight * Ix_val * Ix_val;
                Jxy += weight * Ix_val * Iy_val;
                Jyy += weight * Iy_val * Iy_val;

                // Error term: intensity difference
                float err = I_val - I_center;
                Jx_err += weight * Ix_val * err;
                Jy_err += weight * Iy_val * err;

                // Track gradient magnitude for threshold check
                grad_mag_sum += weight * (fabsf(Ix_val) + fabsf(Iy_val));
                patch_pixels++;
            }
        }

        // Check gradient magnitude threshold
        float avg_grad_mag = grad_mag_sum / patch_pixels;
        if (avg_grad_mag < grad_thresh) {
            break;  // Patch too flat, skip update
        }

        // Solve for delta: (J^T * J) * delta = J^T * error
        float det = Jxx * Jyy - Jxy * Jxy;
        
        // Condition number / det guard
        if (fabsf(det) < det_min) {
            break;  // System ill-conditioned, skip update
        }

        float dx = (Jyy * Jx_err - Jxy * Jy_err) / det;
        float dy = (-Jxy * Jx_err + Jxx * Jy_err) / det;

        // Step clamping with configurable step_max
        float delta_mag = sqrtf(dx * dx + dy * dy);
        if (delta_mag > step_max) {
            dx = dx * step_max / delta_mag;
            dy = dy * step_max / delta_mag;
            delta_mag = step_max;
        }

        // Update position
        float x0_new = x0 + dx;
        float y0_new = y0 + dy;

        // Check convergence with early exit
        if (delta_mag < epsilon) {
            converged_count++;
            if (converged_count >= 2) {
                // Converged for 2 consecutive iterations, stop
                x0 = x0_new;
                y0 = y0_new;
                break;
            }
        } else {
            converged_count = 0;  // Reset if not converged
        }

        x0 = x0_new;
        y0 = y0_new;

        // Clamp to valid bounds
        x0 = fmaxf(window_size + 1.0f, fminf(w - 2.0f - window_size, x0));
        y0 = fmaxf(window_size + 1.0f, fminf(h - 2.0f - window_size, y0));
    }

    x[idx] = x0;
    y[idx] = y0;
}
"""

_refine_kernel = None


def _get_refine_kernel():
    """Get or compile the refinement kernel."""
    global _refine_kernel
    if _refine_kernel is None and cp is not None:
        _refine_kernel = RawKernel(_refine_kernel_code, "refine_corners_kernel")
    return _refine_kernel


def gpu_refine_corners(
    gray_gpu: cp.ndarray,
    corners_in_gpu: cp.ndarray,
    window_size: int = 5,
    max_iters: int = 10,
    epsilon: float = 0.01,
    step_max: float = 1.0,
    det_min: float = 1e-4,
    grad_thresh: float = 1.0,
    threads_per_block: int = 128,
) -> tuple[cp.ndarray, dict]:
    """
    GPU corner refinement using batched processing and RawKernels.

    Args:
        gray_gpu: (H, W) uint8 or float32 CuPy array.
        corners_in_gpu: (N, 4, 2) float32 CuPy array (initial corners).
        window_size: integer window radius.
        max_iters: max iterations (reduced to 5 for 14D_2).
        epsilon: convergence threshold.

    Returns:
        corners_refined_gpu: (N, 4, 2) float32 CuPy array.
        timings: dict with keys like {'t_refine_ms': float}
    """
    if cp is None:
        raise RuntimeError(f"CuPy not available for GPU corner refinement: {_gpu_import_error}")

    if gray_gpu.ndim != 2:
        raise ValueError("gpu_refine_corners expects 2D grayscale image")

    if corners_in_gpu.ndim != 3 or corners_in_gpu.shape[1] != 4 or corners_in_gpu.shape[2] != 2:
        raise ValueError(f"corners_in_gpu must be (N, 4, 2), got {corners_in_gpu.shape}")

    # Use CUDA events for accurate timing
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()

    # Convert to float32 (0-255 range, matching CPU)
    gray_f32 = gray_gpu.astype(cp.float32)
    h, w = gray_f32.shape

    # Precompute gradients (Sobel) - vectorized
    sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
    sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)

    # Pad image for gradient computation
    gray_padded = cp.pad(gray_f32, ((1, 1), (1, 1)), mode="edge")
    Ix = cp.zeros_like(gray_f32)
    Iy = cp.zeros_like(gray_f32)

    # Compute gradients using Sobel (vectorized)
    for i in range(3):
        for j in range(3):
            Ix += gray_padded[i:i+h, j:j+w] * sobel_x[i, j]
            Iy += gray_padded[i:i+h, j:j+w] * sobel_y[i, j]

    # Flatten corners to 1D arrays for batched processing
    N_quads = corners_in_gpu.shape[0]
    N_corners = N_quads * 4
    x_flat = corners_in_gpu[:, :, 0].flatten().astype(cp.float32)
    y_flat = corners_in_gpu[:, :, 1].flatten().astype(cp.float32)

    # Get kernel
    kernel = _get_refine_kernel()
    if kernel is None:
        raise RuntimeError("Failed to compile refinement kernel")

    # Launch kernel with appropriate block/grid size
    blocks_per_grid = (N_corners + threads_per_block - 1) // threads_per_block

    kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (
            gray_f32,
            h,
            w,
            Ix,
            Iy,
            x_flat,
            y_flat,
            window_size,
            max_iters,
            epsilon,
            step_max,
            det_min,
            grad_thresh,
            N_corners,
        ),
    )

    # Reshape back to (N, 4, 2)
    corners_refined = cp.stack([x_flat, y_flat], axis=1).reshape(N_quads, 4, 2)

    # Record end time
    end_event.record()
    end_event.synchronize()
    refine_ms = cp.cuda.get_elapsed_time(start_event, end_event)

    timings = {"t_refine_ms": refine_ms}

    return corners_refined, timings
