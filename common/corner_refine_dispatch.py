"""
Corner refinement dispatch function supporting CPU, GPU, HYBRID, and AUTO modes.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:
    import cupy as cp
    from gpu.corner_refine import gpu_refine_corners
except Exception as exc:
    cp = None
    gpu_refine_corners = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None

from cpu.corner_refine import cpu_refine_corners


def dispatch_corner_refine(
    gray: np.ndarray,
    corners_in: np.ndarray,
    cfg: Dict[str, Any],
    image_shape: tuple[int, int] | None = None,
    num_tags: int = 1,
) -> tuple[np.ndarray, dict]:
    """
    Dispatch corner refinement based on config mode.

    Args:
        gray: Grayscale image (CPU numpy array)
        corners_in: (N, 4, 2) float32 array of initial corners
        cfg: Configuration dict
        image_shape: (height, width) of image (for AUTO mode)
        num_tags: Number of detected tags (for AUTO mode)

    Returns:
        corners_refined: (N, 4, 2) float32 refined corners
        timings: dict with timing information
    """
    refine_cfg = cfg.get("corner_refine", {})
    mode = refine_cfg.get("mode", "CPU")
    window_size = int(refine_cfg.get("window_size", 5))
    max_iters = int(refine_cfg.get("max_iters", 5))
    epsilon = float(refine_cfg.get("epsilon", 0.01))
    allow_failover = bool(refine_cfg.get("allow_failover", False))
    fallback_mode = refine_cfg.get("fallback_mode", "CPU")

    # GPU-specific parameters
    step_max = float(refine_cfg.get("step_max", 1.0))
    det_min = float(refine_cfg.get("det_min", 1e-4))
    grad_thresh = float(refine_cfg.get("grad_thresh", 1.0))
    threads_per_block = int(refine_cfg.get("threads_per_block", 128))

    if mode == "CPU":
        corners_refined = cpu_refine_corners(gray, corners_in, window_size, max_iters, epsilon)
        timings = {"t_refine_ms": 0.0}  # CPU timing not measured here
        return corners_refined, timings

    elif mode == "GPU":
        if gpu_refine_corners is None:
            if allow_failover:
                # Fallback to specified mode
                if fallback_mode == "CPU":
                    corners_refined = cpu_refine_corners(gray, corners_in, window_size, max_iters, epsilon)
                    timings = {"t_refine_ms": 0.0}
                    return corners_refined, timings
                else:
                    raise ValueError(f"Fallback mode {fallback_mode} not supported")
            else:
                # No failover - propagate error
                raise RuntimeError(f"GPU corner refinement unavailable: {_gpu_import_error}")
        
        try:
            gray_gpu = cp.asarray(gray, dtype=cp.uint8)
            corners_in_gpu = cp.asarray(corners_in, dtype=cp.float32)
            corners_refined_gpu, timings = gpu_refine_corners(
                gray_gpu, corners_in_gpu, window_size, max_iters, epsilon,
                step_max, det_min, grad_thresh, threads_per_block
            )
            corners_refined = cp.asnumpy(corners_refined_gpu)
            return corners_refined, timings
        except Exception as e:
            if allow_failover:
                # Fallback to specified mode
                if fallback_mode == "CPU":
                    corners_refined = cpu_refine_corners(gray, corners_in, window_size, max_iters, epsilon)
                    timings = {"t_refine_ms": 0.0}
                    return corners_refined, timings
                else:
                    raise ValueError(f"Fallback mode {fallback_mode} not supported")
            else:
                # No failover - propagate error
                raise

    elif mode == "HYBRID":
        # GPU for initial refinement, CPU for final polish
        if gpu_refine_corners is None:
            raise RuntimeError(f"GPU corner refinement unavailable: {_gpu_import_error}")
        gray_gpu = cp.asarray(gray, dtype=cp.uint8)
        corners_in_gpu = cp.asarray(corners_in, dtype=cp.float32)
        corners_gpu, timings_gpu = gpu_refine_corners(
            gray_gpu, corners_in_gpu, window_size, max_iters - 1, epsilon,
            step_max, det_min, grad_thresh, threads_per_block
        )
        corners_cpu_input = cp.asnumpy(corners_gpu)
        # Final polish with CPU
        corners_refined = cpu_refine_corners(
            gray, corners_cpu_input, window_size, 1, epsilon
        )
        timings = timings_gpu
        return corners_refined, timings

    elif mode == "AUTO":
        # Decide based on resolution and tag count
        gpu_min_tags = int(refine_cfg.get("gpu_min_tags", 8))
        gpu_min_resolution = refine_cfg.get("gpu_min_resolution", [1600, 1200])
        min_h, min_w = gpu_min_resolution

        if image_shape is None:
            image_shape = gray.shape[:2]
        h, w = image_shape

        use_gpu = (h >= min_h and w >= min_w) and (num_tags >= gpu_min_tags)

        if use_gpu:
            if gpu_refine_corners is None:
                use_gpu = False
            else:
                gray_gpu = cp.asarray(gray, dtype=cp.uint8)
                corners_in_gpu = cp.asarray(corners_in, dtype=cp.float32)
                corners_refined_gpu, timings = gpu_refine_corners(
                    gray_gpu, corners_in_gpu, window_size, max_iters, epsilon,
                    step_max, det_min, grad_thresh, threads_per_block
                )
                corners_refined = cp.asnumpy(corners_refined_gpu)
                return corners_refined, timings

        # Fall back to CPU
        corners_refined = cpu_refine_corners(gray, corners_in, window_size, max_iters, epsilon)
        timings = {"t_refine_ms": 0.0}
        return corners_refined, timings

    else:
        raise ValueError(f"Unknown corner_refine mode: {mode}")

