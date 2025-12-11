"""
Phase 14D_3 deep diagnostics: Patch-level, trajectory, and LK conditioning analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from common.config import ensure_output_dirs, load_config
from common.video import load_frames
from cpu.corner_refine import cpu_refine_corners
from cpu.edges import cpu_canny_edges
from cpu.quads import cpu_quad_candidates_from_edges
from gpu.corner_refine import gpu_refine_corners

try:
    import cupy as cp
except Exception as exc:
    cp = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None


def select_best_quad(quads: np.ndarray) -> np.ndarray | None:
    """Select best quad candidate based on area."""
    if quads.shape[0] == 0:
        return None
    areas = [cv2.contourArea(quad) for quad in quads]
    best_idx = np.argmax(areas)
    return quads[best_idx]


def compute_gradients_cpu(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute gradients using OpenCV Sobel (matching CPU reference)."""
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return Ix, Iy


def compute_gradients_gpu(gray_gpu: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """Compute gradients using CuPy Sobel (matching GPU implementation)."""
    gray_f32 = gray_gpu.astype(cp.float32)
    sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
    sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)

    gray_padded = cp.pad(gray_f32, ((1, 1), (1, 1)), mode="edge")
    h, w = gray_f32.shape
    Ix = cp.zeros_like(gray_f32)
    Iy = cp.zeros_like(gray_f32)

    for i in range(3):
        for j in range(3):
            Ix += gray_padded[i:i+h, j:j+w] * sobel_x[i, j]
            Iy += gray_padded[i:i+h, j:j+w] * sobel_y[i, j]

    return Ix, Iy


def extract_patch(img: np.ndarray, x: float, y: float, window_size: int) -> np.ndarray:
    """Extract patch around (x, y) with bilinear sampling."""
    x_int = int(x)
    y_int = int(y)
    x_min = max(0, x_int - window_size)
    x_max = min(img.shape[1], x_int + window_size + 1)
    y_min = max(0, y_int - window_size)
    y_max = min(img.shape[0], y_int + window_size + 1)
    return img[y_min:y_max, x_min:x_max]


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    if cp is None:
        raise RuntimeError(f"CuPy not available: {_gpu_import_error}")

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(5))

    # Get parameters
    edges_cfg = cfg.get("edges", {})
    low_thresh = int(edges_cfg.get("low_thresh", 35))
    high_thresh = int(edges_cfg.get("high_thresh", 110))
    refine_cfg = cfg.get("corner_refine", {})
    window_size = int(refine_cfg.get("window_size", 5))

    # Create output directories
    patches_dir = Path("outputs/phase14D3_patches")
    patches_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = Path("outputs/phase14D3_traj")
    traj_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "Phase 14D_3 Deep Diagnostics: Patch-Level Analysis",
        "=" * 70,
        "",
    ]

    for idx, frame in enumerate(frames):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges_cpu = cpu_canny_edges(gray, low_thresh=low_thresh, high_thresh=high_thresh)
        quads_cpu = cpu_quad_candidates_from_edges(edges_cpu, cfg)
        quad_best = select_best_quad(quads_cpu)

        if quad_best is None:
            continue

        corners_in = quad_best.reshape(1, 4, 2).astype(np.float32)

        # Compute gradients
        Ix_cpu, Iy_cpu = compute_gradients_cpu(gray.astype(np.float32))
        gray_gpu = cp.asarray(gray, dtype=cp.uint8)
        Ix_gpu, Iy_gpu = compute_gradients_gpu(gray_gpu)
        Ix_gpu_cpu = cp.asnumpy(Ix_gpu)
        Iy_gpu_cpu = cp.asnumpy(Iy_gpu)

        # Refine corners
        corners_cpu = cpu_refine_corners(gray, corners_in, window_size, 5, 0.01)
        corners_cpu = corners_cpu[0]
        corners_in_gpu = cp.asarray(corners_in, dtype=cp.float32)
        corners_gpu, _ = gpu_refine_corners(gray_gpu, corners_in_gpu, window_size, 5, 0.01)
        corners_gpu_cpu = cp.asnumpy(corners_gpu[0])

        summary_lines.append(f"frame {idx}")
        summary_lines.append("")

        for corner_idx in range(min(4, corners_cpu.shape[0])):
            # Initial corner position
            x_init = corners_in[0, corner_idx, 0]
            y_init = corners_in[0, corner_idx, 1]

            # Final positions
            x_cpu = corners_cpu[corner_idx, 0]
            y_cpu = corners_cpu[corner_idx, 1]
            x_gpu = corners_gpu_cpu[corner_idx, 0]
            y_gpu = corners_gpu_cpu[corner_idx, 1]

            # Extract patches at initial position
            patch_cpu_I = extract_patch(gray.astype(np.float32), x_init, y_init, window_size)
            patch_cpu_Ix = extract_patch(Ix_cpu, x_init, y_init, window_size)
            patch_cpu_Iy = extract_patch(Iy_cpu, x_init, y_init, window_size)

            patch_gpu_I = extract_patch(gray.astype(np.float32), x_init, y_init, window_size)
            patch_gpu_Ix = extract_patch(Ix_gpu_cpu, x_init, y_init, window_size)
            patch_gpu_Iy = extract_patch(Iy_gpu_cpu, x_init, y_init, window_size)

            # Compute patch differences
            diff_I = np.abs(patch_cpu_I - patch_gpu_I)
            diff_Ix = np.abs(patch_cpu_Ix - patch_gpu_Ix)
            diff_Iy = np.abs(patch_cpu_Iy - patch_gpu_Iy)

            mean_diff_I = np.mean(diff_I)
            mean_diff_Ix = np.mean(diff_Ix)
            mean_diff_Iy = np.mean(diff_Iy)

            summary_lines.append(
                f"  corner {corner_idx}: "
                f"mean_diff_I={mean_diff_I:.4f}, "
                f"mean_diff_Ix={mean_diff_Ix:.4f}, "
                f"mean_diff_Iy={mean_diff_Iy:.4f}"
            )

            # Save patches
            np.save(patches_dir / f"frame{idx}_corner{corner_idx}_cpu_I.npy", patch_cpu_I)
            np.save(patches_dir / f"frame{idx}_corner{corner_idx}_gpu_I.npy", patch_gpu_I)
            np.save(patches_dir / f"frame{idx}_corner{corner_idx}_cpu_Ix.npy", patch_cpu_Ix)
            np.save(patches_dir / f"frame{idx}_corner{corner_idx}_gpu_Ix.npy", patch_gpu_Ix)
            np.save(patches_dir / f"frame{idx}_corner{corner_idx}_cpu_Iy.npy", patch_cpu_Iy)
            np.save(patches_dir / f"frame{idx}_corner{corner_idx}_gpu_Iy.npy", patch_gpu_Iy)

            # Save trajectory (initial and final)
            traj_cpu = f"initial: ({x_init:.4f}, {y_init:.4f})\nfinal: ({x_cpu:.4f}, {y_cpu:.4f})\n"
            traj_gpu = f"initial: ({x_init:.4f}, {y_init:.4f})\nfinal: ({x_gpu:.4f}, {y_gpu:.4f})\n"
            (traj_dir / f"frame{idx}_corner{corner_idx}_cpu.txt").write_text(traj_cpu)
            (traj_dir / f"frame{idx}_corner{corner_idx}_gpu.txt").write_text(traj_gpu)

        summary_lines.append("")

    # Write summary
    summary_path = Path("outputs/phase14D3_patches_summary.txt")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Phase 14D_3 diagnostics complete: {summary_path}")


if __name__ == "__main__":
    main()

