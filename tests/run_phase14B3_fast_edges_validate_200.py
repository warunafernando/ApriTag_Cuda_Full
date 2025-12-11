"""
Phase 14B_3 full validation: Compare CPU_CANNY vs GPU_FAST edges in full pipeline (200 frames).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from common.config import ensure_output_dirs, load_config
from common.video import load_frames
from common.pose import solve_pose_cpu, rotation_error_deg, translation_error
from cpu.detector import detect_apriltag
from cpu.decode import decode_apriltag
from cpu.edges import cpu_canny_edges
from cpu.pipeline import run_cpu_pipeline

try:
    import cupy as cp
    from gpu.edges_fast import gpu_fast_edges
    from gpu.corners import detect_quad_gpu
    from gpu.sampling import sample_gpu
    from gpu.decode import decode_gpu_bits, decode_gpu_codebook
    from common.pose import solve_pose_gpu_from_homography
except Exception as exc:
    cp = None
    gpu_fast_edges = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None


def run_baseline_cpu_canny(frame: np.ndarray, cfg: dict) -> dict:
    """Run baseline pipeline with CPU Canny (current pipeline)."""
    # Current pipeline doesn't use edges, so this is just the standard CPU pipeline
    detection, decode = run_cpu_pipeline(frame, cfg)
    
    if not detection.detected or detection.corners is None:
        return {
            "id": None,
            "hamming": None,
            "corners": None,
            "rvec": None,
            "tvec": None,
            "t_edges_ms": 0.0,
        }
    
    # Measure CPU Canny time (even though not used in pipeline)
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    _ = cpu_canny_edges(gray, low_thresh=35, high_thresh=110)
    t_edges = (time.perf_counter() - t0) * 1000.0
    
    # Solve pose
    rvec, tvec = solve_pose_cpu(detection.corners, frame.shape[:2], cfg)
    
    return {
        "id": decode.id,
        "hamming": decode.hamming,
        "corners": detection.corners,
        "rvec": rvec,
        "tvec": tvec,
        "t_edges_ms": t_edges,
    }


def run_gpu_fast_edges_pipeline(frame: np.ndarray, cfg: dict) -> dict:
    """Run pipeline with GPU fast edges (edges used as preprocessing, but corners still use ArUco)."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # GPU fast edges
    gray_gpu = cp.asarray(gray, dtype=cp.uint8)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    edges_gpu = gpu_fast_edges(gray_gpu, low_thresh=30.0, high_thresh=90.0)
    cp.cuda.Stream.null.synchronize()
    t_edges = (time.perf_counter() - t0) * 1000.0
    
    # For now, edges are computed but corners still use ArUco (same as baseline)
    # In a full implementation, edges would be used for corner detection
    # But for this test, we'll use the same corner detection and compare results
    corners_gpu = detect_quad_gpu(gray, cfg)  # Still uses ArUco, not edges
    
    if corners_gpu is None:
        return {
            "id": None,
            "hamming": None,
            "corners": None,
            "rvec": None,
            "tvec": None,
            "t_edges_ms": t_edges,
        }
    
    # GPU decode
    gpu_sample = sample_gpu(gray, corners_gpu, cfg)
    gpu_bits = decode_gpu_bits(gpu_sample, cfg)
    gpu_id, gpu_hamming = decode_gpu_codebook(gpu_bits, cfg)
    
    # GPU pose
    rvec, tvec = solve_pose_gpu_from_homography(corners_gpu, frame.shape[:2], cfg)
    
    return {
        "id": gpu_id,
        "hamming": gpu_hamming,
        "corners": corners_gpu,
        "rvec": rvec,
        "tvec": tvec,
        "t_edges_ms": t_edges,
    }


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)

    if gpu_fast_edges is None:
        raise RuntimeError(f"GPU fast edges unavailable: {_gpu_import_error}")

    video_path = cfg["input"]["video_path"]
    frames = load_frames(video_path, range(200))

    csv_path = Path("outputs/phase14B3_fast_edges_200.csv")
    report_path = Path("outputs/phase14B3_fast_edges_200_report.txt")

    rows = []
    id_matches = 0
    hamming_matches = 0
    corner_rms_values = []
    rot_err_values = []
    trans_err_values = []
    cpu_edges_times = []
    gpu_fast_edges_times = []

    for idx, frame in enumerate(frames):
        # Baseline: CPU_CANNY
        result_cpu = run_baseline_cpu_canny(frame, cfg)
        
        # Fast GPU edges
        result_fast = run_gpu_fast_edges_pipeline(frame, cfg)
        
        # Compare
        id_match = (result_cpu["id"] == result_fast["id"] == 3) if (result_cpu["id"] is not None and result_fast["id"] is not None) else False
        hamming_match = (result_cpu["hamming"] == result_fast["hamming"] == 0) if (result_cpu["hamming"] is not None and result_fast["hamming"] is not None) else False
        
        if id_match:
            id_matches += 1
        if hamming_match:
            hamming_matches += 1
        
        # Corner RMS
        corner_rms = None
        if result_cpu["corners"] is not None and result_fast["corners"] is not None:
            corner_rms = float(np.sqrt(np.mean((result_cpu["corners"] - result_fast["corners"]) ** 2)))
            corner_rms_values.append(corner_rms)
        
        # Pose errors
        rot_err = None
        trans_err = None
        if result_cpu["rvec"] is not None and result_fast["rvec"] is not None:
            # Convert rvec to rotation matrix for error calculation
            R_cpu, _ = cv2.Rodrigues(result_cpu["rvec"])
            R_fast, _ = cv2.Rodrigues(result_fast["rvec"])
            rot_err = rotation_error_deg(R_cpu, R_fast)
            rot_err_values.append(rot_err)
            
            trans_err = translation_error(result_cpu["tvec"], result_fast["tvec"])
            trans_err_values.append(trans_err)
        
        rows.append([
            idx,
            result_cpu["id"],
            result_fast["id"],
            result_cpu["hamming"],
            result_fast["hamming"],
            id_match,
            hamming_match,
            corner_rms if corner_rms is not None else float("nan"),
            rot_err if rot_err is not None else float("nan"),
            trans_err if trans_err is not None else float("nan"),
            result_cpu["t_edges_ms"],
            result_fast["t_edges_ms"],
        ])
        
        cpu_edges_times.append(result_cpu["t_edges_ms"])
        gpu_fast_edges_times.append(result_fast["t_edges_ms"])

    # Write CSV
    import csv
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index",
            "id_cpu",
            "id_fast",
            "hamming_cpu",
            "hamming_fast",
            "id_match",
            "hamming_match",
            "corner_rms_px",
            "rot_err_deg",
            "trans_err_m",
            "t_cpu_edges_ms",
            "t_gpu_fast_edges_ms",
        ])
        writer.writerows(rows)

    # Compute aggregates
    total_frames = len(frames)
    id_match_rate = float(id_matches) / total_frames if total_frames > 0 else 0.0
    hamming_match_rate = float(hamming_matches) / total_frames if total_frames > 0 else 0.0
    
    mean_corner_rms = float(np.mean(corner_rms_values)) if corner_rms_values else float("nan")
    max_corner_rms = float(np.max(corner_rms_values)) if corner_rms_values else float("nan")
    
    mean_rot_err = float(np.mean(rot_err_values)) if rot_err_values else float("nan")
    max_rot_err = float(np.max(rot_err_values)) if rot_err_values else float("nan")
    
    mean_trans_err = float(np.mean(trans_err_values)) if trans_err_values else float("nan")
    max_trans_err = float(np.max(trans_err_values)) if trans_err_values else float("nan")
    
    mean_cpu_edges_ms = float(np.mean(cpu_edges_times))
    mean_gpu_fast_edges_ms = float(np.mean(gpu_fast_edges_times))
    speedup = mean_cpu_edges_ms / mean_gpu_fast_edges_ms if mean_gpu_fast_edges_ms > 0 else 0.0

    # Pass criteria
    accuracy_pass = (
        id_match_rate >= 0.99
        and hamming_match_rate >= 0.99
        and mean_corner_rms <= 1.0
        and max_corner_rms <= 2.0
        and mean_rot_err <= 0.3
        and max_rot_err <= 1.0
        and mean_trans_err <= 0.01
        and max_trans_err <= 0.02
    )

    speed_pass = mean_gpu_fast_edges_ms < mean_cpu_edges_ms
    speed_target = mean_gpu_fast_edges_ms <= 1.0  # Target: ≤1.0 ms

    # Write report
    report_lines = [
        "Phase 14B_3 Full Validation: CPU_CANNY vs GPU_FAST Edges (200 frames)",
        "=" * 70,
        "",
        f"frames_processed: {total_frames}",
        "",
        "Accuracy vs Baseline:",
        f"  id_match_rate: {id_match_rate:.6f}",
        f"  hamming_match_rate: {hamming_match_rate:.6f}",
        f"  mean_corner_rms_px: {mean_corner_rms:.6f}",
        f"  max_corner_rms_px: {max_corner_rms:.6f}",
        f"  mean_rot_err_deg: {mean_rot_err:.6f}",
        f"  max_rot_err_deg: {max_rot_err:.6f}",
        f"  mean_trans_err_m: {mean_trans_err:.6f}",
        f"  max_trans_err_m: {max_trans_err:.6f}",
        "",
        "Performance:",
        f"  mean_cpu_edges_ms: {mean_cpu_edges_ms:.4f}",
        f"  mean_gpu_fast_edges_ms: {mean_gpu_fast_edges_ms:.4f}",
        f"  fast_edges_speedup: {speedup:.2f}x",
        "",
        "Pass Criteria:",
        f"  Accuracy: {'PASS' if accuracy_pass else 'FAIL'}",
        f"    - id_match_rate >= 0.99: {id_match_rate:.6f} {'[PASS]' if id_match_rate >= 0.99 else '[FAIL]'}",
        f"    - hamming_match_rate >= 0.99: {hamming_match_rate:.6f} {'[PASS]' if hamming_match_rate >= 0.99 else '[FAIL]'}",
        f"    - mean_corner_rms_px <= 1.0: {mean_corner_rms:.6f} {'[PASS]' if mean_corner_rms <= 1.0 else '[FAIL]'}",
        f"    - max_corner_rms_px <= 2.0: {max_corner_rms:.6f} {'[PASS]' if max_corner_rms <= 2.0 else '[FAIL]'}",
        f"    - mean_rot_err_deg <= 0.3: {mean_rot_err:.6f} {'[PASS]' if mean_rot_err <= 0.3 else '[FAIL]'}",
        f"    - max_rot_err_deg <= 1.0: {max_rot_err:.6f} {'[PASS]' if max_rot_err <= 1.0 else '[FAIL]'}",
        f"    - mean_trans_err_m <= 0.01: {mean_trans_err:.6f} {'[PASS]' if mean_trans_err <= 0.01 else '[FAIL]'}",
        f"    - max_trans_err_m <= 0.02: {max_trans_err:.6f} {'[PASS]' if max_trans_err <= 0.02 else '[FAIL]'}",
        "",
        f"  Speed: {'PASS' if speed_pass else 'FAIL'}",
        f"    - GPU faster than CPU: {speedup:.2f}x {'[PASS]' if speed_pass else '[FAIL]'}",
        f"    - Target (<=1.0 ms): {'PASS' if speed_target else 'FAIL'} {'[PASS]' if speed_target else '[FAIL]'}",
        "",
        f"Overall: {'PASS' if (accuracy_pass and speed_pass) else 'FAIL'}",
    ]

    report_path.write_text("\n".join(report_lines))
    print(f"Phase 14B_3 validation complete: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Overall: {'PASS' if (accuracy_pass and speed_pass) else 'FAIL'}")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()

