from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ensure_output_dirs, load_config
from common.pose import solve_pose_cpu, solve_pose_gpu_from_homography
from common.video import load_frames
from cpu.decode import decode_apriltag
from cpu.detector import detect_apriltag
from cpu.pipeline import run_cpu_pipeline
from tests.helpers import write_csv, save_overlay
from gpu.corners import detect_quad_gpu

try:
    from gpu.sampling import sample_gpu
    from gpu.decode import decode_gpu_bits, decode_gpu_codebook
except Exception as exc:  # pragma: no cover
    sample_gpu = None
    decode_gpu_bits = None
    decode_gpu_codebook = None
    _gpu_import_error = exc
else:
    _gpu_import_error = None

try:
    import cupy as cp
except Exception:
    cp = None

MODES = [
    "CPU_ONLY",
    "GPU_DECODE_WITH_CPU_CORNERS",
    "GPU_SAMPLING_WITH_GPU_CORNERS_COMPARE",
    "CPU_DECODE_WITH_GPU_CORNERS_COMPARE",
    "GPU_DECODE_WITH_GPU_CORNERS_COMPARE",
    "GPU_FULL_GPU_CORNERS",
    # Phase 7 perf modes
    "CPU_ONLY_PERF",
    "GPU_DECODE_WITH_CPU_CORNERS_PERF",
    "GPU_FULL_GPU_CORNERS_PERF",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation and perf modes.")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="CPU_ONLY",
        help="Validation mode.",
    )
    parser.add_argument(
        "--frames",
        choices=["SMOKE", "VALIDATE", "FULL"],
        default="VALIDATE",
        help="Frame set to use.",
    )
    return parser.parse_args()


def frame_range(args_frames: str):
    if args_frames == "SMOKE":
        return range(5)
    if args_frames == "FULL":
        return None  # signal full video
    return range(200)


def run_gpu_decode(gray: np.ndarray, corners, cfg) -> Tuple[Optional[int], Optional[int]]:
    if None in (sample_gpu, decode_gpu_bits, decode_gpu_codebook):
        raise RuntimeError(f"GPU modules unavailable: {_gpu_import_error}")

    gpu_sample = sample_gpu(gray, corners, cfg)
    gpu_bits = decode_gpu_bits(gpu_sample, cfg)
    gpu_id, gpu_hamming = decode_gpu_codebook(gpu_bits, cfg)
    return gpu_id, gpu_hamming


def ensure_gpu_modules():
    if None in (sample_gpu, decode_gpu_bits, decode_gpu_codebook):
        raise RuntimeError(f"GPU modules unavailable: {_gpu_import_error}")


def _now_ms():
    return time.perf_counter() * 1000.0


def _sync_gpu():
    if cp is not None:
        cp.cuda.Stream.null.synchronize()


def _quantiles(vals: List[float]):
    arr = np.array(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _write_perf_report(path: Path, columns: List[str], rows: List[List]):
    stats_lines = [f"frames: {len(rows)}"]
    cols_idx = {c: i for i, c in enumerate(columns)}
    for col in columns:
        vals = [r[cols_idx[col]] for r in rows if r[cols_idx[col]] is not None]
        if not vals:
            continue
        s = _quantiles(vals)
        stats_lines.append(f"{col}: mean={s['mean']:.3f} median={s['median']:.3f} p90={s['p90']:.3f} p99={s['p99']:.3f} max={s['max']:.3f}")
    if "t_cpu_total_ms" in columns:
        total_vals = [r[cols_idx["t_cpu_total_ms"]] for r in rows if r[cols_idx["t_cpu_total_ms"]] is not None]
        if total_vals:
            fps = 1000.0 / (_quantiles(total_vals)["mean"])
            stats_lines.append(f"fps_cpu: {fps:.2f}")
    if "t_gpu_total_ms" in columns:
        total_vals = [r[cols_idx["t_gpu_total_ms"]] for r in rows if r[cols_idx["t_gpu_total_ms"]] is not None]
        if total_vals:
            fps = 1000.0 / (_quantiles(total_vals)["mean"])
            stats_lines.append(f"fps_gpu: {fps:.2f}")

    # Optional CPU vs GPU comparison
    if "t_gpu_total_ms" in columns:
        total_vals_gpu = [r[cols_idx["t_gpu_total_ms"]] for r in rows if r[cols_idx["t_gpu_total_ms"]] is not None]
        # If we have a CPU baseline report, we could read it; for now, log placeholder.
        # Caller may append actual CPU FPS externally.

    # Segment drift analysis
    def segment_means(vals: List[float]):
        if not vals:
            return None, None, None
        n = len(vals)
        third = max(1, n // 3)
        first = vals[:third]
        middle = vals[third : 2 * third]
        last = vals[2 * third :]
        m_first = float(np.mean(first)) if first else None
        m_middle = float(np.mean(middle)) if middle else None
        m_last = float(np.mean(last)) if last else None
        return m_first, m_middle, m_last

    total_col = "t_cpu_total_ms" if "t_cpu_total_ms" in columns else "t_gpu_total_ms" if "t_gpu_total_ms" in columns else None
    if total_col:
        vals = [r[cols_idx[total_col]] for r in rows if r[cols_idx[total_col]] is not None]
        m_first, m_middle, m_last = segment_means(vals)
        if m_first is not None and m_last is not None and m_first != 0:
            drift_pct = (m_last - m_first) / m_first * 100.0
            stats_lines.append("segment_means_total_ms:")
            stats_lines.append(f"  first: {m_first}")
            stats_lines.append(f"  middle: {m_middle}")
            stats_lines.append(f"  last: {m_last}")
            stats_lines.append(f"drift_pct_total_ms: {drift_pct:.2f}%")
            if abs(drift_pct) > 20.0:
                stats_lines.append("WARNING: significant performance drift over video (>|20%| change in mean total_ms).")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(stats_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config()
    ensure_output_dirs(cfg)
    fr = frame_range(args.frames)
    if fr is None:
        frames = _load_all_frames(cfg["input"]["video_path"])
    else:
        frames = load_frames(cfg["input"]["video_path"], fr)

    mode = args.mode

    # Perf modes handled separately
    if mode in ("CPU_ONLY_PERF", "GPU_DECODE_WITH_CPU_CORNERS_PERF", "GPU_FULL_GPU_CORNERS_PERF"):
        run_perf_mode(mode, args.frames, frames, cfg)
        return

    debug_dir = Path("outputs/debug_phase3")

    # Mode-specific outputs
    if mode == "CPU_ONLY":
        csv_path = Path("outputs/validation_200.csv" if args.frames == "VALIDATE" else "outputs/validation_5.csv")
        report_path = Path("outputs/report_200.txt" if args.frames == "VALIDATE" else "outputs/report_5.txt")
    elif mode == "GPU_DECODE_WITH_CPU_CORNERS":
        csv_path = Path("outputs/validation_200.csv" if args.frames == "VALIDATE" else "outputs/validation_5.csv")
        report_path = Path("outputs/report_200.txt" if args.frames == "VALIDATE" else "outputs/report_5.txt")
    elif mode == "GPU_SAMPLING_WITH_GPU_CORNERS_COMPARE":
        suffix = "200" if args.frames == "VALIDATE" else "smoke5"
        csv_path = Path(f"outputs/phase5_step5A_sampling_{suffix}.csv")
        report_path = None
    elif mode == "CPU_DECODE_WITH_GPU_CORNERS_COMPARE":
        suffix = "200" if args.frames == "VALIDATE" else "smoke5"
        csv_path = Path(f"outputs/phase5_step5B_cpu_decode_gpu_corners_{suffix}.csv")
        report_path = None
    elif mode == "GPU_DECODE_WITH_GPU_CORNERS_COMPARE":
        suffix = "200" if args.frames == "VALIDATE" else "smoke5"
        csv_path = Path(f"outputs/phase5_step5C_gpu_decode_gpu_corners_{suffix}.csv")
        report_path = Path("outputs/phase5_step5C_report.txt") if args.frames == "VALIDATE" else None
    else:  # GPU_FULL_GPU_CORNERS
        csv_path = Path("outputs/phase5_gpu_full_200.csv" if args.frames == "VALIDATE" else "outputs/phase5_gpu_full_smoke5.csv")
        report_path = Path("outputs/phase5_gpu_full_report.txt") if args.frames == "VALIDATE" else None

    rows: List[List] = []

    cpu_detected_frames = 0
    gpu_detected_frames = 0
    id_matches = 0
    hamming_matches = 0
    rms_values = []

    from cpu.pipeline import run_cpu_pipeline
    from tests.helpers import save_overlay

    for idx, frame in enumerate(frames):
        detection, decode_ref = run_cpu_pipeline(frame, cfg)
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners_gpu = detect_quad_gpu(gray, cfg)

        if detection.detected:
            cpu_detected_frames += 1
        if corners_gpu is not None:
            gpu_detected_frames += 1

        if mode == "GPU_DECODE_WITH_CPU_CORNERS":
            gpu_id, gpu_hamming = (None, None)
            if detection.detected and detection.corners is not None:
                gpu_id, gpu_hamming = run_gpu_decode(gray, detection.corners, cfg)
                if gpu_id is not None:
                    id_matches += int(decode_ref.id == gpu_id)
                    hamming_matches += int(decode_ref.hamming == gpu_hamming)
            rows.append(
                [
                    idx,
                    decode_ref.id,
                    decode_ref.hamming,
                    gpu_id,
                    gpu_hamming,
                    decode_ref.id == gpu_id if gpu_id is not None else None,
                    decode_ref.hamming == gpu_hamming if gpu_hamming is not None else None,
                ]
            )
        elif mode == "GPU_SAMPLING_WITH_GPU_CORNERS_COMPARE":
            if corners_gpu is None or detection.corners is None:
                rows.append([idx, None])
                continue
            # CPU sampling using CPU corners
            cpu_sample = decode_ref.sample_grid
            # GPU sampling using GPU corners
            gpu_sample = sample_gpu(gray, corners_gpu, cfg)
            if cpu_sample is None or gpu_sample is None:
                rows.append([idx, None])
                continue
            rms = float(np.sqrt(np.mean((gpu_sample.astype(np.float32) - cpu_sample.astype(np.float32)) ** 2)))
            rms_values.append(rms)
            rows.append([idx, rms])
        elif mode == "CPU_DECODE_WITH_GPU_CORNERS_COMPARE":
            if corners_gpu is None:
                rows.append([idx, decode_ref.id, decode_ref.hamming, None, None, None, None])
                continue
            decode_gpu_corners = decode_apriltag(gray, corners_gpu, cfg)
            id_match = decode_ref.id == decode_gpu_corners.id
            hamming_match = decode_ref.hamming == decode_gpu_corners.hamming
            id_matches += int(bool(id_match))
            hamming_matches += int(bool(hamming_match))
            rows.append(
                [
                    idx,
                    decode_ref.id,
                    decode_ref.hamming,
                    decode_gpu_corners.id,
                    decode_gpu_corners.hamming,
                    id_match,
                    hamming_match,
                ]
            )
        elif mode == "GPU_DECODE_WITH_GPU_CORNERS_COMPARE":
            if corners_gpu is None:
                rows.append([idx, decode_ref.id, decode_ref.hamming, None, None, None, None])
                continue
            gpu_id, gpu_hamming = run_gpu_decode(gray, corners_gpu, cfg)
            id_match = decode_ref.id == gpu_id
            hamming_match = decode_ref.hamming == gpu_hamming
            id_matches += int(bool(id_match))
            hamming_matches += int(bool(hamming_match))
            rows.append(
                [
                    idx,
                    decode_ref.id,
                    decode_ref.hamming,
                    gpu_id,
                    gpu_hamming,
                    id_match,
                    hamming_match,
                ]
            )
        elif mode == "GPU_FULL_GPU_CORNERS":
            gpu_id = None
            gpu_hamming = None
            if corners_gpu is not None:
                gpu_id, gpu_hamming = run_gpu_decode(gray, corners_gpu, cfg)
            id_match = (decode_ref.id == gpu_id) if (decode_ref.id is not None and gpu_id is not None) else None
            hamming_match = (
                decode_ref.hamming == gpu_hamming if (decode_ref.hamming is not None and gpu_hamming is not None) else None
            )
            if id_match:
                id_matches += 1
            if hamming_match:
                hamming_matches += 1
            rows.append(
                [
                    idx,
                    gpu_id,
                    gpu_hamming,
                    decode_ref.id,
                    decode_ref.hamming,
                    id_match,
                    hamming_match,
                ]
            )
        else:  # CPU_ONLY
            rows.append([idx, decode_ref.id, decode_ref.hamming, None, None, None, None])

        # Overlays for modes that involve GPU corners
        if mode in (
            "GPU_DECODE_WITH_CPU_CORNERS",
            "GPU_SAMPLING_WITH_GPU_CORNERS_COMPARE",
            "CPU_DECODE_WITH_GPU_CORNERS_COMPARE",
            "GPU_DECODE_WITH_GPU_CORNERS_COMPARE",
            "GPU_FULL_GPU_CORNERS",
        ):
            if detection.detected:
                save_overlay(
                    frame,
                    detection.corners,
                    decode_ref.id,
                    debug_dir / f"frame_{idx:03d}_cpu_overlay.png",
                )
            if corners_gpu is not None:
                save_overlay(
                    frame,
                    corners_gpu,
                    decode_ref.id if decode_ref.id is not None else -1,
                    debug_dir / f"frame_{idx:03d}_gpu_overlay.png",
                )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, _header_for_mode(mode), rows)

    if report_path:
        report_lines = _report_for_mode(
            mode,
            rows,
            len(frames),
            cpu_detected_frames,
            gpu_detected_frames,
            id_matches,
            hamming_matches,
            rms_values,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print("Validation complete:", report_path)
    else:
        print("Validation complete:", csv_path)


def run_perf_mode(mode: str, frameset: str, frames: List[np.ndarray], cfg: Dict[str, Any]) -> None:
    ensure_gpu_modules() if "GPU" in mode else None

    if mode == "CPU_ONLY_PERF":
        if frameset == "VALIDATE":
            csv_path = Path("outputs/phase7_cpu_only_perf_200.csv")
            report_path = Path("outputs/phase7_cpu_only_perf_report.txt")
        elif frameset == "SMOKE":
            csv_path = Path("outputs/phase7_cpu_only_perf_smoke5.csv")
            report_path = Path("outputs/phase7_cpu_only_perf_report.txt")
        else:  # FULL
            csv_path = Path("outputs/phase9_cpu_only_perf_full.csv")
            report_path = Path("outputs/phase9_cpu_only_perf_full_report.txt")
        rows = []
        for idx, frame in enumerate(frames):
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t0 = _now_ms()
            t_detect0 = _now_ms()
            det = detect_apriltag(frame, cfg)
            t_detect1 = _now_ms()

            if det.detected and det.corners is not None:
                t_decode0 = _now_ms()
                dec = decode_apriltag(gray, det.corners, cfg)
                t_decode1 = _now_ms()
                t_pose0 = _now_ms()
                solve_pose_cpu(det.corners, gray.shape[:2], cfg)
                t_pose1 = _now_ms()
            else:
                dec = None
                t_decode0 = t_decode1 = _now_ms()
                t_pose0 = t_pose1 = _now_ms()

            t1 = _now_ms()
            rows.append(
                [
                    idx,
                    0.0,  # capture placeholder
                    t_detect1 - t_detect0,
                    (t_decode1 - t_decode0) if dec is not None else None,
                    (t_pose1 - t_pose0) if det.detected else None,
                    t1 - t0,
                ]
            )
        header = ["frame_index", "t_cpu_capture_ms", "t_cpu_detect_ms", "t_cpu_decode_ms", "t_cpu_pose_ms", "t_cpu_total_ms"]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(csv_path, header, rows)
        _write_perf_report(report_path, header[1:], rows)
        print(f"Perf CPU_ONLY complete: {csv_path} frames={len(rows)}")
        return

    if mode == "GPU_DECODE_WITH_CPU_CORNERS_PERF":
        if frameset == "VALIDATE":
            csv_path = Path("outputs/phase7_gpu_cpu_corners_perf_200.csv")
            report_path = Path("outputs/phase7_gpu_cpu_corners_perf_report.txt")
        elif frameset == "SMOKE":
            csv_path = Path("outputs/phase7_gpu_cpu_corners_perf_smoke5.csv")
            report_path = Path("outputs/phase7_gpu_cpu_corners_perf_report.txt")
        else:
            csv_path = Path("outputs/phase9_gpu_cpu_corners_perf_full.csv")
            report_path = Path("outputs/phase9_gpu_cpu_corners_perf_full_report.txt")
        rows = []
        for idx, frame in enumerate(frames):
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            det = detect_apriltag(frame, cfg)
            if not det.detected or det.corners is None:
                rows.append([idx, None, None, None, None, None, None])
                continue
            t_total0 = _now_ms()
            t_sampling0 = _now_ms()
            gpu_sample = sample_gpu(gray, det.corners, cfg)
            _sync_gpu()
            t_sampling1 = _now_ms()

            t_decode0 = _now_ms()
            gpu_bits = decode_gpu_bits(gpu_sample, cfg)
            gpu_id, gpu_hamming = decode_gpu_codebook(gpu_bits, cfg)
            _sync_gpu()
            t_decode1 = _now_ms()

            t_pose0 = _now_ms()
            solve_pose_gpu_from_homography(det.corners, gray.shape[:2], cfg)
            t_pose1 = _now_ms()
            t_total1 = _now_ms()

            rows.append(
                [
                    idx,
                    0.0,  # copy_in placeholder
                    0.0,  # preprocess placeholder
                    t_sampling1 - t_sampling0,
                    t_decode1 - t_decode0,
                    t_pose1 - t_pose0,
                    t_total1 - t_total0,
                ]
            )
        header = [
            "frame_index",
            "t_gpu_copy_in_ms",
            "t_gpu_preprocess_ms",
            "t_gpu_sampling_ms",
            "t_gpu_decode_ms",
            "t_gpu_pose_ms",
            "t_gpu_total_ms",
        ]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(csv_path, header, rows)
        _write_perf_report(report_path, header[1:], rows)
        print(f"Perf GPU decode with CPU corners complete: {csv_path} frames={len(rows)}")
        return

    if mode == "GPU_FULL_GPU_CORNERS_PERF":
        if frameset == "VALIDATE":
            csv_path = Path("outputs/phase7_gpu_full_perf_200.csv")
            report_path = Path("outputs/phase7_gpu_full_perf_report.txt")
        elif frameset == "SMOKE":
            csv_path = Path("outputs/phase7_gpu_full_perf_smoke5.csv")
            report_path = Path("outputs/phase7_gpu_full_perf_report.txt")
        else:
            csv_path = Path("outputs/phase10_gpu_full_perf_full.csv")
            report_path = Path("outputs/phase10_gpu_full_perf_full_report.txt")
        rows = []
        for idx, frame in enumerate(frames):
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t_total0 = _now_ms()

            t_corners0 = _now_ms()
            corners_gpu = detect_quad_gpu(gray, cfg)
            t_corners1 = _now_ms()
            if corners_gpu is None:
                rows.append([idx, 0.0, 0.0, None, None, None, None, t_corners1 - t_corners0])
                continue

            t_sampling0 = _now_ms()
            gpu_sample = sample_gpu(gray, corners_gpu, cfg)
            _sync_gpu()
            t_sampling1 = _now_ms()

            t_decode0 = _now_ms()
            gpu_bits = decode_gpu_bits(gpu_sample, cfg)
            gpu_id, gpu_hamming = decode_gpu_codebook(gpu_bits, cfg)
            _sync_gpu()
            t_decode1 = _now_ms()

            t_pose0 = _now_ms()
            solve_pose_gpu_from_homography(corners_gpu, gray.shape[:2], cfg)
            t_pose1 = _now_ms()

            t_total1 = _now_ms()

            rows.append(
                [
                    idx,
                    0.0,  # copy_in placeholder
                    0.0,  # preprocess placeholder
                    t_corners1 - t_corners0,
                    t_sampling1 - t_sampling0,
                    t_decode1 - t_decode0,
                    t_pose1 - t_pose0,
                    t_total1 - t_total0,
                ]
            )
        header = [
            "frame_index",
            "t_gpu_copy_in_ms",
            "t_gpu_preprocess_ms",
            "t_gpu_corners_ms",
            "t_gpu_sampling_ms",
            "t_gpu_decode_ms",
            "t_gpu_pose_ms",
            "t_gpu_total_ms",
        ]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(csv_path, header, rows)
        _write_perf_report(report_path, header[1:], rows)
        # Append Phase-9 baseline comparison for Phase-10 full runs.
        if "phase10_gpu_full_perf_full_report" in str(report_path):
            total_vals = [r[7] for r in rows if r[7] is not None]
            if total_vals:
                mean_total = sum(total_vals) / len(total_vals)
                lines = report_path.read_text(encoding="utf-8").splitlines()
                lines.append("baseline_phase9_mean_total_ms: 14.48")
                lines.append("baseline_phase9_fps: 69.0")
                lines.append(f"phase10_mean_total_ms: {mean_total}")
                lines.append(f"phase10_fps: {1000.0/mean_total}")
                lines.append(f"speedup_total: {(1000.0/mean_total)/69.0}x")
                report_path.write_text("\n".join(lines), encoding="utf-8")
        print("Perf GPU full pipeline complete:", csv_path)
        return


def _header_for_mode(mode: str):
    if mode == "GPU_SAMPLING_WITH_GPU_CORNERS_COMPARE":
        return ["frame_index", "rms"]
    if mode == "CPU_DECODE_WITH_GPU_CORNERS_COMPARE":
        return [
            "frame_index",
            "id_cpu_ref",
            "hamming_cpu_ref",
            "id_cpu_gpu_corners",
            "hamming_cpu_gpu_corners",
            "id_match_flag",
            "hamming_match_flag",
        ]
    if mode == "GPU_DECODE_WITH_GPU_CORNERS_COMPARE":
        return [
            "frame_index",
            "id_cpu_ref",
            "hamming_cpu_ref",
            "id_gpu",
            "hamming_gpu",
            "id_match_flag",
            "hamming_match_flag",
        ]
    if mode == "GPU_FULL_GPU_CORNERS":
        return [
            "frame_index",
            "id_gpu",
            "hamming_gpu",
            "id_cpu_ref",
            "hamming_cpu_ref",
            "id_match_flag",
            "hamming_match_flag",
        ]
    # Default / legacy
    return [
        "frame_index",
        "id_cpu",
        "hamming_cpu",
        "id_gpu",
        "hamming_gpu",
        "id_match_flag",
        "hamming_match_flag",
    ]


def _mean_safe(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _report_for_mode(
    mode: str,
    rows: List[List],
    total_frames: int,
    cpu_detected_frames: int,
    gpu_detected_frames: int,
    id_matches: int,
    hamming_matches: int,
    rms_values: List[float],
):
    if mode == "GPU_SAMPLING_WITH_GPU_CORNERS_COMPARE":
        mean_rms = _mean_safe(rms_values)
        max_rms = max(rms_values) if rms_values else None
        return [
            f"frames_processed: {total_frames}",
            f"mean_rms: {mean_rms}",
            f"max_rms: {max_rms}",
        ]

    if mode in ("GPU_DECODE_WITH_GPU_CORNERS_COMPARE", "GPU_FULL_GPU_CORNERS"):
        id_match_rate = id_matches / total_frames if total_frames else None
        hamming_match_rate = hamming_matches / total_frames if total_frames else None
        return [
            f"frames_processed: {total_frames}",
            f"detection_rate_cpu: {cpu_detected_frames/total_frames if total_frames else None}",
            f"detection_rate_gpu: {gpu_detected_frames/total_frames if total_frames else None}",
            f"id_match_rate: {id_match_rate}",
            f"hamming_match_rate: {hamming_match_rate}",
        ]

    # Default / CPU_ONLY
    return [
        f"frames_processed: {total_frames}",
        f"cpu_detection_rate: {cpu_detected_frames/total_frames if total_frames else None}",
        f"gpu_detection_rate: {gpu_detected_frames/total_frames if total_frames else None}",
        f"id_match_rate: {id_matches/total_frames if total_frames else None}",
        f"average_hamming_cpu: {_mean_safe([r[2] for r in rows])}",
        f"average_hamming_gpu: {_mean_safe([r[4] for r in rows])}",
    ]


def _load_all_frames(video_path: str):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        idx += 1
    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path} (reported {total})")
    return frames


if __name__ == "__main__":
    main()

