# Phases 10–13 Summary (GPU Full, Pose, Tracking, Perf)

## Status
- GPU_FULL correctness: IDs/Hamming match CPU; pose within Phase-8 tolerances.
- GPU pose: real homography-based solver active (`pose.gpu.force_cpu_pose=false`).
- Tracking: DETECT/TRACK state machine with KLT; validation and perf passing.
- ArUco tuning: adopted profile `fast2` (see gpu/corners.py) from Phase-13 sweep.

## Key Scripts
- Baseline (no tracking): `tests/run_phase12_baseline_gpu_full.py`
- Tracking validation: `tests/run_phase12_validate_full_tracking.py`
- Tracking perf: `tests/run_phase12_perf_tracking_full.py`
- Phase-13 breakdown (no tracking): `tests/run_phase13_perf_gpu_full_breakdown.py`
- ArUco sweep: `tests/run_phase13_aruco_sweep.py`

## Important Outputs
- Baseline GPU full: `outputs/phase12_baseline_gpu_full_report.txt` (mean_total≈12.63 ms, ~79 FPS)
- Tracking perf: `outputs/phase12_tracking_perf_full_report.txt` (mean_total≈8.80 ms, ~114 FPS, detect 40 / track 160)
- Phase-13 breakdown: `outputs/phase13_gpu_full_breakdown_report.txt` (mean_total≈12.35 ms after tuned params; corners ~9.33 ms)
- ArUco sweep summary: `outputs/phase13_aruco_sweep_summary.txt` (best_valid_profile=fast2, mean_total≈10.72 ms, ~93 FPS in sweep)

## Config Highlights
- `configs/default.json`:
  - `input.scale_factor = 1.0`, `input.use_roi = false`
  - Tracking block present; disable for pure GPU_FULL tests.
  - Pose: homography-based GPU pose active.
  - ArUco params (fast2) set as defaults via `aruco_params` in `gpu/corners.py`.

## How to Run (full video, 200 frames)
- Pure GPU baseline: `python3 tests/run_phase12_baseline_gpu_full.py`
- Tracking validation: `python3 tests/run_phase12_validate_full_tracking.py`
- Tracking perf: `python3 tests/run_phase12_perf_tracking_full.py`
- Phase-13 breakdown: `python3 tests/run_phase13_perf_gpu_full_breakdown.py`
- ArUco sweep: `python3 tests/run_phase13_aruco_sweep.py`

## Notes
- One full-frame H2D upload per frame; no full-frame D2H; syncs kept minimal.
- Tracking can be sped up further by increasing `tracking.detect_every_n` if FPS needs to approach 120+ on this asset.

