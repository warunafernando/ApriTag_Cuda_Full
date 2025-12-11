# GPU-Accelerated AprilTag Detector

A high-performance, GPU-accelerated AprilTag detection pipeline implemented in Python using CuPy for GPU computing and OpenCV for computer vision operations. This project provides both CPU reference and GPU-accelerated implementations with comprehensive validation and performance benchmarking.

## Features

- **GPU-Accelerated Pipeline**: Full GPU implementation of AprilTag detection, including:
  - GPU-accelerated corner detection (ArUco-based)
  - GPU-accelerated homography-based sampling
  - GPU-accelerated bit extraction and codebook lookup
  - GPU-accelerated pose estimation
- **CPU Reference Path**: OpenCV ArUco-based reference implementation for validation
- **Tag Tracking**: KLT optical flow-based tracking for improved performance (Phase 11)
- **Performance Optimizations**: ArUco parameter tuning and overhead reduction (Phase 13)
- **Comprehensive Testing**: Smoke tests (5 frames) and validation tests (200 frames) for each phase
- **JSON Configuration**: Centralized configuration system for easy parameter tuning

## Performance

- **Pure GPU Baseline** (no tracking): ~79-81 FPS (mean_total_ms ≈ 12.35-12.63 ms)
- **GPU + Tracking**: ~114 FPS (mean_total_ms ≈ 8.80 ms)
- **Accuracy**: 100% ID match rate, 0 Hamming distance, corner RMS < 0.6 px, pose errors within tolerances

## Project Structure

```
.
├── common/          # Shared utilities (config, geometry, pose, tracking, video)
├── cpu/             # CPU reference implementation
│   ├── detector.py  # OpenCV ArUco detection
│   ├── decode.py    # CPU decode pipeline
│   └── pipeline.py  # CPU pipeline wrapper
├── gpu/             # GPU-accelerated implementation
│   ├── corners.py   # GPU corner detection (ArUco-based, optimized)
│   ├── decode.py    # GPU bit extraction and codebook lookup
│   └── sampling.py  # GPU homography-based sampling
├── configs/         # Configuration files
│   └── default.json # Main configuration file
├── tests/           # Test scripts for validation and benchmarking
├── inputs/          # Input video files
├── outputs/         # Generated reports and CSVs
└── docs/            # Documentation

```

## Requirements

- Python 3.8+
- NumPy 1.24.4 (compatible with CuPy)
- CuPy (for GPU acceleration)
- OpenCV 4.x (with ArUco support)
- SciPy

Install dependencies:

```bash
pip install numpy==1.24.4 opencv-python scipy cupy-cuda11x  # Adjust cupy-cuda11x for your CUDA version
```

## Configuration

Edit `configs/default.json` to configure:

- **Input**: Video path, scale factor, ROI settings
- **Tag**: Family (tag36h11), expected ID, size
- **Sampling**: Warp size, cells, border cells
- **Decode**: Maximum Hamming distance
- **GPU**: Tile sizes, corner detection parameters
- **Pose**: Tag size, IPPE usage, GPU pose settings
- **Tracking**: Enable/disable, detect frequency, optical flow thresholds

## Usage

### Basic CPU Pipeline

```bash
python3 tests/run_smoke_5_cpu_only.py
python3 tests/run_validate_200_cpu_only.py
```

### GPU Pipeline Tests

**Baseline GPU Full (no tracking):**
```bash
python3 tests/run_phase12_baseline_gpu_full.py
```

**GPU + Tracking:**
```bash
# Functional validation
python3 tests/run_phase12_validate_full_tracking.py

# Performance measurement
python3 tests/run_phase12_perf_tracking_full.py
```

**Phase 13 - Performance Breakdown:**
```bash
python3 tests/run_phase13_perf_gpu_full_breakdown.py
```

**ArUco Parameter Sweep:**
```bash
python3 tests/run_phase13_aruco_sweep.py
```

### Unified Validation Script

The main validation script supports multiple modes:

```bash
python3 tests/run_validate_200.py --mode GPU_FULL_GPU_CORNERS --frames FULL
```

Available modes:
- `CPU_ONLY`: CPU reference pipeline
- `GPU_DECODE_WITH_CPU_CORNERS`: GPU decode with CPU corners
- `GPU_FULL_GPU_CORNERS`: Full GPU pipeline
- `GPU_FULL_GPU_CORNERS_PERF`: Full GPU with performance timing

Frame options: `SMOKE` (5 frames), `VALIDATE` (200 frames), `FULL` (entire video)

## Implementation Phases

The project was developed in phases:

- **Phases 0-3**: Infrastructure, CPU pipeline, config system
- **Phase 4**: GPU corner detection
- **Phase 5**: GPU sampling and decode
- **Phase 6**: GPU pose estimation
- **Phase 7-9**: Performance benchmarking
- **Phase 10**: GPU corner extractor optimization
- **Phase 11**: Tag tracking with KLT optical flow
- **Phase 12**: Baseline and tracking validation
- **Phase 13**: Performance tuning and ArUco parameter optimization

See `docs/phase13_notes.md` for detailed phase summaries.

## Output Files

Test scripts generate:

- **CSV files**: Per-frame metrics (IDs, Hamming distances, corner RMS, pose errors, timings)
- **Report files**: Summary statistics (mean, median, p90, p99, max, FPS)
- **Debug overlays**: Visualizations of detected tags (when enabled)

Outputs are written to `outputs/` directory.

## Key Components

### CPU Detector (`cpu/detector.py`)
- Uses OpenCV ArUco detector for AprilTag 36h11
- Subpixel corner refinement
- Canonical corner ordering (TL, TR, BR, BL)

### GPU Sampling (`gpu/sampling.py`)
- CuPy-based homography computation
- Bilinear interpolation for intensity grid extraction
- Exact parity with CPU sampling when `force_cpu_exact_sampling=true`

### GPU Decode (`gpu/decode.py`)
- GPU-accelerated bit thresholding
- Hamming distance calculation against codebook
- Returns best matching tag ID and Hamming distance

### GPU Corners (`gpu/corners.py`)
- Optimized ArUco-based corner detection
- Tuned parameters for performance (Phase 13)
- Fast path with minimal overhead

### Pose Estimation (`common/pose.py`)
- CPU: OpenCV `solvePnP` with IPPE
- GPU: Homography-based decomposition with refinement
- Rotation and translation error metrics

### Tracking (`common/tracking.py`)
- KLT optical flow for corner tracking
- Detect+Track state machine
- Automatic fallback to full detection on tracking failure

## Validation Criteria

- **ID Match Rate**: Must be 1.0 (100% match with CPU reference)
- **Hamming Distance**: Must be 0 (exact match)
- **Corner RMS**: Mean < 0.6 px, Max < 1.5 px
- **Pose Errors**: Rotation < 1.5° (mean), Translation < 0.015 m (mean)

## License

This project is provided as-is for research and development purposes.

## Contributing

This is a research/development project. For issues or improvements, please open an issue or submit a pull request.

## Acknowledgments

- OpenCV for ArUco/AprilTag support
- CuPy for GPU array computing
- AprilTag library for tag design and codebook

