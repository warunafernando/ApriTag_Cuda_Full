# GPU-Accelerated AprilTag Detector - Complete Project Summary

## Project Overview

This repository implements a **high-performance, GPU-accelerated AprilTag detection pipeline** using Python, CuPy (for GPU computing), and OpenCV. The project was developed incrementally through 14+ phases, with each phase adding new functionality and optimizations while maintaining strict validation against CPU reference implementations.

**Key Achievement**: Achieved **~114 FPS** with tracking enabled (from ~69 FPS baseline) while maintaining **100% accuracy** (ID match rate = 1.0, Hamming distance = 0).

---

## Implementation Phases

### **Phases 0-3: Foundation & Infrastructure**
- **Phase 0**: Project structure, config system, CPU reference pipeline
- **Phase 1**: GPU-accelerated sampling (homography-based bilinear interpolation)
- **Phase 2**: GPU-accelerated decode (bit extraction + codebook lookup)
- **Phase 3**: Integration and initial validation

**Key Components:**
- JSON-based configuration system (`configs/default.json`)
- CPU reference pipeline using OpenCV ArUco
- GPU sampling with exact CPU parity (`force_cpu_exact_sampling`)
- GPU decode with Hamming distance calculation

**Validation**: Achieved exact parity (RMS=0) for sampling and decode stages.

---

### **Phase 4: GPU Corner Detection**
- Implemented GPU corner extraction using OpenCV ArUco (hybrid approach)
- Canonical corner ordering (TL, TR, BR, BL)
- Initial validation against CPU reference

**Files**: `gpu/corners.py`, `tests/run_smoke_5_gpu_corners.py`

---

### **Phase 5: Full GPU Pipeline Integration**
- Integrated GPU corners, sampling, and decode into full pipeline
- Validation of end-to-end GPU path
- ID and Hamming distance matching

**Result**: 100% ID match rate, 0 Hamming distance across all test frames.

---

### **Phase 6: GPU Pose Estimation**
- Implemented GPU pose solver using homography decomposition
- Added `solve_pose_gpu_from_homography` with `solvePnP` refinement
- Rotation and translation error metrics

**Accuracy**: Mean rotation error ~0.07°, mean translation error ~0.001 m

**Files**: `common/pose.py`, `tests/run_phase6_smoke_5.py`, `tests/run_phase6_validate_200.py`

---

### **Phases 7-9: Performance Benchmarking**
- **Phase 7**: Per-stage timing (CPU and GPU) with quantile statistics
- **Phase 8**: Full video performance analysis
- **Phase 9**: Performance drift analysis

**Key Metrics:**
- CPU-only: ~328 FPS (decode only), ~69 FPS (full pipeline)
- GPU decode with CPU corners: ~328 FPS
- GPU full: ~69 FPS (baseline)

**Files**: 
- `tests/run_phase7_perf_*.py`
- `tests/run_phase9_perf_*.py`
- `tests/run_validate_200.py` (unified validation script)

---

### **Phase 10: GPU Corner Extractor Optimization**
- Optimized GPU corner detection to rely solely on fast ArUco path
- Removed slower contour-based fallback
- Reduced `t_gpu_corners_ms` significantly

**Result**: Improved GPU_FULL performance while maintaining accuracy.

**Files**: `gpu/corners.py` (optimized), `tests/run_phase10_*.py`

---

### **Phase 11: Tag Tracking with KLT Optical Flow**
- Implemented Detect+Track hybrid pipeline
- KLT optical flow for corner tracking between frames
- Automatic fallback to full detection on tracking failure
- State machine for DETECT/TRACK mode switching

**Performance Impact**: 
- Baseline (no tracking): ~79 FPS
- With tracking: ~114 FPS (44% speedup)

**Files**: 
- `common/tracking.py` (TagTrackerState, track_corners_klt)
- `tests/run_phase11_*.py`

---

### **Phase 12: Baseline & Tracking Validation**
- Established pure GPU baseline (tracking disabled)
- Validated GPU + Tracking pipeline
- Full functional and performance validation

**Results**:
- Baseline: mean_total ≈ 12.63 ms (~79 FPS)
- Tracking: mean_total ≈ 8.80 ms (~114 FPS)
- Accuracy: 100% maintained with tracking

**Files**: `tests/run_phase12_*.py`

---

### **Phase 13: Performance Tuning & ArUco Optimization**
- Fine-grained per-stage timing breakdown
- ArUco parameter sweep (multiple profiles tested)
- Adopted "fast2" parameter profile for optimal performance
- Reduced overhead (single H2D copy, minimal syncs)

**Results**:
- Mean total: ~12.35 ms (~81 FPS) after tuning
- ArUco sweep best: ~10.72 ms (~93 FPS) in sweep conditions

**Files**: 
- `tests/run_phase13_perf_gpu_full_breakdown.py`
- `tests/run_phase13_aruco_sweep.py`
- `gpu/corners.py` (tuned ArUco parameters)

---

### **Phase 14A: GPU Adaptive Thresholding**
- Implemented CPU reference using OpenCV `adaptiveThreshold`
- Implemented GPU version using CuPy box filter (cumulative sums)
- Fixed thresholding logic to match OpenCV (`>=` instead of `>`)

**Status**: Implemented and tested, accuracy and performance tuning ongoing.

**Files**: 
- `cpu/adaptive_threshold.py`
- `gpu/adaptive_threshold.py`
- `tests/run_phase14A_*.py`

---

### **Phase 14B: GPU Canny Edge Detection**
- Implemented CPU reference using OpenCV `Canny`
- Implemented GPU version with:
  - Gaussian blur (vectorized)
  - Sobel gradients (vectorized)
  - Non-maximum suppression (vectorized)
  - Hysteresis thresholding (using binary dilation)

**Status**: Implemented and tested, vectorized to remove Python loops.

**Files**: 
- `cpu/edges.py`
- `gpu/edges.py`
- `tests/run_phase14B_*.py`

---

### **Phase 14C: GPU Quad Candidate Extraction**
- Implemented CPU reference using OpenCV `findContours` + `approxPolyDP`
- Implemented GPU version using hybrid approach (CPU findContours + GPU filtering)
- Quad filtering: area, aspect ratio, rectangularity checks

**Result**: Perfect accuracy (RMS=0.0) with CPU reference, exact quad matches.

**Files**: 
- `cpu/quads.py`
- `gpu/quads.py`
- `tests/run_phase14C_*.py`

---

### **Phase 14D: GPU Corner Refinement (Subpixel)**
- Implemented CPU reference using OpenCV `cornerSubPix`
- Implemented GPU version using gradient-based iterative refinement
- Lucas-Kanade-like approach with structure tensor

**Status**: Implemented and tested, algorithm tuning needed for better accuracy.

**Files**: 
- `cpu/corner_refine.py`
- `gpu/corner_refine.py`
- `tests/run_phase14D_*.py`

---

## Project Structure

```
Apriltag_Full_CUDA/
├── common/              # Shared utilities
│   ├── config.py        # JSON config loader
│   ├── geometry.py      # Corner canonicalization
│   ├── pose.py          # CPU/GPU pose estimation
│   ├── tracking.py       # KLT optical flow tracking
│   └── video.py         # Video frame loading
│
├── cpu/                 # CPU reference implementations
│   ├── detector.py      # OpenCV ArUco detection
│   ├── decode.py        # CPU decode pipeline
│   ├── pipeline.py      # CPU pipeline wrapper
│   ├── adaptive_threshold.py  # CPU adaptive threshold
│   ├── edges.py         # CPU Canny edges
│   ├── quads.py         # CPU quad extraction
│   └── corner_refine.py # CPU corner refinement
│
├── gpu/                 # GPU-accelerated implementations
│   ├── corners.py       # GPU corner detection (ArUco, optimized)
│   ├── sampling.py      # GPU homography-based sampling
│   ├── decode.py        # GPU bit extraction + codebook
│   ├── adaptive_threshold.py  # GPU adaptive threshold
│   ├── edges.py         # GPU Canny edges
│   ├── quads.py         # GPU quad extraction
│   └── corner_refine.py # GPU corner refinement
│
├── configs/
│   └── default.json     # Central configuration file
│
├── tests/               # Comprehensive test suite
│   ├── run_validate_200.py  # Unified validation script
│   ├── run_phase*_*.py  # Phase-specific tests
│   └── helpers.py       # Test utilities
│
├── inputs/
│   └── arducam_tag3.mp4 # Test video (1200p, Tag #3)
│
├── outputs/             # Generated reports and CSVs
│   └── (various phase reports)
│
└── docs/
    ├── README.md
    ├── phase13_notes.md
    ├── CUDA_ACCELERATION_PLAN.md
    └── PROJECT_SUMMARY.md (this file)
```

---

## Key Features & Components

### **1. CPU Reference Pipeline**
- **Detection**: OpenCV ArUco AprilTag 36h11 detector
- **Decode**: Homography-based warping, bilinear sampling, codebook lookup
- **Pose**: OpenCV `solvePnP` with IPPE
- **Accuracy**: 100% detection rate, 0 Hamming distance

### **2. GPU-Accelerated Pipeline**
- **Corners**: Optimized ArUco-based detection (tuned parameters)
- **Sampling**: CuPy-based homography + bilinear interpolation
- **Decode**: GPU bit thresholding + Hamming distance calculation
- **Pose**: Homography decomposition with `solvePnP` refinement
- **Performance**: ~79-81 FPS (baseline), ~114 FPS (with tracking)

### **3. Tag Tracking**
- **Algorithm**: KLT optical flow (Lucas-Kanade)
- **State Machine**: DETECT/TRACK with automatic fallback
- **Performance**: 44% speedup (114 FPS vs 79 FPS)
- **Accuracy**: 100% maintained

### **4. Configuration System**
- **Format**: JSON-based (`configs/default.json`)
- **Sections**: Input, tag, sampling, decode, GPU, pose, tracking, outputs
- **Override**: Supports local config overrides

### **5. Testing & Validation**
- **Smoke Tests**: 5-frame quick validation
- **Full Validation**: 200-frame comprehensive testing
- **Performance Tests**: Full video benchmarking
- **Metrics**: ID match, Hamming, corner RMS, pose errors, timings

---

## Performance Metrics

### **Baseline Performance (No Tracking)**
- **CPU-only**: ~69 FPS (mean_total ≈ 14.48 ms)
- **GPU decode (CPU corners)**: ~328 FPS (mean_total ≈ 3.05 ms)
- **GPU full**: ~79-81 FPS (mean_total ≈ 12.35-12.63 ms)

### **With Tracking**
- **GPU + Tracking**: ~114 FPS (mean_total ≈ 8.80 ms)
- **Speedup**: 44% improvement over baseline
- **Detection frequency**: Every 5 frames (configurable)

### **Per-Stage Timings (GPU Full, Phase 13)**
- `t_gpu_copy_in_ms`: ~0.5 ms
- `t_gpu_preprocess_ms`: ~0.1 ms
- `t_gpu_corners_ms`: ~9.33 ms (largest component)
- `t_gpu_sampling_ms`: ~0.8 ms
- `t_gpu_decode_ms`: ~0.3 ms
- `t_gpu_pose_ms`: ~1.2 ms
- `t_gpu_total_ms`: ~12.35 ms

### **Accuracy Metrics**
- **ID Match Rate**: 1.0000 (100%)
- **Hamming Distance**: 0 (exact match)
- **Corner RMS**: Mean < 0.6 px, Max < 1.5 px
- **Pose Errors**: 
  - Rotation: Mean ~0.07°, Max < 1.5°
  - Translation: Mean ~0.001 m, Max < 0.015 m

---

## Technical Highlights

### **1. Exact CPU Parity**
- GPU sampling achieves RMS=0.0 with CPU when `force_cpu_exact_sampling=true`
- GPU decode produces identical IDs and Hamming distances
- GPU pose matches CPU within tight tolerances

### **2. Performance Optimizations**
- Single H2D copy per frame (no full-frame D2H)
- Minimal GPU synchronizations
- ArUco parameter tuning (fast2 profile)
- Tracking reduces full detection frequency

### **3. Vectorization**
- GPU Canny edges: Fully vectorized (no Python loops)
- GPU adaptive threshold: CuPy cumulative sums
- GPU sampling: Vectorized bilinear interpolation

### **4. Robustness**
- Automatic fallback mechanisms (tracking → detection)
- Comprehensive error handling
- Validation at every phase

---

## Test Assets

- **Video**: `inputs/arducam_tag3.mp4`
  - Resolution: 1200p (1920x1200)
  - Tag: AprilTag 36h11, ID #3
  - Frame count: ~200+ frames
- **Test Sets**:
  - Smoke: 5 frames (frames 0-4)
  - Validation: 200 frames (frames 0-199)
  - Full: Entire video

---

## Configuration Highlights

Key configuration options in `configs/default.json`:

```json
{
  "input": {
    "video_path": "inputs/arducam_tag3.mp4",
    "scale_factor": 1.0,
    "use_roi": false
  },
  "tag": {
    "family": "tag36h11",
    "id_expected": 3,
    "size_meters": 0.16
  },
  "sampling": {
    "warp_size": 96,
    "cells": 6,
    "border_cells": 1
  },
  "gpu": {
    "enabled": true,
    "force_cpu_exact_sampling": true
  },
  "pose": {
    "tag_size_meters": 0.16,
    "use_ippe": true,
    "gpu": {
      "force_cpu_pose": false
    }
  },
  "tracking": {
    "enabled": true,
    "detect_every_n": 5,
    "max_optical_flow_error": 10.0
  }
}
```

---

## Dependencies

- **Python**: 3.8+
- **NumPy**: 1.24.4 (compatible with CuPy)
- **CuPy**: GPU array computing (CUDA 11.x)
- **OpenCV**: 4.x (with ArUco support)
- **SciPy**: For scientific computing

---

## Output Files

Test scripts generate comprehensive outputs:

- **CSV Files**: Per-frame metrics (IDs, Hamming, corners, pose, timings)
- **Report Files**: Summary statistics (mean, median, p90, p99, max, FPS)
- **Debug Overlays**: Visualizations of detected tags (when enabled)

All outputs written to `outputs/` directory with phase-specific naming.

---

## Current Status

### **Completed & Validated**
✅ Phases 0-13: Full GPU pipeline with tracking  
✅ Phase 14A: GPU adaptive thresholding (implemented)  
✅ Phase 14B: GPU Canny edge detection (implemented)  
✅ Phase 14C: GPU quad extraction (perfect accuracy)  
✅ Performance: ~114 FPS with tracking  
✅ Accuracy: 100% ID match, 0 Hamming distance  

### **In Progress / Needs Tuning**
⚠️ Phase 14A: Adaptive threshold accuracy/performance tuning  
⚠️ Phase 14B: Canny edge detection accuracy/performance tuning  
⚠️ Phase 14D: Corner refinement accuracy tuning  

---

## Key Achievements

1. **Performance**: Achieved 44% speedup with tracking (114 FPS vs 79 FPS)
2. **Accuracy**: Maintained 100% accuracy across all optimizations
3. **Modularity**: Clean separation of CPU/GPU paths, easy to switch
4. **Validation**: Comprehensive testing at every phase
5. **Documentation**: Detailed README, phase notes, and summaries

---

## Future Work

1. **Phase 14A/B Optimization**: Improve GPU adaptive threshold and Canny edge accuracy/performance
2. **Phase 14D Tuning**: Refine GPU corner refinement algorithm for better accuracy
3. **Pure GPU Pipeline**: Replace remaining CPU components (corners, preprocessing) with full GPU implementations
4. **Multi-Tag Support**: Extend to detect multiple tags per frame
5. **Real-time Integration**: Optimize for real-time camera input

---

## Repository Information

- **GitHub**: https://github.com/warunafernando/ApriTag_Cuda_Full
- **Main Branch**: `main`
- **License**: Research/development project

---

## Summary

This project successfully implements a GPU-accelerated AprilTag detector that achieves **~114 FPS** (with tracking) while maintaining **100% accuracy**. The implementation follows a rigorous phase-by-phase approach with validation at each step, resulting in a robust, well-tested, and well-documented system. The codebase is modular, configurable, and ready for further optimization and extension.

