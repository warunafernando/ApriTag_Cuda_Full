# CUDA Acceleration Plan

## Current Status

**Using CUDA (via CuPy):**
- ✅ GPU sampling (homography + bilinear interpolation): ~0.53 ms
- ✅ GPU decode (thresholding + codebook lookup): ~2.12 ms

**Still on CPU (bottlenecks):**
- ❌ Corner detection: **9.33 ms (75% of total time!)** - OpenCV ArUco on CPU
- ❌ Preprocessing: Grayscale conversion on CPU
- ❌ Pose: Homography decomposition on CPU (~0.37 ms, minor)

## Performance Breakdown (Phase 13)

```
mean_corners_ms:    9.33 ms  (75.6%)  ← BIGGEST BOTTLENECK
mean_decode_ms:     2.12 ms  (17.2%)
mean_sampling_ms:   0.53 ms  (4.3%)
mean_pose_ms:       0.37 ms  (3.0%)
mean_total_ms:     12.35 ms  (~81 FPS)
```

## Acceleration Opportunities

### 1. GPU Corner Detection (HIGHEST PRIORITY)
**Current:** OpenCV ArUco on CPU (~9.33 ms)
**Target:** CUDA-based quad detection (~1-2 ms estimated)

**Implementation Strategy:**
- Use CuPy for GPU-accelerated image processing:
  - Adaptive thresholding on GPU
  - Canny edge detection on GPU (or contour detection)
  - Connected components on GPU
  - Polygon approximation on GPU
- Alternative: Use OpenCV CUDA modules (`cv2.cuda`) if available
- Keep ArUco as fallback for accuracy validation

**Expected Speedup:** 9.33 ms → ~1.5 ms = **6x faster**, total pipeline: 12.35 ms → ~4.5 ms = **~220 FPS**

### 2. GPU Preprocessing (MEDIUM PRIORITY)
**Current:** `cv2.cvtColor(BGR2GRAY)` on CPU
**Target:** CuPy-based grayscale conversion on GPU

**Implementation:**
```python
import cupy as cp
# BGR to grayscale: gray = 0.299*R + 0.587*G + 0.114*B
gray_gpu = cp.dot(frame_gpu[..., :3], cp.array([0.114, 0.587, 0.299]))
```

**Expected Speedup:** Eliminates H2D copy for grayscale, keeps data on GPU

### 3. GPU Pose Estimation (LOW PRIORITY)
**Current:** NumPy-based homography decomposition (~0.37 ms)
**Target:** CuPy-based matrix operations

**Expected Speedup:** 0.37 ms → ~0.1 ms (minor, but keeps data on GPU)

## Implementation Plan

### Phase 14: GPU Corner Detection
1. **GPU Preprocessing Module** (`gpu/preprocess.py`):
   - GPU grayscale conversion
   - GPU adaptive thresholding
   - GPU edge detection (Canny or similar)

2. **GPU Corner Detection** (`gpu/corners_cuda.py`):
   - GPU-based contour detection
   - GPU-based quad formation
   - Fallback to CPU ArUco for validation

3. **Integration**:
   - Update `gpu/corners.py` to use GPU path
   - Keep CPU path as fallback
   - Validate accuracy (corner RMS, ID match)

### Phase 15: Full GPU Pipeline
1. Keep all data on GPU (no H2D/D2H except final results)
2. Pipeline: GPU gray → GPU corners → GPU sampling → GPU decode → GPU pose
3. Single H2D copy per frame (input), single D2H copy (corners/pose)

## Expected Final Performance

**Current (Phase 13):**
- Total: 12.35 ms (~81 FPS)
- Corners: 9.33 ms (CPU)

**After GPU Corner Detection:**
- Total: ~4.5 ms (~220 FPS)
- Corners: ~1.5 ms (GPU)
- **Speedup: 2.7x overall**

**With Tracking (Phase 11):**
- Detect frames: ~4.5 ms
- Track frames: ~1.0 ms (KLT on GPU)
- Effective: ~180-200 FPS (depending on detect_every_n)

## Technical Notes

### CuPy vs OpenCV CUDA
- **CuPy**: Pure Python/CUDA, full control, requires manual kernel implementation
- **OpenCV CUDA**: Pre-built functions, easier integration, but may have overhead

### Recommended Approach
1. Start with CuPy for preprocessing (grayscale, threshold)
2. Use OpenCV CUDA for edge detection if available
3. Implement custom CUDA kernels for quad detection if needed
4. Keep CPU ArUco as accuracy reference

### Validation Requirements
- Corner RMS < 0.6 px (mean), < 1.5 px (max)
- ID match rate = 1.0
- Hamming distance = 0
- Pose errors within Phase-8 tolerances

