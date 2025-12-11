"""
Phase 14D_4: Test that GPU mode does not silently failover when allow_failover=false.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from common.config import load_config
from common.corner_refine_dispatch import dispatch_corner_refine


def test_no_failover_on_error():
    """Test that GPU mode raises error instead of failing over."""
    cfg = load_config()
    cfg["corner_refine"]["mode"] = "GPU"
    cfg["corner_refine"]["allow_failover"] = False

    # Create invalid input to trigger error
    gray = np.zeros((100, 100), dtype=np.uint8)
    corners_in = np.array([[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]], dtype=np.float32)

    # This should work normally
    try:
        corners_refined, timings = dispatch_corner_refine(
            gray, corners_in, cfg, image_shape=(100, 100), num_tags=1
        )
        print("✅ Normal case: GPU refinement succeeded")
    except Exception as e:
        print(f"❌ Normal case failed unexpectedly: {e}")
        return False

    # Test with invalid shape (should raise error, not failover)
    try:
        invalid_corners = np.array([[[10.0, 10.0]]], dtype=np.float32)  # Wrong shape
        corners_refined, timings = dispatch_corner_refine(
            gray, invalid_corners, cfg, image_shape=(100, 100), num_tags=1
        )
        print("❌ Invalid input: Should have raised error but didn't")
        return False
    except Exception as e:
        print(f"✅ Invalid input: Correctly raised error: {type(e).__name__}")
        return True


def test_failover_when_enabled():
    """Test that failover works when allow_failover=true."""
    cfg = load_config()
    cfg["corner_refine"]["mode"] = "GPU"
    cfg["corner_refine"]["allow_failover"] = True
    cfg["corner_refine"]["fallback_mode"] = "CPU"

    gray = np.zeros((100, 100), dtype=np.uint8)
    corners_in = np.array([[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]], dtype=np.float32)

    # This should work (either GPU or CPU fallback)
    try:
        corners_refined, timings = dispatch_corner_refine(
            gray, corners_in, cfg, image_shape=(100, 100), num_tags=1
        )
        print("✅ Failover enabled: Refinement succeeded (may have used CPU fallback)")
        return True
    except Exception as e:
        print(f"❌ Failover enabled: Unexpected error: {e}")
        return False


def main() -> None:
    print("Phase 14D_4 No-Failover Test")
    print("=" * 70)
    print()

    test1_passed = test_no_failover_on_error()
    print()
    test2_passed = test_failover_when_enabled()
    print()

    if test1_passed and test2_passed:
        print("Overall: PASS")
    else:
        print("Overall: FAIL")


if __name__ == "__main__":
    main()

