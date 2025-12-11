from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_validate_200 as runner


def main() -> None:
    sys.argv = [sys.argv[0], "--mode", "GPU_FULL_GPU_CORNERS_PERF", "--frames", "VALIDATE"]
    runner.main()


if __name__ == "__main__":
    main()

