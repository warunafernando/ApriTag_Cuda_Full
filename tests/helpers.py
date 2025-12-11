from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def save_overlay(frame: np.ndarray, corners: np.ndarray | None, tag_id: int | None, path: Path) -> None:
    """
    Draw detection overlay and save.
    """
    img = frame.copy()
    if corners is not None:
        pts = corners.reshape(-1, 1, 2).astype(int)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    if tag_id is not None:
        cv2.putText(
            img,
            f"ID {tag_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def rms(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sqrt(np.mean(diff * diff)))

