import json
from pathlib import Path
from typing import Any, Dict


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base without mutating inputs."""
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(
    default_path: Path | str = Path("configs/default.json"),
    local_path: Path | str = Path("configs/local.json"),
) -> Dict[str, Any]:
    """
    Load default config and optionally merge local overrides.
    """
    default_path = Path(default_path)
    local_path = Path(local_path)

    with default_path.open("r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            local_cfg = json.load(f)
        return _deep_merge(base_cfg, local_cfg)

    return base_cfg


def ensure_output_dirs(cfg: Dict[str, Any]) -> None:
    """
    Create output directories referenced by the config if they do not exist.
    """
    outputs = cfg.get("outputs", {})
    paths = [
        outputs.get("root"),
        outputs.get("debug_dir"),
        Path(outputs.get("metrics_csv", "")).parent if outputs.get("metrics_csv") else None,
        Path(outputs.get("report_txt", "")).parent if outputs.get("report_txt") else None,
    ]
    for p in paths:
        if not p:
            continue
        Path(p).mkdir(parents=True, exist_ok=True)

