"""LeRobot v3 feature schema for Cozmo datasets (SmolVLA-compatible)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Defaults aligned with configs/cozmo_action_space.json
ACTION_DIM = 4
STATE_DIM = 8
IMAGE_KEY = "observation.images.front"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_action_space_config(path: Path | None = None) -> dict[str, Any]:
    p = path or _repo_root() / "configs" / "cozmo_action_space.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_cozmo_features(
    image_shape_hwc: tuple[int, int, int] | None = None,
    state_dim: int | None = None,
    action_dim: int | None = None,
    use_video: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Feature dict for `LeRobotDataset.create`.

    Visual features use dtype ``video`` when ``use_videos=True`` in ``create`` (default),
    or ``image`` if you disable video encoding.
    """
    cfg = load_action_space_config()
    h, w, c = image_shape_hwc or tuple(cfg["image_shape_hwc"])
    sd = state_dim if state_dim is not None else int(cfg["state_dim"])
    ad = action_dim if action_dim is not None else int(cfg["action_dim"])
    visual_dtype = "video" if use_video else "image"

    return {
        IMAGE_KEY: {
            "dtype": visual_dtype,
            "shape": (h, w, c),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (sd,),
            "names": cfg.get("state_names"),
        },
        "action": {
            "dtype": "float32",
            "shape": (ad,),
            "names": cfg.get("action_names"),
        },
    }
