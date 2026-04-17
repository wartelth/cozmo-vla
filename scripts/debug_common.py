#!/usr/bin/env python
"""Shared helpers for offline debugging of the Cozmo SmolVLA policy.

Keeps policy loading, observation building, and per-step inference in one place
so `replay_offline.py` / `prompt_ablation.py` / `view_rollout.py` all stay short.
"""

from __future__ import annotations

import os

# Cozmo's Wi-Fi AP has no internet; default HF to offline. Override with
# HF_HUB_OFFLINE=0 / TRANSFORMERS_OFFLINE=0 on a machine with real internet.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from cozmo_vla.datasets.build_features import IMAGE_KEY  # noqa: E402

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402


@dataclass
class DebugContext:
    policy: SmolVLAPolicy
    preprocessor: Any
    postprocessor: Any
    dataset: LeRobotDataset
    image_hwc: tuple[int, int, int]
    action_names: list[str]
    state_names: list[str]


def load_context(
    policy_path: str | Path,
    dataset_repo_id: str,
    dataset_root: str | Path | None = None,
) -> DebugContext:
    """Load policy + processors + dataset in eval mode. Safe to call once per run."""
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy = SmolVLAPolicy.from_pretrained(str(policy_path))
    policy.eval()

    ds = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=str(dataset_root) if dataset_root else None,
        download_videos=False,
    )

    pre, post = make_pre_post_processors(
        policy_cfg,
        pretrained_path=str(policy_path),
        dataset_stats=ds.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
        },
    )

    image_hwc = tuple(int(x) for x in ds.meta.features[IMAGE_KEY]["shape"])
    action_names = list(ds.meta.features["action"].get("names") or [])
    state_names = list(ds.meta.features["observation.state"].get("names") or [])

    return DebugContext(
        policy=policy,
        preprocessor=pre,
        postprocessor=post,
        dataset=ds,
        image_hwc=image_hwc,
        action_names=action_names,
        state_names=state_names,
    )


def frame_to_obs(frame: dict, task: str, image_key: str = IMAGE_KEY) -> dict:
    """Convert a LeRobotDataset frame into a policy observation with batch dim.

    Dataset-decoded images are already (C, H, W) float32 in [0, 1], so we just add
    a batch dim and wrap in the expected keys.
    """
    img = frame[image_key]
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if img.ndim == 3:
        img = img.unsqueeze(0)

    st = frame["observation.state"]
    if isinstance(st, np.ndarray):
        st = torch.from_numpy(st)
    if st.ndim == 1:
        st = st.unsqueeze(0)

    return {
        image_key: img.float(),
        "observation.state": st.float(),
        "task": [task],
    }


def predict_action(ctx: DebugContext, raw_obs: dict) -> np.ndarray:
    """Run one policy step and return a flat numpy action vector."""
    with torch.inference_mode():
        batch = ctx.preprocessor(raw_obs)
        action = ctx.policy.select_action(batch)
        action = ctx.postprocessor(action)
    if isinstance(action, dict):
        act_tensor = action[ACTION]
    elif isinstance(action, torch.Tensor):
        act_tensor = action
    else:
        act_tensor = getattr(action, ACTION, action)
    act = act_tensor.to("cpu").numpy()
    if act.ndim == 2:
        act = act[0]
    return act.astype(np.float32)


def reset_policy(ctx: DebugContext) -> None:
    """Clear any cached action chunk inside SmolVLA before a new rollout."""
    ctx.policy.reset()


def episode_bounds(ctx: DebugContext, episode_index: int) -> tuple[int, int, int]:
    """Return (from_idx, to_idx, length) for one episode of the local dataset."""
    episodes = ctx.dataset.meta.episodes
    if episodes is None:
        raise RuntimeError("Dataset has no episodes metadata loaded.")

    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(
            f"episode_index={episode_index} out of range [0, {len(episodes)})"
        )

    row = episodes[int(episode_index)]
    from_idx = int(row["dataset_from_index"])
    to_idx = int(row["dataset_to_index"])
    length = int(row.get("length", to_idx - from_idx))
    return from_idx, to_idx, length
