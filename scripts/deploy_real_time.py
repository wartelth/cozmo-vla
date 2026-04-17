#!/usr/bin/env python
"""Run a fine-tuned SmolVLA policy on Cozmo via PyCozmo (host-side inference)."""

from __future__ import annotations

# Cozmo's Wi-Fi AP has no internet, so default HuggingFace to offline mode. Users
# with a dual-NIC setup can override with HF_HUB_OFFLINE=0 / TRANSFORMERS_OFFLINE=0.
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from cozmo_vla.datasets.build_features import IMAGE_KEY
from cozmo_vla.robot.pycozmo_client import PyCozmoRobot, run_with_robot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deploy")

try:
    import einops
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION
except ImportError as e:
    einops = None  # type: ignore[assignment]
    _LEROBOT_ERR = e
else:
    _LEROBOT_ERR = None


def _ensure_lerobot():
    if _LEROBOT_ERR is not None:
        raise ImportError(
            "LeRobot must be installed to run deployment. See README.md for editable install."
        ) from _LEROBOT_ERR


def _build_raw_observation(
    rgb_uint8: np.ndarray,
    state: np.ndarray,
    task: str,
    image_hwc: tuple[int, int, int],
) -> dict:
    """Build a policy observation dict (batch size 1) matching dataset keys."""
    h, w, _ = image_hwc
    if rgb_uint8.shape[0] != h or rgb_uint8.shape[1] != w:
        rgb_uint8 = cv2.resize(rgb_uint8, (w, h), interpolation=cv2.INTER_AREA)
    img = torch.from_numpy(rgb_uint8).unsqueeze(0).float() / 255.0
    img = einops.rearrange(img, "b h w c -> b c h w")
    st = torch.from_numpy(state).unsqueeze(0).float()
    return {
        "observation.images.front": img,
        "observation.state": st,
        "task": [task],
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--policy.path",
        dest="policy_path",
        type=str,
        required=True,
        help="Fine-tuned policy repo id or local directory (HF LeRobot format)",
    )
    p.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="Same dataset repo used for training (loads normalization stats)",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Optional local dataset root to avoid downloading from Hub",
    )
    p.add_argument("--task", type=str, default="perform the trained task", help="Language instruction")
    p.add_argument("--hz", type=float, default=20.0, help="Control loop rate")
    p.add_argument("--max-seconds", type=float, default=120.0, help="Stop after this many seconds")
    p.add_argument(
        "--async-inference",
        action="store_true",
        help="Reserved: SmolVLA select_action already queues action chunks internally",
    )
    return p.parse_args()


def run_policy(robot: PyCozmoRobot, args: argparse.Namespace) -> None:
    _ensure_lerobot()

    if args.async_inference:
        logger.info(
            "SmolVLA caches action chunks in select_action; extra async overlap is optional for future tuning."
        )

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.eval()

    ds = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        download_videos=False,
    )
    stats = ds.meta.stats
    image_hwc = tuple(int(x) for x in ds.meta.features[IMAGE_KEY]["shape"])

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=str(args.policy_path),
        dataset_stats=stats,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
        },
    )

    policy.reset()
    dt = 1.0 / args.hz
    t_end = time.time() + args.max_seconds

    while time.time() < t_end:
        t0 = time.perf_counter()
        rgb = robot.get_rgb_uint8()
        state = robot.get_state_vector()
        raw = _build_raw_observation(rgb, state, args.task, image_hwc)
        with torch.inference_mode():
            batch = preprocessor(raw)
            action = policy.select_action(batch)
            action = postprocessor(action)
        if isinstance(action, dict):
            act_tensor = action[ACTION]
        elif isinstance(action, torch.Tensor):
            act_tensor = action
        else:
            act_tensor = getattr(action, ACTION, action)
        act = act_tensor.to("cpu").numpy()
        if act.ndim == 2:
            act = act[0]
        act4 = act[:4].astype(np.float32)
        robot.apply_action_normalized(act4, stop_if_unsafe=True)
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, dt - elapsed))

    robot.stop()


def main() -> None:
    args = _parse_args()

    def body(robot: PyCozmoRobot) -> None:
        run_policy(robot, args)

    run_with_robot(body)


if __name__ == "__main__":
    main()
