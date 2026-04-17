#!/usr/bin/env python
"""Run a fine-tuned SmolVLA policy on Cozmo via PyCozmo (host-side inference)."""

from __future__ import annotations

# Cozmo's Wi-Fi AP has no internet, so default HuggingFace to offline mode. Users
# with a dual-NIC setup can override with HF_HUB_OFFLINE=0 / TRANSFORMERS_OFFLINE=0.
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import logging
import socket
import sys
import time
from datetime import datetime
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
    p.add_argument(
        "--record-dir",
        type=str,
        default=None,
        help=(
            "If set, dump per-step telemetry (jsonl + jpeg frames) to this directory "
            "for later inspection via scripts/view_rollout.py."
        ),
    )
    p.add_argument(
        "--record-every",
        type=int,
        default=1,
        help="Save a jpeg every N steps (state/action are always logged).",
    )
    return p.parse_args()


class RolloutRecorder:
    """Writes per-step telemetry for offline inspection (Layer 3 debug tool)."""

    def __init__(self, out_dir: Path, run_meta: dict, record_every: int = 1):
        self.dir = out_dir
        self.frames_dir = out_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.record_every = max(1, int(record_every))
        (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        self._jsonl = open(out_dir / "telemetry.jsonl", "w", encoding="utf-8")
        self._step = 0
        self._t0 = time.time()

    def log(
        self,
        *,
        rgb_uint8: np.ndarray,
        state: np.ndarray,
        applied_action: np.ndarray,
        raw_action: np.ndarray,
        unsafe: bool,
        inference_s: float,
        step_s: float,
    ) -> None:
        ts = time.time() - self._t0
        frame_rel = None
        if self._step % self.record_every == 0:
            fname = f"frame_{self._step:06d}.jpg"
            bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(self.frames_dir / fname), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            frame_rel = f"frames/{fname}"

        rec = {
            "step": self._step,
            "t": round(ts, 4),
            "frame": frame_rel,
            "state": [float(x) for x in state.tolist()],
            "applied_action": [float(x) for x in applied_action.tolist()],
            "raw_action": [float(x) for x in raw_action.tolist()],
            "unsafe": bool(unsafe),
            "inference_s": round(inference_s, 4),
            "step_s": round(step_s, 4),
        }
        self._jsonl.write(json.dumps(rec) + "\n")
        self._jsonl.flush()
        self._step += 1

    def close(self) -> None:
        if not self._jsonl.closed:
            self._jsonl.close()


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

    recorder: RolloutRecorder | None = None
    if args.record_dir:
        run_dir = Path(args.record_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            run_dir = run_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        run_meta = {
            "policy_path": str(args.policy_path),
            "dataset_repo_id": args.dataset_repo_id,
            "task": args.task,
            "hz": args.hz,
            "image_hwc": list(image_hwc),
            "hostname": socket.gethostname(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        recorder = RolloutRecorder(run_dir, run_meta, record_every=args.record_every)
        logger.info("Recording telemetry to %s", run_dir)

    try:
        while time.time() < t_end:
            t0 = time.perf_counter()
            rgb = robot.get_rgb_uint8()
            state = robot.get_state_vector()
            raw = _build_raw_observation(rgb, state, args.task, image_hwc)
            t_inf = time.perf_counter()
            with torch.inference_mode():
                batch = preprocessor(raw)
                action = policy.select_action(batch)
                action = postprocessor(action)
            inference_s = time.perf_counter() - t_inf
            if isinstance(action, dict):
                act_tensor = action[ACTION]
            elif isinstance(action, torch.Tensor):
                act_tensor = action
            else:
                act_tensor = getattr(action, ACTION, action)
            act = act_tensor.to("cpu").numpy()
            if act.ndim == 2:
                act = act[0]
            raw4 = act[:4].astype(np.float32)
            act4 = np.clip(raw4, -1.0, 1.0).astype(np.float32)
            unsafe = robot.is_unsafe()
            robot.apply_action_normalized(act4, stop_if_unsafe=True)
            step_s = time.perf_counter() - t0
            if recorder is not None:
                recorder.log(
                    rgb_uint8=rgb,
                    state=state,
                    applied_action=act4,
                    raw_action=raw4,
                    unsafe=unsafe,
                    inference_s=inference_s,
                    step_s=step_s,
                )
            time.sleep(max(0.0, dt - step_s))
    finally:
        if recorder is not None:
            recorder.close()
        robot.stop()


def main() -> None:
    args = _parse_args()

    def body(robot: PyCozmoRobot) -> None:
        run_policy(robot, args)

    run_with_robot(body)


if __name__ == "__main__":
    main()
