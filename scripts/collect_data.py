#!/usr/bin/env python
"""Teleoperate Cozmo (PyCozmo) and record a LeRobot v3 dataset (SmolVLA-ready)."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure src/ on path when run as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from cozmo_vla.datasets.build_features import IMAGE_KEY, build_cozmo_features, load_action_space_config
from cozmo_vla.robot.pycozmo_client import PyCozmoRobot, run_with_robot
from cozmo_vla.teleop import TeleopProvider, make_teleop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("collect_data")

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    LeRobotDataset = None  # type: ignore[misc, assignment]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-id",
        type=str,
        default="local/cozmo_vla",
        help=(
            "Dataset id (folder name under HF_LEROBOT_HOME). Does NOT contact the Hub unless you pass "
            "--push. Use e.g. myuser/my_dataset only if you plan to upload; for local-only, keep the "
            "default or use local/my_run."
        ),
    )
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Exact folder to write the dataset (overrides default HF_LEROBOT_HOME/<repo-id>)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --root if it already exists and start a new dataset (only when you mean to wipe data).",
    )
    p.add_argument("--fps", type=int, default=None, help="Override FPS (default from config JSON)")
    p.add_argument(
        "--episode-time-s",
        type=float,
        default=45.0,
        help="Max seconds per episode (keyboard N / gamepad next button ends sooner)",
    )
    p.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        help="Video codec for dataset encoding (h264 is reliable on Windows; libsvtav1 if available)",
    )
    p.add_argument("--push", action="store_true", help="Upload dataset to Hugging Face Hub after finalize")
    p.add_argument("--private", action="store_true", help="Create private Hub dataset repo when pushing")
    p.add_argument(
        "--teleop",
        choices=("keyboard", "gamepad"),
        default="keyboard",
        help="keyboard: WASD+R/F+T/G | gamepad: left stick drive, right stick lift/head (Xbox layout)",
    )
    p.add_argument(
        "--joystick-index",
        type=int,
        default=0,
        help="SDL joystick index when --teleop gamepad (0 = first controller)",
    )
    p.add_argument(
        "--gamepad-deadzone",
        type=float,
        default=0.15,
        help="Stick deadzone for gamepad [0,1]",
    )
    p.add_argument(
        "--invert-forward",
        action="store_true",
        help="Flip left-stick forward/back if your controller drives the wrong way",
    )
    p.add_argument(
        "--gamepad-next-button",
        type=int,
        default=7,
        help="Gamepad button index to finish the current episode early (default 7 ≈ Start on many Xbox pads). See script stderr help.",
    )
    return p.parse_args()


def _is_lerobot_dataset_dir(path: Path) -> bool:
    return path.is_dir() and (path / "meta" / "info.json").is_file()


def _input_instruction() -> str:
    """Read stdin without pycozmo INFO lines breaking the same terminal line as the prompt."""
    pycoz = logging.getLogger("pycozmo")
    prev = pycoz.level
    pycoz.setLevel(logging.WARNING)
    try:
        sys.stderr.flush()
        sys.stdout.flush()
        return input("Instruction for next episode (empty to quit): ").strip()
    finally:
        pycoz.setLevel(prev)


def _open_dataset_for_recording(args: argparse.Namespace, fps: int, features: dict) -> object:
    assert LeRobotDataset is not None
    root = args.root
    if root is not None:
        root = root.resolve()
        if _is_lerobot_dataset_dir(root):
            if args.overwrite:
                shutil.rmtree(root)
                logger.info("Removed existing dataset at %s (--overwrite).", root)
                return LeRobotDataset.create(
                    repo_id=args.repo_id,
                    fps=fps,
                    root=root,
                    robot_type="cozmo",
                    features=features,
                    use_videos=True,
                    vcodec=args.vcodec,
                )
            logger.info("Resuming recording in existing dataset at %s", root)
            return LeRobotDataset.resume(repo_id=args.repo_id, root=root, vcodec=args.vcodec, batch_encoding_size=1)
        if root.exists():
            if args.overwrite:
                shutil.rmtree(root)
                logger.info("Removed path %s (--overwrite).", root)
            else:
                raise ValueError(
                    f"--root {root} exists but is not a LeRobot dataset (missing meta/info.json). "
                    "Remove the folder or pass --overwrite to replace it."
                )
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        root=root,
        robot_type="cozmo",
        features=features,
        use_videos=True,
        vcodec=args.vcodec,
    )


def _record_episode(
    robot: PyCozmoRobot,
    dataset: object,
    task: str,
    fps: int,
    episode_time_s: float,
    tele_op: TeleopProvider,
    target_hwc: tuple[int, int, int],
) -> None:
    dt = 1.0 / fps
    n_frames = int(episode_time_s * fps)
    for _ in range(n_frames):
        t0 = time.perf_counter()
        action = tele_op.action_vector()
        robot.apply_action_normalized(action, stop_if_unsafe=True)
        rgb = robot.get_rgb_uint8()
        th, tw, _ = target_hwc
        if rgb.shape[0] != th or rgb.shape[1] != tw:
            rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
        state = robot.get_state_vector()
        frame = {
            IMAGE_KEY: rgb,
            "observation.state": state,
            "action": action,
            "task": task,
        }
        dataset.add_frame(frame)
        if tele_op.consume_episode_done():
            logger.info("Episode ended early (saved); enter instruction for the next one.")
            break
        elapsed = time.perf_counter() - t0
        sleep_for = max(0.0, dt - elapsed)
        time.sleep(sleep_for)
    robot.stop()
    dataset.save_episode()


def main_collect(robot: PyCozmoRobot, args: argparse.Namespace, tele_op: TeleopProvider) -> None:
    if LeRobotDataset is None:
        raise ImportError(
            "lerobot is not installed. Clone https://github.com/huggingface/lerobot and run "
            "`pip install -e \".[smolvla]\"` in that repo, then re-run this script."
        )

    cfg = load_action_space_config()
    fps = args.fps or int(cfg["fps"])
    target_hwc = tuple(int(x) for x in cfg["image_shape_hwc"])
    features = build_cozmo_features(image_shape_hwc=target_hwc, use_video=True)

    dataset = _open_dataset_for_recording(args, fps, features)

    if args.teleop == "keyboard":
        print(
            "Teleop: W/S drive, A/D turn, R/F lift, T/G head. "
            "During recording, press N to finish this episode (save) and get the next instruction prompt. "
            "At the instruction prompt, empty line = quit.\n",
            file=sys.stderr,
        )
    else:
        print(
            "Teleop (gamepad): left stick = drive/turn, right stick = lift (vertical) / head (horizontal). "
            f"During recording, press gamepad button index {args.gamepad_next_button} (default: Start on many Xbox pads) "
            "to finish this episode early. At the instruction prompt, empty line = quit. "
            "Override with --gamepad-next-button if needed.\n",
            file=sys.stderr,
        )
    try:
        while True:
            task = _input_instruction()
            if not task:
                break
            logger.info("Recording episode index %s ...", dataset.meta.total_episodes)
            _record_episode(robot, dataset, task, fps, args.episode_time_s, tele_op, target_hwc)
    finally:
        dataset.finalize()
        logger.info("Dataset finalized at %s", dataset.root)

    if args.push:
        logger.info("Pushing to Hub: %s", args.repo_id)
        dataset.push_to_hub(private=args.private)


def entry() -> None:
    args = _parse_args()
    tele_op: TeleopProvider = make_teleop(
        args.teleop,
        joystick_index=args.joystick_index,
        gamepad_deadzone=args.gamepad_deadzone,
        invert_forward=args.invert_forward,
        gamepad_next_button=args.gamepad_next_button,
    )
    tele_op.start()
    try:

        def body(robot: PyCozmoRobot) -> None:
            main_collect(robot, args, tele_op)

        run_with_robot(body)
    finally:
        tele_op.stop()


if __name__ == "__main__":
    entry()
