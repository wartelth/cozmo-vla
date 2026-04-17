#!/usr/bin/env python
"""
Interactive teleop test for Cozmo (no dataset, no LeRobot).

Use this to verify Wi‑Fi connection, PyCozmo, camera, and controls before recording.

  python scripts/teleop_debug.py --teleop keyboard
  python scripts/teleop_debug.py --teleop gamepad -v
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from cozmo_vla.robot.pycozmo_client import PyCozmoRobot, run_with_robot
from cozmo_vla.teleop import GamepadTeleop, TeleopProvider, make_teleop


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=sys.stderr)
    # Quiet noisy third-party loggers unless verbose
    if not verbose:
        for name in ("pycozmo", "PIL", "pygame"):
            logging.getLogger(name).setLevel(logging.WARNING)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--teleop",
        choices=("keyboard", "gamepad"),
        default="keyboard",
        help="keyboard: WASD + A/D turn, R/F lift, T/G head | gamepad: sticks (see README)",
    )
    p.add_argument("--joystick-index", type=int, default=0)
    p.add_argument("--gamepad-deadzone", type=float, default=0.15)
    p.add_argument("--invert-forward", action="store_true")
    p.add_argument(
        "--hz",
        type=float,
        default=30.0,
        help="Target control loop rate (Hz). Cozmo / Wi‑Fi may not always keep up.",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after N seconds (0 = run until Ctrl+C)",
    )
    p.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="Seconds between INFO summary lines (action, wheels, battery, safety)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG: log every control tick + raw gamepad axes when using gamepad",
    )
    p.add_argument(
        "--no-camera-read",
        action="store_true",
        help="Skip fetching a camera frame each tick (faster loop test; less load)",
    )
    return p.parse_args()


def _fmt_action(a: np.ndarray) -> str:
    return f"[{a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f},{a[3]:+.2f}]"


def _run_loop(robot: PyCozmoRobot, args: argparse.Namespace, tele_op: TeleopProvider) -> None:
    log = logging.getLogger("teleop_debug")
    dt = 1.0 / max(args.hz, 1.0)
    t_stop = time.perf_counter() + args.duration if args.duration > 0 else None

    log.info("Starting control loop at target %.1f Hz (actual rate is logged).", args.hz)
    log.info(
        "Keyboard: W/S forward-back, A/D turn, R/F lift, T/G head | Ctrl+C to exit."
        if args.teleop == "keyboard"
        else "Gamepad: left stick drive, right stick lift/head | Ctrl+C to exit."
    )

    tick = 0
    last_status = time.perf_counter()
    prev_loop: float | None = None
    hz_ema = 0.0

    while True:
        loop_start = time.perf_counter()
        if t_stop is not None and loop_start >= t_stop:
            log.info("Duration elapsed (%.1f s). Stopping.", args.duration)
            break

        action = tele_op.action_vector()
        robot.apply_action_normalized(action, stop_if_unsafe=True)

        cam_note = "skip"
        if not args.no_camera_read:
            try:
                rgb = robot.get_rgb_uint8()
                cam_note = f"{rgb.shape[1]}x{rgb.shape[0]}"
            except Exception as e:
                cam_note = f"err:{e!r}"

        info = robot.get_debug_info()
        tick += 1

        loop_dt = (loop_start - prev_loop) if prev_loop is not None else 0.0
        prev_loop = loop_start
        if loop_dt > 0 and tick >= 2:
            hz_inst = 1.0 / loop_dt
            hz_ema = hz_inst if hz_ema == 0 else 0.85 * hz_ema + 0.15 * hz_inst

        if args.verbose:
            extra = ""
            if isinstance(tele_op, GamepadTeleop):
                extra = f" | raw_axes={tele_op.raw_axes_snapshot()}"
            log.debug(
                "tick=%d loop_dt_ms=%.1f action=%s wheels_mmps=(%.0f,%.0f) unsafe=%s cam=%s%s",
                tick,
                loop_dt * 1000.0,
                _fmt_action(action),
                info["left_wheel_mmps"],
                info["right_wheel_mmps"],
                info["unsafe"],
                cam_note,
                extra,
            )

        now = time.perf_counter()
        if now - last_status >= args.status_interval and not args.verbose:
            log.info(
                "hz~=%.1f | action=%s | L/R wheel=%.0f/%.0f mm/s | bat=%.2fV | cliff=%s picked=%s unsafe=%s | cam=%s | status=%s",
                hz_ema,
                _fmt_action(action),
                info["left_wheel_mmps"],
                info["right_wheel_mmps"],
                info["battery_v"],
                info["cliff"],
                info["picked_up"],
                info["unsafe"],
                cam_note,
                info["robot_status_hex"],
            )
            last_status = now

        elapsed = time.perf_counter() - loop_start
        sleep_for = max(0.0, dt - elapsed)
        time.sleep(sleep_for)

    robot.stop()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.verbose)

    log = logging.getLogger("teleop_debug")
    log.info("cozmo-vla teleop debug (PyCozmo). No LeRobot required for this script.")

    tele_op = make_teleop(
        args.teleop,
        joystick_index=args.joystick_index,
        gamepad_deadzone=args.gamepad_deadzone,
        invert_forward=args.invert_forward,
    )
    tele_op.start()
    try:

        def body(robot: PyCozmoRobot) -> None:
            logging.getLogger("teleop_debug").info("Robot ready. %s", robot.get_debug_info())
            _run_loop(robot, args, tele_op)

        try:
            run_with_robot(body)
        except KeyboardInterrupt:
            logging.getLogger("teleop_debug").info("Interrupted by user (Ctrl+C).")
    finally:
        try:
            tele_op.stop()
        except BaseException:
            pass


if __name__ == "__main__":
    main()
