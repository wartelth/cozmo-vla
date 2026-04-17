#!/usr/bin/env python
"""Layer-3 debug viewer: scrub through a recorded Cozmo rollout.

Open an interactive matplotlib window showing, for a run recorded with
``deploy_real_time.py --record-dir``:

  - the RGB camera frame at time t
  - line plots of applied actions (4-D) over the full run
  - line plots of the 8-D proprioception state
  - vertical cursor synced to a slider so you can scrub step-by-step

Example:
    python scripts/view_rollout.py --run-dir D:/cozmo-vla/outputs/rollouts/20260418_003123
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Pillow is required: pip install pillow") from e


ACTION_NAMES = ["left_wheel", "right_wheel", "lift_vel", "head_vel"]
STATE_NAMES = [
    "lift_ratio",
    "head_angle_norm",
    "left_wheel_norm",
    "right_wheel_norm",
    "pose_yaw_norm",
    "accel_z_norm",
    "padding_0",
    "cliff_or_picked_up",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Path to a rollout directory")
    return p.parse_args()


def _load_run(run_dir: Path):
    meta_path = run_dir / "run_meta.json"
    tele_path = run_dir / "telemetry.jsonl"
    if not tele_path.exists():
        raise SystemExit(f"No telemetry.jsonl in {run_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    steps: list[dict] = []
    with open(tele_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    if not steps:
        raise SystemExit("Telemetry file is empty.")

    t = np.array([s["t"] for s in steps], dtype=np.float32)
    applied = np.array([s["applied_action"] for s in steps], dtype=np.float32)
    raw = np.array([s["raw_action"] for s in steps], dtype=np.float32)
    state = np.array([s["state"] for s in steps], dtype=np.float32)
    unsafe = np.array([s.get("unsafe", False) for s in steps], dtype=bool)
    inf_s = np.array([s.get("inference_s", 0.0) for s in steps], dtype=np.float32)

    frame_paths = []
    last = None
    for s in steps:
        rel = s.get("frame")
        if rel:
            last = run_dir / rel
        frame_paths.append(last)

    return meta, steps, t, applied, raw, state, unsafe, inf_s, frame_paths


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    meta, steps, t, applied, raw, state, unsafe, inf_s, frame_paths = _load_run(run_dir)

    n = len(steps)
    task = meta.get("task", "")
    hz = meta.get("hz", None)
    title = f"{run_dir.name}  -  task: {task!r}" + (f"  -  {hz} Hz" if hz else "")

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(title)

    gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1.6], height_ratios=[1, 1, 0.08], hspace=0.35, wspace=0.15)
    ax_img = fig.add_subplot(gs[0:2, 0])
    ax_act = fig.add_subplot(gs[0, 1])
    ax_st = fig.add_subplot(gs[1, 1], sharex=ax_act)
    ax_slider = fig.add_subplot(gs[2, :])

    ax_img.set_axis_off()
    first_frame = _safe_load_image(frame_paths[0])
    if first_frame is None:
        first_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    im_handle = ax_img.imshow(first_frame)
    title_handle = ax_img.set_title(_frame_title(0, t, inf_s, unsafe, applied))

    for i, name in enumerate(ACTION_NAMES[: applied.shape[1]]):
        ax_act.plot(t, applied[:, i], label=name, linewidth=1.1)
    ax_act.axhline(0, color="k", linewidth=0.4, alpha=0.4)
    ax_act.set_ylim(-1.1, 1.1)
    ax_act.set_ylabel("action (norm)")
    ax_act.grid(alpha=0.25)
    ax_act.legend(loc="upper right", fontsize=8, ncol=4)
    for idx in np.flatnonzero(unsafe):
        ax_act.axvline(t[idx], color="red", alpha=0.2, linewidth=0.6)
    cursor_act = ax_act.axvline(t[0], color="k", linewidth=1.0, alpha=0.85)

    for i, name in enumerate(STATE_NAMES[: state.shape[1]]):
        ax_st.plot(t, state[:, i], label=name, linewidth=0.9)
    ax_st.set_ylabel("state")
    ax_st.set_xlabel("t (s)")
    ax_st.grid(alpha=0.25)
    ax_st.legend(loc="upper right", fontsize=7, ncol=4)
    cursor_st = ax_st.axvline(t[0], color="k", linewidth=1.0, alpha=0.85)

    slider = Slider(ax_slider, "step", 0, n - 1, valinit=0, valstep=1)

    def update(val):
        i = int(slider.val)
        img = _safe_load_image(frame_paths[i])
        if img is not None:
            im_handle.set_data(img)
        cursor_act.set_xdata([t[i], t[i]])
        cursor_st.set_xdata([t[i], t[i]])
        title_handle.set_text(_frame_title(i, t, inf_s, unsafe, applied))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event):
        if event.key in ("right", "n"):
            slider.set_val(min(n - 1, int(slider.val) + 1))
        elif event.key in ("left", "p"):
            slider.set_val(max(0, int(slider.val) - 1))
        elif event.key in ("pageup",):
            slider.set_val(min(n - 1, int(slider.val) + 10))
        elif event.key in ("pagedown",):
            slider.set_val(max(0, int(slider.val) - 10))
        elif event.key == "home":
            slider.set_val(0)
        elif event.key == "end":
            slider.set_val(n - 1)

    fig.canvas.mpl_connect("key_press_event", on_key)

    fps = 1.0 / max(float(inf_s.mean()), 1e-6)
    print(f"Loaded {n} steps from {run_dir}")
    print(f"  mean inference = {1000 * inf_s.mean():.1f} ms  ({fps:.1f} Hz max)")
    print(f"  unsafe frames  = {int(unsafe.sum())}")
    print("Keys: left/right arrow = step, pageup/pagedown = +/-10, home/end = ends.")

    plt.show()


def _safe_load_image(path):
    if path is None or not Path(path).exists():
        return None
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


def _frame_title(i, t, inf_s, unsafe, applied):
    flags = []
    if unsafe[i]:
        flags.append("UNSAFE")
    if np.any(np.abs(applied[i]) >= 0.99):
        flags.append("SAT")
    flag_str = f"  [{', '.join(flags)}]" if flags else ""
    return f"step {i}   t={t[i]:.2f}s   inf={1000 * inf_s[i]:.1f}ms{flag_str}"


if __name__ == "__main__":
    main()
