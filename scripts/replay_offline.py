#!/usr/bin/env python
"""Layer-1 debug: replay the policy on a training episode and plot predicted vs.
ground-truth actions.

If the predicted line tracks the recorded one, the model has learned the data
and real-world failures are distribution-shift / closed-loop problems.
If it looks like noise, the checkpoint is undertrained or the inference
pipeline differs from training (preprocessing / normalization / input order).

Example:
    python scripts/replay_offline.py \
        --policy.path D:/cozmo-vla/lerobot/outputs/train/cozmo_smolvla/checkpoints/002000/pretrained_model \
        --dataset-repo-id local/cozmo_vla \
        --dataset-root D:/cozmo-vla/datasets/my_run \
        --episode 0 \
        --output D:/cozmo-vla/outputs/debug/replay_ep0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from debug_common import (
    DebugContext,
    episode_bounds,
    frame_to_obs,
    load_context,
    predict_action,
    reset_policy,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy.path", dest="policy_path", required=True)
    p.add_argument("--dataset-repo-id", required=True)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 = full episode. Useful for quick smoke tests.",
    )
    p.add_argument(
        "--task-override",
        type=str,
        default=None,
        help="If set, use this task string instead of the dataset's stored one.",
    )
    p.add_argument(
        "--reset-every",
        type=int,
        default=0,
        help=(
            "If >0, call policy.reset() every N steps. SmolVLA caches a 50-step action "
            "chunk, so without reset the model sees a new observation only every 50 steps "
            "(this mimics deployment but hides per-step model quality). Use --reset-every 1 "
            "to force a fresh chunk each step = true 'per-step fit to training data'."
        ),
    )
    p.add_argument("--output", type=str, default="outputs/debug/replay", help="Output directory")
    return p.parse_args()


def run(ctx: DebugContext, args: argparse.Namespace) -> dict:
    from_idx, to_idx, length = episode_bounds(ctx, args.episode)
    if args.max_frames > 0:
        to_idx = min(to_idx, from_idx + args.max_frames)

    print(f"Replaying episode {args.episode}: frames [{from_idx}, {to_idx}) "
          f"({to_idx - from_idx} frames out of {length}).")

    reset_policy(ctx)

    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    tasks_seen: set[str] = set()

    t_log_every = max(1, (to_idx - from_idx) // 10)
    for step, idx in enumerate(range(from_idx, to_idx)):
        if args.reset_every > 0 and step % args.reset_every == 0 and step > 0:
            reset_policy(ctx)

        frame = ctx.dataset[idx]
        task = args.task_override if args.task_override else str(frame.get("task", ""))
        tasks_seen.add(task)

        raw = frame_to_obs(frame, task)
        pred = predict_action(ctx, raw)
        gt = frame["action"].detach().cpu().numpy().astype(np.float32).reshape(-1)

        preds.append(pred[: len(gt)])
        gts.append(gt)

        if step % t_log_every == 0:
            l1 = float(np.abs(pred[: len(gt)] - gt).mean())
            print(f"  [{step:5d}/{to_idx - from_idx}] |pred-gt|_1={l1:.4f}  task='{task[:60]}'")

    preds_arr = np.stack(preds)  # (T, A)
    gts_arr = np.stack(gts)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _compute_metrics(preds_arr, gts_arr, ctx.action_names)
    _write_csv(out_dir / "actions.csv", preds_arr, gts_arr, ctx.action_names)
    _write_plot(out_dir / "actions.png", preds_arr, gts_arr, ctx.action_names, args.episode)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "episode": args.episode,
                "num_frames": int(preds_arr.shape[0]),
                "tasks_seen": sorted(tasks_seen),
                "action_names": ctx.action_names,
                **metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {out_dir}/actions.png, actions.csv, metrics.json")
    print("Per-dimension MAE / correlation:")
    for name, mae, corr in zip(
        ctx.action_names or [f"a{i}" for i in range(preds_arr.shape[1])],
        metrics["mae_per_dim"],
        metrics["pearson_per_dim"],
    ):
        print(f"  {name:20s}  MAE={mae:.4f}  pearson={corr:+.3f}")
    print(f"Overall MAE={metrics['mae_overall']:.4f}  |  worst-dim pearson={min(metrics['pearson_per_dim']):+.3f}")
    return metrics


def _compute_metrics(preds: np.ndarray, gts: np.ndarray, names: list[str]) -> dict:
    mae_per_dim = np.abs(preds - gts).mean(axis=0)
    pearson_per_dim = []
    for i in range(preds.shape[1]):
        p, g = preds[:, i], gts[:, i]
        if p.std() < 1e-8 or g.std() < 1e-8:
            pearson_per_dim.append(0.0)
        else:
            pearson_per_dim.append(float(np.corrcoef(p, g)[0, 1]))
    return {
        "mae_overall": float(np.abs(preds - gts).mean()),
        "mae_per_dim": [float(x) for x in mae_per_dim],
        "pearson_per_dim": pearson_per_dim,
        "action_std_pred": [float(x) for x in preds.std(axis=0)],
        "action_std_gt": [float(x) for x in gts.std(axis=0)],
    }


def _write_csv(path: Path, preds: np.ndarray, gts: np.ndarray, names: list[str]) -> None:
    a = preds.shape[1]
    header = ["step"]
    for n in names or [f"a{i}" for i in range(a)]:
        header.append(f"pred_{n}")
    for n in names or [f"a{i}" for i in range(a)]:
        header.append(f"gt_{n}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t in range(preds.shape[0]):
            w.writerow([t, *preds[t].tolist(), *gts[t].tolist()])


def _write_plot(
    path: Path,
    preds: np.ndarray,
    gts: np.ndarray,
    names: list[str],
    episode: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = preds.shape[1]
    names = names or [f"a{i}" for i in range(a)]
    fig, axes = plt.subplots(a, 1, figsize=(10, 2.2 * a), sharex=True)
    if a == 1:
        axes = [axes]
    t = np.arange(preds.shape[0])
    for i, ax in enumerate(axes):
        ax.plot(t, gts[:, i], label="ground truth", color="#2b7a2b", linewidth=1.4)
        ax.plot(t, preds[:, i], label="predicted", color="#c03030", linewidth=1.0, alpha=0.85)
        ax.set_ylabel(names[i])
        ax.grid(alpha=0.25)
        ax.set_ylim(-1.1, 1.1)
    axes[0].set_title(f"Offline replay - episode {episode} - pred vs. GT")
    axes[-1].set_xlabel("frame")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    ctx = load_context(args.policy_path, args.dataset_repo_id, args.dataset_root)
    run(ctx, args)


if __name__ == "__main__":
    main()
