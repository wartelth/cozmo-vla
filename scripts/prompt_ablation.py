#!/usr/bin/env python
"""Layer-2 debug: does the VLM actually listen to the task string?

For a handful of random dataset frames we run the same observation through the
policy with several different task strings and compare the predicted actions.

Interpretation:
  - Pairwise L2 distances near 0  => language is being ignored; model predicts
    actions from (image, state) alone. This is very common with tiny datasets
    that only ever saw 1-2 task strings.
  - Pairwise L2 distances large   => VLM is conditioning on language; policy
    errors are more likely due to undertrained action expert / data scarcity.

Example:
    python scripts/prompt_ablation.py \
        --policy.path D:/cozmo-vla/lerobot/outputs/train/cozmo_smolvla/checkpoints/002000/pretrained_model \
        --dataset-repo-id local/cozmo_vla \
        --dataset-root D:/cozmo-vla/datasets/my_run \
        --num-frames 16 \
        --tasks "find the teddy bear, and drive close to it" "stop and do nothing" "spin in place" ""
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from debug_common import (
    DebugContext,
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
    p.add_argument("--num-frames", type=int, default=16, help="How many dataset frames to sample")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "find the teddy bear, and drive close to it",
            "stop and do nothing",
            "spin in place",
            "",
        ],
        help="Space-separated list of task strings to compare (quote each).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="outputs/debug/ablation")
    return p.parse_args()


def run(ctx: DebugContext, args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    n_total = len(ctx.dataset)
    indices = sorted(rng.choice(n_total, size=min(args.num_frames, n_total), replace=False).tolist())

    tasks = list(args.tasks)
    n_tasks = len(tasks)
    n_frames = len(indices)
    a_dim = int(ctx.dataset.meta.features["action"]["shape"][0])

    actions = np.zeros((n_frames, n_tasks, a_dim), dtype=np.float32)

    print(f"Sampled {n_frames} frames, comparing {n_tasks} task strings:")
    for t in tasks:
        print(f"  - {t!r}")

    for fi, idx in enumerate(indices):
        frame = ctx.dataset[idx]
        for ti, task in enumerate(tasks):
            # Reset between prompts so cached action chunks can't leak across tasks.
            reset_policy(ctx)
            raw = frame_to_obs(frame, task)
            actions[fi, ti] = predict_action(ctx, raw)[:a_dim]

    pair_dists = np.zeros((n_tasks, n_tasks), dtype=np.float32)
    for i in range(n_tasks):
        for j in range(n_tasks):
            d = np.linalg.norm(actions[:, i, :] - actions[:, j, :], axis=-1)
            pair_dists[i, j] = float(d.mean())

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_plot(out_dir / "task_distance_matrix.png", pair_dists, tasks)

    per_dim_var_across_tasks = actions.var(axis=1).mean(axis=0)  # avg over frames
    per_dim_var_across_frames = actions.var(axis=0).mean(axis=0)  # avg over tasks

    payload = {
        "tasks": tasks,
        "frame_indices": [int(x) for x in indices],
        "pairwise_mean_l2": pair_dists.tolist(),
        "per_dim_var_across_tasks": [float(x) for x in per_dim_var_across_tasks],
        "per_dim_var_across_frames": [float(x) for x in per_dim_var_across_frames],
        "action_names": ctx.action_names,
    }
    (out_dir / "ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nPairwise mean L2 between task outputs (off-diagonal):")
    hdr = "          " + "  ".join(f"T{i:<2d}" for i in range(n_tasks))
    print(hdr)
    for i in range(n_tasks):
        row = "  ".join(f"{pair_dists[i, j]:.3f}" for j in range(n_tasks))
        print(f"  T{i:<2d}  {row}")

    print("\nVariance attributable to task (averaged over frames):", per_dim_var_across_tasks)
    print("Variance attributable to frame (averaged over tasks):", per_dim_var_across_frames)

    ratio = per_dim_var_across_tasks.sum() / max(per_dim_var_across_frames.sum(), 1e-8)
    print(f"\nTask-variance / frame-variance ratio = {ratio:.3f}")
    if ratio < 0.05:
        print("  => Model almost IGNORES the task string. VLM language conditioning is weak.")
    elif ratio < 0.5:
        print("  => Model is weakly language-conditioned.")
    else:
        print("  => Model responds meaningfully to different task strings.")

    return payload


def _write_plot(path: Path, pair_dists: np.ndarray, tasks: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.2 * len(tasks) + 3, 1.2 * len(tasks) + 2))
    im = ax.imshow(pair_dists, cmap="viridis")
    for i in range(pair_dists.shape[0]):
        for j in range(pair_dists.shape[1]):
            ax.text(j, i, f"{pair_dists[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
    short = [t if len(t) <= 28 else t[:25] + "..." for t in tasks]
    ax.set_xticks(range(len(tasks)))
    ax.set_yticks(range(len(tasks)))
    ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_yticklabels(short)
    ax.set_title("Mean L2 between policy actions across task strings\n(higher = more language-sensitive)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    ctx = load_context(args.policy_path, args.dataset_repo_id, args.dataset_root)
    run(ctx, args)


if __name__ == "__main__":
    main()
