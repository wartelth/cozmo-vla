#!/usr/bin/env python
"""Thin wrapper around the LeRobot CLI ``lerobot-train`` for SmolVLA fine-tuning."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# SmolVLA base was trained with 3 cams (camera1..3). Cozmo recordings use observation.images.front only.
_COZMO_SMOLVLA_RENAME = {"observation.images.front": "observation.images.camera1"}


def _resolve_policy_path_hub_safe(policy_path: str) -> str:
    """
    LeRobot parses ``--policy.path`` as a pathlib Path; on Windows, ``org/name`` becomes
    ``org\\\\name`` and ``hf_hub_download`` rejects it. Resolve Hub model ids to a cache
    directory first (same files, valid local path).
    """
    raw = policy_path.strip()
    candidate = Path(raw)
    if candidate.is_dir():
        return str(candidate.resolve())
    # Heuristic: Hugging Face model id (e.g. lerobot/smolvla_base), not a relative Windows path.
    if (
        "/" in raw
        and not raw.startswith((".", "/"))
        and not candidate.is_absolute()
        and all(part not in (".", "..") for part in raw.split("/"))
    ):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            print("Install huggingface_hub to resolve Hub policy paths.", file=sys.stderr)
            raise e
        print(f"Resolving Hub model to local cache: {raw}", file=sys.stderr)
        return snapshot_download(repo_id=raw, repo_type="model")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="HF dataset repo id used for training (same as --dataset.repo_id for lerobot-train)",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Local dataset folder (sets --dataset.root= for offline training; use the same path as collect_data --root)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/train/cozmo_smolvla",
        help="Training output directory",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--policy.device", dest="policy_device", type=str, default="cuda")
    parser.add_argument("--job_name", type=str, default="cozmo_smolvla")
    parser.add_argument(
        "--wandb.enable",
        dest="wandb_enable",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=True,
    )
    parser.add_argument(
        "--policy.path",
        dest="policy_path",
        type=str,
        default="lerobot/smolvla_base",
        help="Base policy on the Hub",
    )
    parser.add_argument(
        "--hub-push",
        action="store_true",
        help="Set to upload checkpoints to the Hub (you must also pass e.g. --policy.repo_id=USER/name as an extra arg).",
    )
    parser.add_argument(
        "--no-cozmo-smolvla-presets",
        action="store_true",
        help="Do not add rename_map (front→camera1) and policy.empty_cameras=2 needed for lerobot/smolvla_base on single-camera Cozmo data.",
    )
    args, unknown = parser.parse_known_args()

    policy_path_arg = _resolve_policy_path_hub_safe(args.policy_path)

    # Prefer the CLI in this Python environment so we do not pick another env on PATH.
    # Conda on Windows: python.exe at env root, scripts in Scripts\; venv: python.exe in Scripts\.
    root = Path(sys.executable).resolve().parent
    if sys.platform == "win32":
        candidates = (root / "Scripts" / "lerobot-train.exe", root / "lerobot-train.exe")
    else:
        candidates = (root / "lerobot-train",)
    exe = next((str(c) for c in candidates if c.is_file()), None) or shutil.which(
        "lerobot-train"
    )
    if exe is None:
        print(
            "Could not find `lerobot-train` on PATH. Install LeRobot from source:\n"
            "  git clone https://github.com/huggingface/lerobot.git\n"
            "  cd lerobot && pip install -e \".[smolvla]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [
        exe,
        f"--policy.path={policy_path_arg}",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--policy.device={args.policy_device}",
        f"--wandb.enable={str(args.wandb_enable).lower()}",
    ]
    if not args.hub_push:
        cmd.append("--policy.push_to_hub=false")
    if args.dataset_root:
        cmd.append(f"--dataset.root={args.dataset_root}")
    if not args.no_cozmo_smolvla_presets:
        cmd.append(f"--rename_map={json.dumps(_COZMO_SMOLVLA_RENAME)}")
        cmd.append("--policy.empty_cameras=2")
    cmd.extend(unknown)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
