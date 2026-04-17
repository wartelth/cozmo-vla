# Cozmo VLA (PyCozmo + LeRobot + SmolVLA)

End-to-end learning stack for collecting demonstrations on **Cozmo** with **PyCozmo**, storing them as a **LeRobot v3** dataset, fine-tuning **SmolVLA**, and running inference on the host PC.

## Prerequisites

- **Python 3.12+** for the full stack (training/deploy with LeRobot 0.5.x; see [`lerobot/pyproject.toml`](lerobot/pyproject.toml)). The `cozmo-vla` package alone allows `>=3.10`, but **SmolVLA + `lerobot-train` need 3.12**.
- Cozmo on Wi‑Fi, PyCozmo-compatible firmware
- Windows: [FFmpeg](https://ffmpeg.org/) on `PATH` (recommended for video encoding in LeRobot)
- Hugging Face account and token for datasets/models (`HF_TOKEN` or `huggingface-cli login`)
- **GPU fine-tuning:** NVIDIA driver + PyTorch built with CUDA (see below)

## 1. Install this repo

```bash
cd cozmo-vla
py -3.12 -m venv .venv   # Windows; elsewhere e.g. python3.12 -m venv .venv
.venv\Scripts\activate   # Linux/macOS: source .venv/bin/activate
pip install -e .
```

Copy `.env.example` to `.env` and set `HF_TOKEN` if you use private repos or uploads.

### GPU environment (CUDA 12.6, aligned with `requirements-gpu-torch.txt`)

[`requirements-gpu-torch.txt`](requirements-gpu-torch.txt) pins **PyTorch 2.9.1+cu126** (comments reference an external `CUDASetup/dl-gpu.yml` if you use that layout; it is not shipped in this repo). Use **Python 3.12** for new envs so it matches [`environment-gpu.yml`](environment-gpu.yml) and LeRobot.

**Option A — reuse an existing GPU conda env (e.g. `dl-gpu`) if it is already Python 3.12+:**

```bash
conda activate dl-gpu
cd cozmo-vla
pip install -e .
# Only if pip chooses a CPU-only torch:  pip install -r requirements-gpu-torch.txt
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"
```

If your env is older than 3.12, use Option B or create a new 3.12 env — LeRobot will refuse to install otherwise.

**Option B — new conda env from [`environment-gpu.yml`](environment-gpu.yml):**

```bash
conda env create -f environment-gpu.yml
conda activate cozmo-vla-gpu
pip install -r requirements-gpu-torch.txt
pip install -e .
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"
```

Verify the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

Training uses `--policy.device cuda` by default in [`scripts/train_smolvla.py`](scripts/train_smolvla.py).

## 2. Install LeRobot with SmolVLA

LeRobot is installed from a **local clone** (SmolVLA extras), in the **same** environment as above:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"
```

Ensure `lerobot-train` is on your `PATH` (same environment as above).

## 3. Test teleop (recommended first)

**LeRobot is not required** — only **PyCozmo** (robot), **Pygame** (Xbox / SDL gamepad), **NumPy**, and **Pynput** (keyboard). Nothing else for this step.

Minimal install (skips OpenCV / `huggingface_hub` / LeRobot):

```bash
pip install -r requirements-teleop.txt
pip install -e . --no-deps
```

Or use the full `pip install -e .` from step 1 if you already did that.

Verifies connection, camera, wheels, lift/head, battery, and cliff flags with readable logs:

```bash
python scripts/teleop_debug.py --teleop keyboard
python scripts/teleop_debug.py --teleop gamepad -v
```

- **INFO** (default): summary line about once per second (`hz`, action, wheels mm/s, battery, safety, camera size).
- **`-v`**: DEBUG every tick (includes raw gamepad axes).
- **`--duration 15`**: auto-stop after 15 seconds.
- **`--no-camera-read`**: stress the control loop without decoding frames.

## 4. Record a dataset

Keyboard teleop (hold keys):

| Keys | Action |
|------|--------|
| W / S | Forward / back |
| A / D | Turn |
| R / F | Lift up / down |
| T / G | Head up / down |

Data is **always saved on disk** first (under `%USERPROFILE%\.cache\lerobot\<repo-id>` by default, or `--root`). **`--repo-id` is only a dataset name** — it does **not** contact Hugging Face unless you add **`--push`**.

```bash
python scripts/collect_data.py --vcodec h264
```

**Xbox / gamepad** (often smoother than keyboard): plug in or pair the controller, then:

```bash
python scripts/collect_data.py --vcodec h264 --teleop gamepad
```

Optional: `--repo-id local/my_run` or `--root D:\cozmo-vla\datasets\my_run` to control where files go. If **`--root` already contains a dataset** (`meta/info.json` present), the script **resumes** and appends more episodes. Use **`--overwrite`** only when you intend to delete that folder and start over.

**Episode flow (this repo, not LeRobot-specific):** you type a language instruction, then teleop until the behavior is done. **`--episode-time-s` is a maximum** — you do not have to wait it out. **Keyboard: press `N`** while recording to **save that episode** and return to the next instruction prompt. **Gamepad: default button index 7** (often *Start* on Xbox-style pads); change with `--gamepad-next-button` if your mapping differs.

- **Left stick**: forward/back and turn (same mixing idea as W/A/S/D).
- **Right stick**: vertical = lift, horizontal = head.
- If driving feels reversed, add `--invert-forward`.
- Second controller: `--joystick-index 1`.

- At the **instruction** prompt, leave the line **empty** and press Enter to quit and call `finalize()`.
- Add **`--push`** only if you want to upload the finished dataset to the Hub (needs `HF_TOKEN` / login).

Action space, image size, and FPS are defined in [`configs/cozmo_action_space.json`](configs/cozmo_action_space.json).

## 5. Fine-tune SmolVLA

**Local dataset** (same `--root` as collection), **GPU** (default), modest batch size for ~12 GB VRAM:

```bash
conda activate dl-gpu   # or cozmo-vla-gpu
cd cozmo-vla
python scripts/train_smolvla.py --dataset-repo-id local/cozmo_vla --dataset-root D:\cozmo-vla\datasets\my_run --output_dir outputs/train/cozmo_smolvla --batch_size 8 --policy.device cuda --wandb.enable false
```

Hub-hosted dataset (no `--dataset-root`):

```bash
python scripts/train_smolvla.py --dataset-repo-id YOUR_HF_USER/cozmo_pickplace_v1 --output_dir outputs/train/cozmo_smolvla --policy.device cuda
```

Extra `lerobot-train` flags can be appended; they are forwarded unchanged. The wrapper adds Cozmo/SmolVLA presets (`rename_map`, `empty_cameras`) unless you pass `--no-cozmo-smolvla-presets`.

## 6. Deploy on the robot

Use the **same dataset** repo as training (for normalization statistics):

```bash
python scripts/deploy_real_time.py --policy.path outputs/train/cozmo_smolvla/checkpoints/last/pretrained_model --dataset-repo-id YOUR_HF_USER/cozmo_pickplace_v1 --task "your instruction"
```

Adjust `--policy.path` to your fine-tuned Hub model or local checkpoint directory.

## Layout

- [`environment-gpu.yml`](environment-gpu.yml) — conda env skeleton (**Python 3.12**); pair with [`requirements-gpu-torch.txt`](requirements-gpu-torch.txt) (CUDA 12.6 wheels)
- [`src/cozmo_vla/robot/pycozmo_client.py`](src/cozmo_vla/robot/pycozmo_client.py) — PyCozmo wrapper, safety stops, normalized actions
- [`src/cozmo_vla/datasets/build_features.py`](src/cozmo_vla/datasets/build_features.py) — LeRobot feature schema
- [`scripts/teleop_debug.py`](scripts/teleop_debug.py) — teleop-only test with detailed logs (no LeRobot)
- [`scripts/collect_data.py`](scripts/collect_data.py) — teleop + `LeRobotDataset.create`
- [`scripts/train_smolvla.py`](scripts/train_smolvla.py) — `lerobot-train` wrapper
- [`scripts/deploy_real_time.py`](scripts/deploy_real_time.py) — SmolVLA inference loop

## Notes

- Helper scripts (not required for the main flow): [`scripts/test_connection.py`](scripts/test_connection.py) (minimal PyCozmo connect), [`scripts/diag_torch_dll.py`](scripts/diag_torch_dll.py) (Windows torch DLL paths).
- Inference runs on the **host**; the robot only receives motor commands over Wi‑Fi.
- SmolVLA expects enough demonstrations per variation; see the [SmolVLA docs](https://huggingface.co/docs/lerobot/main/en/smolvla).
- If preprocessor keys differ from `observation.images.front`, adjust [`configs/cozmo_action_space.json`](configs/cozmo_action_space.json) and re-record.
