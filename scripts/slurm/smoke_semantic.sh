#!/usr/bin/env bash
#SBATCH --job-name=smoke-sem
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/m000204-pm06b/joana/runs/smoke_semantic/slurm-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/runs/smoke_semantic/slurm-%j.err

# Semantic finetune SMOKE TEST.
# Goal: 1 clip x 1 epoch through the full semantic pipeline. Verifies:
#   - dataset returns dict with per-frame 'labels'
#   - 4DPreprocesser produces `semantic_labels` (Option A)
#   - InputVideoEmbedder colorizes + VAE-encodes -> 32-ch input_latents
#   - expanded DiT + expanded control_branch forward passes without dtype error
#   - training_loss returns a finite scalar
#   - a checkpoint saves without error at save_freq
#
# Not a real training run -- the output weights are discarded.

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/runs/smoke_semantic

# --- environment ---
# NeoVerse env (torch 2.3.1 cu121, gsplat, torch-scatter, etc.).
# Distinct from the sam3 env we use for labeling.
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
hash -r

# Quieter HF (don't need the download progress noise clogging the log)
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=warning

cd /scratch/m000204-pm06b/joana/NeoVerse

# --- sanity: env is what we expect ---
echo "hostname: $(hostname)"
echo "which python: $(which python)"
python -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
nvidia-smi | head -20

# Install wandb into neoverse env only if it's missing (idempotent, cheap when present).
# training/utils.py catches ImportError anyway, but installing here means live loss curves
# on wandb.ai instead of a silent skip.
python -c "import wandb" 2>/dev/null || python -m pip install --quiet wandb

# --- sanity: labels shape aligns with the video ---
python - <<'PY'
import os, numpy as np
from decord import VideoReader
npz = "/scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels/driving.npz"
vp = "/scratch/m000204-pm06b/joana/NeoVerse/examples/videos/driving.mp4"
d = np.load(npz)
n_labels = d["labels"].shape[0]
n_video = len(VideoReader(vp))
print(f"[sanity] labels: {n_labels} frames, video: {n_video} frames  {'OK' if n_labels == n_video else 'MISMATCH -- rerun with --static_scene'}")
assert n_labels == n_video, "SAM3 labels must be one-per-video-frame for training alignment (run scripts/setup_smoke_data.sh)"
PY

# --- launch training ---
# We run without DeepSpeed for the smoke -- single H100 is enough; the ZeRO
# config is overkill for 1 clip and adds complexity we don't need to test.
python train.py training/configs/smoke_semantic.yaml

echo "==> smoke test complete; check $SLURM_SUBMIT_DIR / $(pwd)/models/train for weights and slurm-*.out for log"
