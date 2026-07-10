#!/usr/bin/env bash
#SBATCH --job-name=train-sem
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/runs/train_semantic_v1/slurm-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/runs/train_semantic_v1/slurm-%j.err

# REAL semantic finetune training on ~30 RUGD clips.
# 10 epochs x ~30 clips x 4-step gradient accumulation -> ~75-90 gradient steps.
# Load time: ~4 min. Per-step time: ~30-60 sec. Wall-clock estimate: ~4-6 hours.
#
# Overshoot the SLURM time budget (12h) so we don't get killed mid-checkpoint.

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v1

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

# Quieter HF / transformers logs
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=warning

cd /scratch/m000204-pm06b/joana/NeoVerse

# --- sanity: env is what we expect ---
echo "hostname: $(hostname)"
echo "which python: $(which python)"
python -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')"
nvidia-smi | head -20

# --- sanity: dataset has clips + labels ---
python - <<'PY'
import os, numpy as np
import pandas as pd
from decord import VideoReader

ROOT = "/scratch/m000204-pm06b/joana/rugd_train_data"
LABELS_DIR = "/scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels"

meta = pd.read_csv(os.path.join(ROOT, "data/train/SpatialVID_HQ_metadata.csv"))
print(f"[sanity] {len(meta)} clips in metadata csv")

ok = 0
missing = 0
for _, row in meta.iterrows():
    vp = os.path.join(ROOT, "SpatialVid/HQ", row["video path"])
    lp = os.path.join(LABELS_DIR, f"{row['id']}.npz")
    if not os.path.exists(vp):
        print(f"  MISSING video: {vp}")
        missing += 1
        continue
    if not os.path.exists(lp):
        print(f"  MISSING label: {lp}")
        missing += 1
        continue
    n_v = len(VideoReader(vp))
    n_l = np.load(lp)["labels"].shape[0]
    if n_v != n_l:
        print(f"  MISALIGNED {row['id']}: video={n_v}, labels={n_l}")
        missing += 1
        continue
    ok += 1
print(f"[sanity] {ok} clips have matching video + label; {missing} missing/misaligned")
assert ok > 0, "no valid clips -- did the labeling job finish? did setup_rugd_train_data.sh run?"
PY

# Install wandb if missing (idempotent)
python -c "import wandb" 2>/dev/null || python -m pip install --quiet wandb

# --- launch training ---
python train.py training/configs/train_semantic.yaml

echo "==> training done; latest checkpoint in /scratch/m000204-pm06b/joana/runs/train_semantic_v1/"
