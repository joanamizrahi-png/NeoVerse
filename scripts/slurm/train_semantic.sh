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
#SBATCH --exclude=n04,n13,n17,n24

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

# Single source of truth for which yaml is being trained on. The sanity block
# below and `python train.py` below both use this path -- keeps them in lockstep.
# CONFIG env overrides (e.g. CONFIG=training/configs/train_semantic_v15.yaml).
CONFIG_PATH="${CONFIG:-training/configs/train_semantic.yaml}"

# --- sanity: dataset has clips + labels ---
# Parses ROOT and labels_dir straight out of the yaml's `train_dataset` line so
# it never disagrees with the actual training. If the parse fails (missing keys,
# odd formatting), the block errors out and the job doesn't waste a GPU slot.
CONFIG_PATH="$CONFIG_PATH" python - <<'PY'
import os, re, sys, numpy as np
import pandas as pd
from decord import VideoReader
import yaml

config_path = os.environ["CONFIG_PATH"]
with open(config_path) as f:
    cfg = yaml.safe_load(f)

dataset_line = cfg.get("train_dataset", "")
m_root = re.search(r'ROOT="([^"]+)"', dataset_line)
m_labels = re.search(r'labels_dir="([^"]+)"', dataset_line)
if not m_root or not m_labels:
    sys.exit(f"[sanity] could not parse ROOT / labels_dir out of train_dataset:\n  {dataset_line}")
ROOT = m_root.group(1)
LABELS_DIR = m_labels.group(1)
print(f"[sanity] config: {config_path}")
print(f"[sanity] ROOT       = {ROOT}")
print(f"[sanity] LABELS_DIR = {LABELS_DIR}")

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
assert ok > 0, "no valid clips -- did the labeling job finish? did the setup script run?"
PY

# Install wandb if missing (idempotent)
python -c "import wandb" 2>/dev/null || python -m pip install --quiet wandb

# --- launch training ---
python train.py "$CONFIG_PATH"

echo "==> training done; latest checkpoint in /scratch/m000204-pm06b/joana/runs/train_semantic_v1/"
