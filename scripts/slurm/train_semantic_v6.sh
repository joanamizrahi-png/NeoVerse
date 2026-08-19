#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v6
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/runs/train_semantic_v6/slurm-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/runs/train_semantic_v6/slurm-%j.err
#SBATCH --exclude=n04,n13,n17,n24

# v6: parallel _sem submodules (full-rank semantic pathway) + distill LoRA merged
# at TRAIN time (same 4-step regime as inference). No v3 warm-start (stale
# control_branch). ~46 clips x 20 epochs x ~5 s/step -> similar wall clock to v5.

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v6

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=warning

cd /scratch/m000204-pm06b/joana/NeoVerse

echo "hostname: $(hostname)"
echo "which python: $(which python)"
python -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')"
nvidia-smi | head -20

CONFIG_PATH="training/configs/train_semantic_v6.yaml"

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
    sys.exit(f"[sanity] could not parse train_dataset:\n  {dataset_line}")
ROOT = m_root.group(1)
LABELS_DIR = m_labels.group(1)
print(f"[sanity] config: {config_path}")
print(f"[sanity] ROOT       = {ROOT}")
print(f"[sanity] LABELS_DIR = {LABELS_DIR}")

pretrained = cfg.get("pretrained_path")
if pretrained:
    if not os.path.exists(pretrained):
        sys.exit(f"[sanity] warm-start checkpoint missing: {pretrained}")
    print(f"[sanity] warm-start from: {pretrained}")
else:
    print("[sanity] fresh start (no warm-start)")

distill = cfg.get("distill_lora_path")
if distill:
    if not os.path.exists(distill):
        sys.exit(f"[sanity] distill LoRA missing: {distill}")
    print(f"[sanity] distill LoRA will be merged at pipeline load: {distill}")

exp_v = cfg.get("semantic_expansion_version", 1)
print(f"[sanity] semantic_expansion_version = {exp_v} (2 = v6 _sem split)")

meta = pd.read_csv(os.path.join(ROOT, "data/train/SpatialVID_HQ_metadata.csv"))
print(f"[sanity] {len(meta)} clips in metadata csv")

ok = 0
missing = 0
for _, row in meta.iterrows():
    vp = os.path.join(ROOT, "SpatialVid/HQ", row["video path"])
    lp = os.path.join(LABELS_DIR, f"{row['id']}.npz")
    if not os.path.exists(vp) or not os.path.exists(lp):
        missing += 1
        continue
    n_v = len(VideoReader(vp))
    n_l = np.load(lp)["labels"].shape[0]
    if abs(n_v - n_l) > 3:
        missing += 1
        continue
    ok += 1
print(f"[sanity] {ok} clips have matching video + label; {missing} missing/misaligned")
assert ok > 0, "no valid clips"
PY

python -c "import wandb" 2>/dev/null || python -m pip install --quiet wandb

python train.py "$CONFIG_PATH"

echo "==> v6 training done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v6/"
