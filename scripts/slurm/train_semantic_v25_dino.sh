#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v25
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --exclude=n04,n06,n13,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v25-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v25-%j.err

# v25-dino: v24 recipe + frozen-DINOv2 hint into the control branch.

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v25_dino

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

# Stale-checkout + dataset + DINO-cache guards.
grep -q "sanpo_v21" training/configs/train_semantic_v25_dino.yaml \
    || { echo "[sanity] FATAL: v25 config not pointing at sanpo_v21 roots"; exit 1; }
grep -q "dino_hint_channels: 384" training/configs/train_semantic_v25_dino.yaml \
    || { echo "[sanity] FATAL: v25 config missing dino_hint_channels"; exit 1; }
grep -q "dino_proj" training/configs/train_semantic_v25_dino.yaml \
    || { echo "[sanity] FATAL: v25 trainable_models missing dino_proj"; exit 1; }
test -f /scratch/m000204-pm06b/joana/data/sanpo_v21/combined_train_data_v21/data/train/SpatialVID_HQ_metadata.csv \
    || { echo "[sanity] FATAL: v25 metadata csv missing (run convert_sanpo)"; exit 1; }
test -d "$HOME/.cache/torch/hub/facebookresearch_dinov2_main" \
    || { echo "[sanity] FATAL: DINOv2 hub cache missing; on a login node run: python -c \"import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')\""; exit 1; }

python train.py training/configs/train_semantic_v25_dino.yaml

echo "==> v25-dino done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v25_dino/"
