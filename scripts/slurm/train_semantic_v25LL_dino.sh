#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v25LL
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --exclude=n04,n06,n13,n14,n17,n24,n30
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v25LL-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v25LL-%j.err

# v25LL: v25-dino warm-continued 30 more epochs (companion to v24L).

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v25LL_dino

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

grep -q "dino_hint_channels: 384" training/configs/train_semantic_v25LL_dino.yaml \
    || { echo "[sanity] FATAL: v25LL config missing dino_hint_channels"; exit 1; }
grep -q "runs/train_semantic_v25L_dino/checkpoint-epoch-30" training/configs/train_semantic_v25LL_dino.yaml \
    || { echo "[sanity] FATAL: v25LL warm-start path wrong"; exit 1; }
test -f /scratch/m000204-pm06b/joana/runs/train_semantic_v25L_dino/checkpoint-epoch-30.safetensors \
    || { echo "[sanity] FATAL: v25 epoch-30 checkpoint missing"; exit 1; }
test -d "$HOME/.cache/torch/hub/facebookresearch_dinov2_main" \
    || { echo "[sanity] FATAL: DINOv2 hub cache missing"; exit 1; }

python train.py training/configs/train_semantic_v25LL_dino.yaml

echo "==> v25LL done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v25LL_dino/"
