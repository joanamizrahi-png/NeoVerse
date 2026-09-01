#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v24LL
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --exclude=n04,n13,n14,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v24LL-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v24LL-%j.err

# v24LL: v24L warm-continued 30 more epochs (90 total, NO DINO, palette 4).
# The no-DINO arm of the compute x DINO grid, and the control for the RGB
# degradation question. ~7.5 h.

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v24LL

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

# Stale-checkout + warm-start guards (the config-generator path-mangling trap
# bit v24L and v25LL before — verify the warm path IS v24L, not v24LL).
grep -q "train_semantic_v24L/checkpoint-epoch-30.safetensors" \
    training/configs/train_semantic_v24LL.yaml \
    || { echo "[sanity] FATAL: v24LL warm-start path wrong"; exit 1; }
grep -q "sanpo_v21" training/configs/train_semantic_v24LL.yaml \
    || { echo "[sanity] FATAL: v24LL config not pointing at sanpo_v21 roots"; exit 1; }
test -f /scratch/m000204-pm06b/joana/runs/train_semantic_v24L/checkpoint-epoch-30.safetensors \
    || { echo "[sanity] FATAL: v24L epoch-30 checkpoint missing"; exit 1; }

python train.py training/configs/train_semantic_v24LL.yaml

echo "==> v24LL done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v24LL/"
