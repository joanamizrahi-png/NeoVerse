#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v24L
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --exclude=n04,n13,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v24L-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v24L-%j.err

# v24L: v24 warm-continued for 30 more epochs (equal-compute vs v21 test).

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v24L

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

# Stale-checkout + dataset guards.
grep -q "sanpo_v21" training/configs/train_semantic_v24L.yaml \
    || { echo "[sanity] FATAL: v24L config not pointing at sanpo_v21 roots"; exit 1; }
test -f /scratch/m000204-pm06b/joana/data/sanpo_v21/combined_train_data_v21/data/train/SpatialVID_HQ_metadata.csv \
    || { echo "[sanity] FATAL: metadata csv missing"; exit 1; }
test -f /scratch/m000204-pm06b/joana/runs/train_semantic_v24/checkpoint-epoch-30.safetensors \
    || { echo "[sanity] FATAL: v24 epoch-30 checkpoint missing"; exit 1; }

python train.py training/configs/train_semantic_v24L.yaml

echo "==> v24L done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v24L/"
