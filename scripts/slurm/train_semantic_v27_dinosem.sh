#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v27
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --exclude=n04,n13,n14,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v27-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v27-%j.err

# v27: DINO hint into the SEMANTIC HEAD ONLY, warm from v24L (RGB-clean).
# The salvage attempt for the DINO idea after the v25 shared-embedding wiring
# was measured to destroy RGB. ~7.5 h.

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v27_dinosem

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

# Guards: warm path must be v24L (not v27 itself), the sem-head route must be
# ON (else this silently repeats the v25 mistake), and dino_proj must be in
# trainable_models (else the projection stays zero and the run is a no-op).
grep -q "train_semantic_v24L/checkpoint-epoch-30.safetensors" \
    training/configs/train_semantic_v27_dinosem.yaml \
    || { echo "[sanity] FATAL: v27 warm-start path wrong"; exit 1; }
grep -q "dino_sem_head_only: true" \
    training/configs/train_semantic_v27_dinosem.yaml \
    || { echo "[sanity] FATAL: v27 not routed to the semantic head"; exit 1; }
grep -q "dit.head.head.dino_proj" \
    training/configs/train_semantic_v27_dinosem.yaml \
    || { echo "[sanity] FATAL: dino_proj missing from trainable_models"; exit 1; }
test -f /scratch/m000204-pm06b/joana/runs/train_semantic_v24L/checkpoint-epoch-30.safetensors \
    || { echo "[sanity] FATAL: v24L epoch-30 checkpoint missing"; exit 1; }

python train.py training/configs/train_semantic_v27_dinosem.yaml

echo "==> v27 done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v27_dinosem/"
