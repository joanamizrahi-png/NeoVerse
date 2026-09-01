#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v27S
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --exclude=n04,n13,n14,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v27S-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v27S-%j.err

# v27S: FROM-SCRATCH twin of v27 — palette 4, DINO into the semantic head
# only, no warm start. Pairs with v24 (scratch, no DINO) at the same level.
# ~7.5 h.

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v27S_dinosem

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

# Guards: must be SCRATCH (no warm start), must route to the semantic head,
# and dino_proj must be trainable (else the projection stays zero = no-op).
grep -q "^pretrained_path: null" \
    training/configs/train_semantic_v27S_dinosem.yaml \
    || { echo "[sanity] FATAL: v27S is not a scratch run"; exit 1; }
grep -q "dino_sem_head_only: true" \
    training/configs/train_semantic_v27S_dinosem.yaml \
    || { echo "[sanity] FATAL: v27S not routed to the semantic head"; exit 1; }
grep -q "dit.head.head.dino_proj" \
    training/configs/train_semantic_v27S_dinosem.yaml \
    || { echo "[sanity] FATAL: dino_proj missing from trainable_models"; exit 1; }
grep -q "train_semantic_v27S_dinosem" \
    training/configs/train_semantic_v27S_dinosem.yaml \
    || { echo "[sanity] FATAL: output path not v27S"; exit 1; }

python train.py training/configs/train_semantic_v27S_dinosem.yaml

echo "==> v27S done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v27S_dinosem/"
