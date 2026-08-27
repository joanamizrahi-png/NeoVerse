#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v20
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --exclude=n04,n13,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v20-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v20-%j.err

# v20: reliable-class masked distillation (see config header). Same diet and
# budget as v19b; the ONLY change is the pseudo-GT loss mask.

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v20

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

# Stale-checkout guard: the config must carry the v20 mask knobs.
grep -q "pseudo_gt_reliable_classes" training/configs/train_semantic_v20.yaml \
    && grep -q "v20_reliable_mask" training/configs/train_semantic_v20.yaml \
    || { echo "[sanity] FATAL: v20 config missing the mask knobs"; exit 1; }

python train.py training/configs/train_semantic_v20.yaml

echo "==> v20 done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v20/"
