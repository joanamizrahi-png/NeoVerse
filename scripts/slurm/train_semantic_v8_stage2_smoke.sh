#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v8-stage2-smoke
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v8-stage2-smoke-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v8-stage2-smoke-%j.err

# v8 stage-1 SMOKE: 1 epoch of the x0-prediction recipe (~25-30 min + load).
# Pass criteria are in the config header (train_semantic_v8_stage2_smoke.yaml).
# Logs go to the joana/ root (always exists) — SLURM won't create log dirs.

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v8_stage2_smoke

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

echo "hostname: $(hostname)"
echo "commit: $(git log --oneline -1)"
python -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')"

GT_DIR=/scratch/m000204-pm06b/joana/NeoVerse/outputs/rugd_gt_labels
N_GT=$(ls "$GT_DIR"/*.npz 2>/dev/null | wc -l)
echo "[sanity] $N_GT GT label files"
if [ "$N_GT" -lt 10 ]; then
    echo "[sanity] FATAL: <10 GT label npz in $GT_DIR"
    exit 1
fi

# Sanity: the config must actually carry the v8 flag — a stale checkout here
# would smoke-test v7's objective and read as a false PASS.
grep -q "semantic_x0_prediction: true" training/configs/train_semantic_v8_stage2_smoke.yaml && grep -q "semantic_ce_weight: 0.1" training/configs/train_semantic_v8_stage2_smoke.yaml \
    || { echo "[sanity] FATAL: smoke config missing semantic_x0_prediction: true"; exit 1; }

python train.py training/configs/train_semantic_v8_stage2_smoke.yaml

echo "==> v8 smoke done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v8_stage2_smoke/"
