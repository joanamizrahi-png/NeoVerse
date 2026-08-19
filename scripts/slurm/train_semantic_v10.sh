#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v10
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v10-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v10-%j.err
#SBATCH --exclude=n04,n13,n17,n24

# v10 FULL RUN: 30 epochs, pres+snr+cegate recipe (~30-34h with the preservation second-forward).
# Pass criteria are in the config header (train_semantic_v10.yaml).
# Logs go to the joana/ root (always exists) — SLURM won't create log dirs.

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v10

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

echo "hostname: $(hostname)"
echo "commit: $(git log --oneline -1)"
python -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')"

GT_DIR=/scratch/m000204-pm06b/joana/NeoVerse/outputs/rugd_gt_labels_v14
N_GT=$(ls "$GT_DIR"/*.npz 2>/dev/null | wc -l)
echo "[sanity] $N_GT GT label files"
if [ "$N_GT" -lt 10 ]; then
    echo "[sanity] FATAL: <10 GT label npz in $GT_DIR"
    exit 1
fi

# Sanity: the config must actually carry the v8 flag — a stale checkout here
# would smoke-test v7's objective and read as a false PASS.
grep -q "rgb_preservation_weight: 1.0" training/configs/train_semantic_v10.yaml && grep -q "snr_gamma: 5.0" training/configs/train_semantic_v10.yaml && grep -q "semantic_ce_sigma_max: 1.0" training/configs/train_semantic_v10.yaml \
    || { echo "[sanity] FATAL: v10 config missing the pres+snr+cegate knobs"; exit 1; }

python train.py training/configs/train_semantic_v10.yaml

echo "==> v10 full run done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v10/"
