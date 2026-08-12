#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v10x
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v10x-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v10x-%j.err

# v10 FULL RUN: 30 epochs, pres+snr+cegate recipe (~30-34h with the preservation second-forward).
# Pass criteria are in the config header (train_semantic_v10${VAR}.yaml).
# Logs go to the joana/ root (always exists) — SLURM won't create log dirs.

set -euo pipefail
VAR=${VAR:?set VAR=b or c via --export}

mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v10${VAR}

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
grep -q "snr_gamma: 5.0" training/configs/train_semantic_v10${VAR}.yaml && grep -q "rgb_preservation_weight:" training/configs/train_semantic_v10${VAR}.yaml \
    || { echo "[sanity] FATAL: v10${VAR} config missing snr/pres knobs"; exit 1; }

python train.py training/configs/train_semantic_v10${VAR}.yaml

echo "==> v10 full run done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v10${VAR}/"
