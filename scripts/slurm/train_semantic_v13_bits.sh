#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v13
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --exclude=n04,n13,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v13-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v13-%j.err

# v13 FULL (Track B): analog-bits semantic slot, 30 epochs (~13h at smoke
# pace of 2h03 / 5 epochs). Promoted from the 2026-08-17 smoke (+10.3 over
# the v10 recipe at equal budget). Judge criteria in the config header.

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v13_bits

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

# Stale-checkout guard: the config must carry the analog-bits knobs.
grep -q "semantic_analog_bits: true" training/configs/train_semantic_v13_bits.yaml && grep -q "snr_gamma: 5.0" training/configs/train_semantic_v13_bits.yaml && grep -q "num_epochs: 30" training/configs/train_semantic_v13_bits.yaml \
    || { echo "[sanity] FATAL: v13 full config missing the analog-bits knobs"; exit 1; }

python train.py training/configs/train_semantic_v13_bits.yaml

echo "==> v13 bits FULL done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v13_bits/"
