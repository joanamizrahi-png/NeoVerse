#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v28b_campus_dino_seg
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# 2026-09-03: 24:00:00 was NEVER ENOUGH. Measured from job 461267: 26.5 s/iter
# x 287 iters = ~2h06m per epoch, so num_epochs 20 needs ~42 h. Job 460016
# (the v26 line) died TIMEOUT at exactly 1-00:00:06 having reached epoch 10 --
# that, not any failure, is why every downstream policy trains against v26
# EPOCH 10 instead of a finished model. Override lower only for a smoke test.
#SBATCH --time=48:00:00
#SBATCH --exclude=n04,n13,n14,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v28b_campus_dino_seg-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v28b_campus_dino_seg-%j.err

# v28_campus_dino: campus-only semantics (SANPO, no RUGD, ~2x clips). See the config
# header for why. Needs the v26 dataset build (job 459173) to have finished.

set -euo pipefail
mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v28b_campus_dino_seg
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

# Guards: campus-only dataset must EXIST and must contain no rugd clips.
CSV=/scratch/m000204-pm06b/joana/data/sanpo_v26/combined_train_data_v21/data/train/SpatialVID_HQ_metadata.csv
test -f "$CSV" || { echo "[sanity] FATAL: v26 dataset missing (is 459173 done?)"; exit 1; }
if grep -q "^rugd" "$CSV"; then
    echo "[sanity] FATAL: rugd clips leaked into the campus-only dataset"; exit 1
fi
echo "[sanity] campus-only clips: $(( $(wc -l < "$CSV") - 1 ))"
grep -q "sanpo_v26" training/configs/train_semantic_v28b_campus_dino_seg.yaml \
    || { echo "[sanity] FATAL: config not pointing at the v26 roots"; exit 1; }

NSEG=$(ls /scratch/m000204-pm06b/joana/NeoVerse/outputs/sam2_segments/sanpo_*.npz 2>/dev/null | wc -l)
echo "[sanity] sanpo sam2 segments: $NSEG"
test "$NSEG" -ge 200 || { echo "[sanity] FATAL: campus segments missing — this relaunch exists ONLY to enable the seg loss"; exit 1; }
python train.py training/configs/train_semantic_v28b_campus_dino_seg.yaml
echo "==> v28_campus_dino done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v28b_campus_dino_seg/"
