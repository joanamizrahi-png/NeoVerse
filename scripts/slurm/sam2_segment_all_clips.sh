#!/usr/bin/env bash
#SBATCH --job-name=sam2-segments
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-sam2-segments-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-sam2-segments-%j.err

# v8 Change 3 preprocessing: SAM2 class-agnostic segments for every training
# clip. Idempotent (per-clip skip if npz exists) — safe to resubmit after a
# timeout. Runs in the sam3 conda env (transformers), like the SAM3 labeler.
# Budget: ~3-7 min/clip on H100 -> ~2.5-5 h for ~44 clips.

set -euo pipefail

module load conda/24.3.0-0
export PATH=/users/jmizrahi/.conda/envs/sam3/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

CLIPS_ROOT=/scratch/m000204-pm06b/joana/combined_train_data
N=0
for CLIP_PATH in "$CLIPS_ROOT"/*.mp4; do
    N=$((N + 1))
    echo "=== [$N] $(basename "$CLIP_PATH") ==="
    python sam2_precompute_segments.py --input_path "$CLIP_PATH"
done

echo "==> sam2 segments done: $N clips -> outputs/sam2_segments/"
ls outputs/sam2_segments/*.npz | wc -l
