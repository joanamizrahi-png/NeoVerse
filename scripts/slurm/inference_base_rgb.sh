#!/usr/bin/env bash
#SBATCH --job-name=inf-base
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --exclude=n04,n06,n13,n14,n17,n21,n24,n26,n30,n31
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-inf-base-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-inf-base-%j.err

# BASE-model RGB render (2026-08-31, her control question: "does the world
# model hallucinate like this even without the semantics head?"). Runs the
# PRETRAINED NeoVerse pipeline — no semantic expansion, no finetune LoRA —
# on the same clip/trajectory as our semantic renders, so the comparison
# isolates what our fine-tune adds vs what the base model already does.
# Knobs: CLIP (basename, no .mp4), CLIPS_DIR, TRAJ (default static).

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

CLIP=${CLIP:-rugd_trail_00}
CLIPS_DIR=${CLIPS_DIR:-/scratch/m000204-pm06b/joana/data/rugd_clips}
TRAJ=${TRAJ:-static}
OUT=/scratch/m000204-pm06b/joana/inference_BASE_${CLIP}_${TRAJ}
mkdir -p "$OUT"

python inference.py \
    --trajectory "$TRAJ" \
    --input_path "$CLIPS_DIR/${CLIP}.mp4" \
    --output_path "$OUT/rgb_base.mp4" \
    --height 336 --width 560 --num_frames 81

echo "==> base RGB render done: $OUT/rgb_base.mp4"
