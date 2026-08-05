#!/usr/bin/env bash
#SBATCH --job-name=inf-vanilla
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-inf-vanilla-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-inf-vanilla-%j.err

# VANILLA NeoVerse render (pretrained weights, no semantic finetune) along a
# chosen trajectory — the RGB the navigation policy would actually be fed
# under the two-pass scheme. Twin of inference_v8_run.sh for A/B comparison:
#   sbatch --export=CLIP=rugd_trail-6_01,TRAJ=move_left scripts/slurm/inference_vanilla_traj.sh

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"
CLIP=${CLIP:-rugd_trail_00}
TRAJ=${TRAJ:-static}
MAG=""
TRAJARGS=""
if [ -n "${ANGLE:-}" ]; then TRAJARGS="$TRAJARGS --angle $ANGLE"; MAG="${MAG}_a${ANGLE}"; fi
if [ -n "${DIST:-}" ]; then TRAJARGS="$TRAJARGS --distance $DIST"; MAG="${MAG}_d${DIST}"; fi
OUT=/scratch/m000204-pm06b/joana/inference_VANILLA_${CLIP}_${TRAJ}${MAG}
mkdir -p "$OUT"
python inference.py \
    --input_path /scratch/m000204-pm06b/joana/data/rugd_clips/${CLIP}.mp4 \
    --trajectory "$TRAJ" $TRAJARGS \
    --output_path "$OUT/rgb_vanilla.mp4" \
    --model_path /scratch/m000204-pm06b/joana/NeoVerse/models \
    --reconstructor_path /scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt
echo "==> vanilla render done: $OUT"
