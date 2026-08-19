#!/usr/bin/env bash
#SBATCH --job-name=poses-more
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/NeoVerse/outputs/poses/more-slurm-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/NeoVerse/outputs/poses/more-slurm-%j.err
#SBATCH --exclude=n04,n13,n17,n24

# Extract per-frame camera poses for a diverse subset of clips to verify our
# "y is the up axis" hypothesis. If ALL of these show y as the smallest-range
# axis and the trajectory tracks visible motion in xz, we can rely on it for
# Milestone B. If any clip shows y as a large-range axis, we need per-scene
# ground-plane detection.

set -euo pipefail

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

MODEL_PATH=/scratch/m000204-pm06b/joana/NeoVerse/models
RECON_CKPT=/scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt

# Diverse mix: 4 more RUGD (different scenes), 4 more Cityscapes clips.
CLIPS=(
    /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_creek_01.mp4
    /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_park-2_00.mp4
    /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_park-8_00.mp4
    /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_village_00.mp4
    /scratch/m000204-pm06b/joana/data/cityscapes_clips/cityscapes_stuttgart_00_01.mp4
    /scratch/m000204-pm06b/joana/data/cityscapes_clips/cityscapes_stuttgart_00_02.mp4
    /scratch/m000204-pm06b/joana/data/cityscapes_clips/cityscapes_stuttgart_01_00.mp4
    /scratch/m000204-pm06b/joana/data/cityscapes_clips/cityscapes_stuttgart_02_00.mp4
)

for CLIP_PATH in "${CLIPS[@]}"; do
    if [ ! -f "$CLIP_PATH" ]; then
        echo "SKIP (not found): $CLIP_PATH"
        continue
    fi
    echo "==== $(basename $CLIP_PATH) ===="
    python scripts/extract_poses.py \
        --input_path "$CLIP_PATH" \
        --model_path "$MODEL_PATH" \
        --reconstructor_path "$RECON_CKPT" \
        --output_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/poses
done

echo "==> extract-more done; new poses in outputs/poses/"
