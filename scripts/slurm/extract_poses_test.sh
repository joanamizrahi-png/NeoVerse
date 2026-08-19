#!/usr/bin/env bash
#SBATCH --job-name=extract-poses
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/m000204-pm06b/joana/NeoVerse/outputs/poses/slurm-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/NeoVerse/outputs/poses/slurm-%j.err
#SBATCH --exclude=n04,n13,n17,n24

# Run NeoVerse's reconstructor on the 4 test clips and save per-frame
# camera poses (position + heading + w2c + K) to outputs/poses/<stem>.npz.
# ~1-2 min per clip after model load. Uses the neoverse env (needs
# WorldMirror weights + PyTorch3D+gsplat).

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/NeoVerse/outputs/poses

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

MODEL_PATH=/scratch/m000204-pm06b/joana/NeoVerse/models
RECON_CKPT=/scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt

for CLIP_PATH in \
    /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_trail_00.mp4 \
    /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_creek_00.mp4 \
    /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_park-1_00.mp4 \
    /scratch/m000204-pm06b/joana/data/cityscapes_clips/cityscapes_stuttgart_00_00.mp4
do
    echo "==== $(basename $CLIP_PATH) ===="
    python scripts/extract_poses.py \
        --input_path "$CLIP_PATH" \
        --model_path "$MODEL_PATH" \
        --reconstructor_path "$RECON_CKPT" \
        --output_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/poses
done

echo "==> extract done; poses in outputs/poses/"
