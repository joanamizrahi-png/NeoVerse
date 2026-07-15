#!/usr/bin/env bash
#SBATCH --job-name=freeze-park
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera/park-diffuse-slurm-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera/park-diffuse-slurm-%j.err

# Freeze-camera + diffusion on rugd_park-1_00 (natural / static scene).
# We saw driving.mp4's diffusion hallucinate parked cars driving off; this
# tests whether the same failure mode hits natural outdoor content or if
# it's a car-scene artifact of the training prior.

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera/diffused

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

python freeze_camera_diffuse.py \
    --input_path /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_park-1_00.mp4 \
    --output_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera/diffused \
    --freeze_frame 0 \
    --num_frames 81

# Rename output so it doesn't clobber the driving.mp4 version
mv /scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera/diffused/freeze_frame0_diffused.mp4 \
   /scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera/diffused/rugd_park-1_00_freeze_frame0_diffused.mp4

echo "==> RUGD park freeze+diffuse done"
