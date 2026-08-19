#!/usr/bin/env bash
#SBATCH --job-name=freeze-diff
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera/diffuse-slurm-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera/diffuse-slurm-%j.err
#SBATCH --exclude=n04,n13,n17,n24

# Full-pipeline freeze camera: reconstructor + diffusion at fixed pose.
# Same idea as freeze_camera_demo.sh but with 4-step Wan diffusion applied,
# so the output is a CLEAN inpainted RGB (not the holey rasterizer output).

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera/diffused

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

python freeze_camera_diffuse.py \
    --input_path examples/videos/driving.mp4 \
    --output_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera/diffused \
    --freeze_frame 0 \
    --num_frames 81

echo "==> freeze camera + diffusion done"
