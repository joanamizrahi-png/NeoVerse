#!/usr/bin/env bash
#SBATCH --job-name=freeze-cam
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera/slurm-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera/slurm-%j.err

# Run freeze_camera_demo.py on driving.mp4.
# Produces two MP4s:
#   orig_rgb.mp4    — camera follows original moving trajectory (baseline)
#   freeze_rgb.mp4  — camera fixed at frame 0, only timestamps advance
# If freeze_rgb.mp4 shows dynamic scene content (cars moving) with a stable
# viewpoint, NeoVerse's pose-vs-time decoupling works -> RL simulator viable.

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

# args: FREEZE_FRAME=0, N_FRAMES=24 (matches script defaults)
python freeze_camera_demo.py 0 24

echo "==> freeze_camera_demo done; outputs in outputs/freeze_camera/"
