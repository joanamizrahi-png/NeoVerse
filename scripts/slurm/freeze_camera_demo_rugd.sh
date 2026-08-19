#!/usr/bin/env bash
#SBATCH --job-name=freeze-park-nodiff
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm_logs/freeze-park-nodiff-slurm-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm_logs/freeze-park-nodiff-slurm-%j.err
#SBATCH --exclude=n04,n13,n17,n24

# Rasterizer-only freeze camera on rugd_park-1_00. Same as freeze_camera_demo.py
# on driving.mp4 but for a natural outdoor scene. Produces:
#   orig_rgb.mp4    — camera follows the original moving path (baseline)
#   freeze_rgb.mp4  — camera fixed at frame 0, only timestamps advance
# Both non-diffused (holey rasterizer output). Comparison against the diffused
# version from freeze_camera_diffuse.py shows what the diffusion contributes.

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/slurm_logs
mkdir -p /scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera_rugd_park

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

# freeze_camera_demo.py has INPUT hardcoded to examples/videos/driving.mp4.
# Symlink our RUGD clip in temporarily so the script uses it without editing.
# The demo also uses N_FRAMES=24 by default (works, 12 was too aggressive).
TMP_LINK=examples/videos/driving.mp4
BACKUP=examples/videos/driving.mp4.backup
if [ -e "$TMP_LINK" ] && [ ! -L "$TMP_LINK" ]; then
    mv "$TMP_LINK" "$BACKUP"
fi
ln -sf /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_park-1_00.mp4 "$TMP_LINK"

python freeze_camera_demo.py 0 12

# Move outputs to a park-specific dir so they don't collide with the driving version.
mkdir -p /scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera_rugd_park
mv /scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera/orig_rgb.mp4 \
   /scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera_rugd_park/orig_rgb.mp4
mv /scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera/freeze_rgb.mp4 \
   /scratch/m000204-pm06b/joana/NeoVerse/outputs/freeze_camera_rugd_park/freeze_rgb.mp4

# Restore driving.mp4 if we backed it up.
rm "$TMP_LINK"
if [ -e "$BACKUP" ]; then
    mv "$BACKUP" "$TMP_LINK"
fi

echo "==> RUGD park rasterizer-only freeze done"
