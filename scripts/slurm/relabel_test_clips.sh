#!/usr/bin/env bash
#SBATCH --job-name=relabel-4
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels/relabel-slurm-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels/relabel-slurm-%j.err
#SBATCH --exclude=n04,n13,n17,n24

# Re-run SAM3 labeling on the 4 test clips after the priority-order fix.
# ~1 min per clip. Uses the sam3 conda env (not neoverse -- different transformers version).

set -euo pipefail

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/sam3/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

for CLIP_PATH in \
    /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_trail_00.mp4 \
    /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_creek_00.mp4 \
    /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_park-1_00.mp4 \
    /scratch/m000204-pm06b/joana/data/cityscapes_clips/cityscapes_stuttgart_00_00.mp4
do
    echo "==== $(basename $CLIP_PATH) ===="
    python sam3_precompute_labels.py --input_path "$CLIP_PATH"
done

echo "==> relabel done; new npz files in outputs/sam3_labels/"
