#!/usr/bin/env bash
#SBATCH --job-name=relabel-all
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels/relabel-all-slurm-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels/relabel-all-slurm-%j.err
#SBATCH --exclude=n04,n13,n17,n24

# Relabel ALL 46 training clips with the SAM3 priority-order fix.
# Overwrites existing outputs/sam3_labels/*.npz. ~1 min per clip -> ~50 min total.
# Overshoot to 2h in case any single clip is slow.

set -euo pipefail

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/sam3/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

# Iterate every mp4 in the two clip directories (RUGD + Cityscapes).
for CLIPS_DIR in \
    /scratch/m000204-pm06b/joana/data/rugd_clips \
    /scratch/m000204-pm06b/joana/data/cityscapes_clips
do
    [ ! -d "$CLIPS_DIR" ] && continue
    for CLIP_PATH in "$CLIPS_DIR"/*.mp4; do
        [ ! -f "$CLIP_PATH" ] && continue
        echo "==== $(basename "$CLIP_PATH") ===="
        python sam3_precompute_labels.py --input_path "$CLIP_PATH"
    done
done

echo "==> full relabel done"
