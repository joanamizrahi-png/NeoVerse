#!/usr/bin/env bash
#SBATCH --job-name=sam3-new
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=n04,n13,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-sam3-new-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-sam3-new-%j.err

# SAM3-label the NEW dataset clips (GND / SCAND / lab bags) and refresh the
# v14 remap. ~1 min per clip.

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/sam3/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

for CLIPS_DIR in \
    /scratch/m000204-pm06b/joana/data/gnd_clips \
    /scratch/m000204-pm06b/joana/data/scand_clips \
    /scratch/m000204-pm06b/joana/data/go2w_clips
do
    [ ! -d "$CLIPS_DIR" ] && continue
    for CLIP_PATH in "$CLIPS_DIR"/*.mp4; do
        [ ! -f "$CLIP_PATH" ] && continue
        # pano-view mp4s are reconstruction anchors, not training clips
        [[ "$CLIP_PATH" == *_pano_yaw* ]] && continue
        echo "==== $(basename "$CLIP_PATH") ===="
        python sam3_precompute_labels.py --input_path "$CLIP_PATH"
    done
done

# regenerate the v14 siblings (name-based remap, covers everything)
/users/jmizrahi/.conda/envs/neoverse/bin/python scripts/remap_labels_to_v14.py \
    --dirs outputs/sam3_labels

echo "==> new-clip SAM3 labels + v14 remap done"
