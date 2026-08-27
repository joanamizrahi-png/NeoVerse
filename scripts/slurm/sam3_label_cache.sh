#!/usr/bin/env bash
#SBATCH --job-name=sam3-cache
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=n04,n13,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-sam3-cache-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-sam3-cache-%j.err

# Re-label a ribbon cache with SAM3 run on the cache's OWN generated RGB.
# Builds sibling cache <CACHE>_<OUT_TAG> (rgb/alpha/manifest symlinked, only
# semantic_labels.npz replaced) -> OBS_CACHE=<sibling> runs are an exact A/B:
# co-generated semantics vs segment-the-generated-image.
#
#   CACHE   (default ribbon_cache_fan)
#   SCENE   (default rugd_trail_00)
#   OUT_TAG (default sam3)
#   CELLS   "a-b" manifest index range (optional; default = all cells)
#
#   sbatch --export=ALL,SCENE=rugd_trail_00 scripts/slurm/sam3_label_cache.sh

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/sam3/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

CACHE=${CACHE:-ribbon_cache_fan}
SCENE=${SCENE:-rugd_trail_00}
EXTRA=()
if [ -n "${CELLS:-}" ]; then
    EXTRA+=(--cells "$CELLS")
fi
python scripts/sam3_label_cache.py --cache "$CACHE" --scene "$SCENE" \
    --out_tag "${OUT_TAG:-sam3}" "${EXTRA[@]}"
echo "==> sibling cache ${CACHE}_${OUT_TAG:-sam3}/${SCENE} done"
