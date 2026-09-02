#!/usr/bin/env bash
#SBATCH --job-name=sam3-vs-gt
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=n04,n13,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-sam3-vs-gt-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-sam3-vs-gt-%j.err

# Per-class SAM3 accuracy against SANPO human GT, on the HELD-OUT val clips
# (convert_sanpo_val.py — last window per session, disjoint from training).
# SAM3 has been the world model's hint and the scene cloud's label source
# since the beginning and its per-class accuracy has never been measured.
# Knobs: NCLIPS (default 12 — ~5 min/clip).

set -euo pipefail
module load conda/24.3.0-0
export PATH=/users/jmizrahi/.conda/envs/sam3/bin:$PATH
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

VAL=/scratch/m000204-pm06b/joana/data/sanpo_val
if [ ! -d "$VAL/clips" ]; then
    echo "FATAL: no val set at $VAL/clips — run scripts/convert_sanpo_val.py first"
    exit 1
fi

mapfile -t CLIPS < <(find "$VAL/clips" -name "*.mp4" | sort)
NCLIPS=${NCLIPS:-12}
echo "val clips: ${#CLIPS[@]}, labelling ${NCLIPS}"

N=0
for CLIP in "${CLIPS[@]}"; do
    N=$((N + 1))
    if [ "$N" -gt "$NCLIPS" ]; then break; fi
    STEM=$(basename "$CLIP" .mp4)
    # sam3_precompute_labels.py writes outputs/sam3_labels/<stem>.npz — a
    # DIFFERENT directory from sam3_labels_v14, where convert_sanpo_val
    # symlinked the GT. No clash, and the per-clip skip keeps this resumable.
    if [ -f "outputs/sam3_labels/${STEM}.npz" ]; then
        echo "=== [$N] $STEM (cached)"
        continue
    fi
    echo "=== [$N/${NCLIPS}] $STEM"
    python sam3_precompute_labels.py --input_path "$CLIP" --num_frames 81
done

# sam3_precompute_labels.py emits RAW prompt indices. They must be remapped
# to v14 before they mean anything against SANPO GT — comparing the raw dir
# produced a confusion matrix that looked like total model failure (2026-09-01).
# Stage into a SEPARATE directory: remap writes <dir>_v14, and the real
# sam3_labels_v14/ holds convert_sanpo_val's GT symlinks, which must not be
# overwritten with predictions.
STAGE=outputs/sam3_val_raw
mkdir -p "$STAGE"
N=0
for CLIP in "${CLIPS[@]:0:$NCLIPS}"; do
    STEM=$(basename "$CLIP" .mp4)
    if [ -f "outputs/sam3_labels/${STEM}.npz" ]; then
        cp -f "outputs/sam3_labels/${STEM}.npz" "$STAGE/"
        N=$((N + 1))
    fi
done
echo "staged $N raw label files -> $STAGE"
python scripts/remap_labels_to_v14.py --dirs "$STAGE"

python scripts/sam3_vs_gt.py \
    --pred_dir "${STAGE}_v14" \
    --gt_dir "$VAL/labels" \
    --csv /scratch/m000204-pm06b/joana/outputs/SAM3_VS_GT_confusion.csv

echo "==> sam3 vs gt done"
