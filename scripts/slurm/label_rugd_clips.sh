#!/usr/bin/env bash
#SBATCH --job-name=label-rugd
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/data/rugd_clips/label-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/data/rugd_clips/label-%j.err

# Batch-label every RUGD MP4 clip in /scratch/.../rugd_clips/ with SAM 3.
# Idempotent: skips clips that already have a .npz in outputs/sam3_labels/.
# Uses --static_scene so every frame gets labeled (matches training dataloader's
# random sampling requirement).

set -euo pipefail

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/sam3/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

IN_DIR="/scratch/m000204-pm06b/joana/data/rugd_clips"
OUT_DIR="outputs/sam3_labels"
mkdir -p "$OUT_DIR"

total=$(ls -1 "$IN_DIR"/*.mp4 2>/dev/null | wc -l)
echo "hostname: $(hostname)"
echo "==> $total MP4s in $IN_DIR"
echo "==> Output .npz will land in $(pwd)/$OUT_DIR/"
echo ""

count=0
skipped=0
t_start=$(date +%s)

for f in "$IN_DIR"/*.mp4; do
    stem=$(basename "$f" .mp4)
    out_npz="$OUT_DIR/${stem}.npz"
    if [ -f "$out_npz" ]; then
        echo "SKIP $stem  (already labeled)"
        skipped=$((skipped + 1))
        continue
    fi
    echo ""
    echo "==> [$((count + 1))/$total] Labeling $stem"
    python sam3_precompute_labels.py \
        --input_path "$f" \
        --static_scene \
        --overlay_every 40
    count=$((count + 1))
done

elapsed=$(( $(date +%s) - t_start ))
echo ""
echo "==> Done in ${elapsed}s. Labeled $count new clips, skipped $skipped already-done."
echo "    Outputs in $(pwd)/$OUT_DIR/"
