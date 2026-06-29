#!/usr/bin/env bash
# Run sam3_precompute_labels.py on every .mp4 in examples/videos/training_pilot/.
# Outputs go to outputs/sam3_labels/<clip_name>.npz + overlay images.
#
# Usage:
#   bash scripts/run_sam3_on_pilot.sh
#
# Each clip takes ~1.5 min on the Jetson (~1 sec / frame, 81 frames).
# Skips clips that already have a saved .npz.

set -e
IN_DIR="examples/videos/training_pilot"
OUT_DIR="outputs/sam3_labels"

if [ ! -d "$IN_DIR" ]; then
    echo "Missing $IN_DIR — run scripts/download_training_clips.sh first."
    exit 1
fi

count=0
skipped=0
for f in "$IN_DIR"/*.mp4; do
    stem=$(basename "$f" .mp4)
    out_npz="$OUT_DIR/${stem}.npz"
    if [ -f "$out_npz" ]; then
        echo "SKIP $stem (already labeled)"
        skipped=$((skipped + 1))
        continue
    fi
    echo ""
    echo "==> Labeling $stem"
    python sam3_precompute_labels.py --input_path "$f"
    count=$((count + 1))
done

echo ""
echo "Done. Labeled $count new clips, skipped $skipped already-done."
echo "Outputs in $OUT_DIR/"
