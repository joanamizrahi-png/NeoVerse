#!/usr/bin/env bash
# Build combined_train_data_v18 + gt_labels_v18 for the v18 semantics run.
#
# v18 = v15's clip set + the Cityscapes Stuttgart clips, and ONE combined
# dense-target dir: RUGD human GT + SegFormer pseudo-GT for campus clips
# (scripts/segmenter_labels.py must have run first). With gt_only on, any
# clip absent from gt_labels_v18 trains appearance only — safe default.
# Idempotent: rebuilds from scratch every run.
set -euo pipefail
SCRATCH=/scratch/m000204-pm06b/joana
SRC=$SCRATCH/combined_train_data_v15
DST=$SCRATCH/combined_train_data_v18
LABELS_DIR=$SCRATCH/NeoVerse/outputs/sam3_labels_v14
SEGF_DIR=$SCRATCH/NeoVerse/outputs/segformer_gt_labels_v14
RUGD_GT=$SCRATCH/NeoVerse/outputs/rugd_gt_labels_v14
GT18=$SCRATCH/NeoVerse/outputs/gt_labels_v18
META_CSV=$DST/data/train/SpatialVID_HQ_metadata.csv

[ -d "$SEGF_DIR" ] || { echo "FATAL: $SEGF_DIR missing — run segmenter_labels.sh first"; exit 1; }

rm -rf "$DST"
cp -as "$SRC" "$DST"
rm "$META_CSV" && cp "$SRC/data/train/SpatialVID_HQ_metadata.csv" "$META_CSV"

count=0
for f in "$SCRATCH/data/cityscapes_clips"/*.mp4; do
    stem=$(basename "$f" .mp4)
    if [ ! -f "$LABELS_DIR/$stem.npz" ]; then
        echo "SKIP $stem (no v14 SAM3 hint label — extend sam3_new_clips.sh to cityscapes_clips)"
        continue
    fi
    clip_dir="$DST/SpatialVid/HQ/$stem"
    mkdir -p "$clip_dir"
    ln -sf "$f" "$clip_dir/$stem.mp4"
    printf '{\n    "SceneDescription": "%s"\n}\n' \
        "driving through a european city with roads, sidewalks, cars and pedestrians" \
        > "$clip_dir/caption.json"
    echo "$stem,$stem/$stem.mp4,$stem,10000,30" >> "$META_CSV"
    count=$((count + 1))
done

# Combined dense-target dir: RUGD human GT wins on name collisions (there are
# none by construction — RUGD stems vs campus stems are disjoint).
rm -rf "$GT18"
mkdir -p "$GT18"
ln -sf "$SEGF_DIR"/*.npz "$GT18"/ 2>/dev/null || true
ln -sf "$RUGD_GT"/*.npz "$GT18"/
echo "==> $DST: +$count cityscapes clips, rows: $(($(wc -l < "$META_CSV") - 1))"
echo "==> $GT18: $(ls "$GT18" | wc -l) dense-target npzs (RUGD GT + SegFormer)"
