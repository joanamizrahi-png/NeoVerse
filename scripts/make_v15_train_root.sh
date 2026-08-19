#!/usr/bin/env bash
# Build combined_train_data_v15: v14 (RUGD minus held-out) + GND + SCAND
# (+ Go2W lab clips if data/go2w_clips exists). SpatialVID-shaped:
#   data/train/SpatialVID_HQ_metadata.csv, SpatialVid/HQ/<clip>/{mp4,caption.json}
# New clips must have SAM3 labels in outputs/sam3_labels_v14 (the training
# labels_dir) or they are SKIPPED with a warning. Idempotent: rebuilds from
# scratch every run.
set -euo pipefail
SCRATCH=/scratch/m000204-pm06b/joana
SRC=$SCRATCH/combined_train_data_v14
DST=$SCRATCH/combined_train_data_v15
LABELS_DIR=$SCRATCH/NeoVerse/outputs/sam3_labels_v14
META_CSV=$DST/data/train/SpatialVID_HQ_metadata.csv

rm -rf "$DST"
cp -as "$SRC" "$DST"
# the CSV must be a real file (we append rows), not a symlink into v14
rm "$META_CSV" && cp "$SRC/data/train/SpatialVID_HQ_metadata.csv" "$META_CSV"

count=0
skipped=0
for clips_dir in "$SCRATCH/data/gnd_clips" "$SCRATCH/data/scand_clips" "$SCRATCH/data/go2w_clips"; do
    [ ! -d "$clips_dir" ] && continue
    for f in "$clips_dir"/*.mp4; do
        [ ! -f "$f" ] && continue
        stem=$(basename "$f" .mp4)
        if [ ! -f "$LABELS_DIR/$stem.npz" ]; then
            echo "SKIP $stem (no v14 SAM3 label — run sam3_new_clips.sh first)"
            skipped=$((skipped + 1))
            continue
        fi
        clip_dir="$DST/SpatialVid/HQ/$stem"
        mkdir -p "$clip_dir"
        ln -sf "$f" "$clip_dir/$stem.mp4"
        case "$stem" in
            gnd_*)   desc="campus robot navigation along sidewalks with lawns, buildings, roads and pedestrians" ;;
            scand_*) desc="university campus walkway navigation among many pedestrians" ;;
            go2w_*)  desc="quadruped robot navigation around a university campus" ;;
            *)       desc="outdoor navigation scene" ;;
        esac
        printf '{\n    "SceneDescription": "%s"\n}\n' "$desc" > "$clip_dir/caption.json"
        echo "$stem,$stem/$stem.mp4,$stem,10000,30" >> "$META_CSV"
        count=$((count + 1))
    done
done
echo "==> $DST: added $count new clips (skipped $skipped), total rows: $(($(wc -l < "$META_CSV") - 1))"
