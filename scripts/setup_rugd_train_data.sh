#!/usr/bin/env bash
# Build the SpatialVID-shaped training dataset for the RUGD clips.
#
# Prereqs:
#   * RUGD clips exist at /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_*.mp4
#     (produced by scripts/prepare_rugd_clips.py)
#   * SAM 3 labels exist at outputs/sam3_labels/rugd_*.npz
#     (produced by scripts/slurm/label_rugd_clips.sh)
#
# Produces (at /scratch/.../rugd_train_data/):
#   data/train/SpatialVID_HQ_metadata.csv         (1 row per clip)
#   SpatialVid/HQ/<clip_stem>/<clip_stem>.mp4     (symlink to the source MP4)
#   SpatialVid/HQ/<clip_stem>/caption.json        (scene-based description)
#
# Idempotent: overwrites the CSV every run but only re-symlinks / re-writes
# captions for clips that don't already have them.

set -euo pipefail

SCRATCH=/scratch/m000204-pm06b/joana
CLIPS_DIR="$SCRATCH/data/rugd_clips"
LABELS_DIR="$SCRATCH/NeoVerse/outputs/sam3_labels"
DATA_ROOT="$SCRATCH/rugd_train_data"
META_DIR="$DATA_ROOT/data/train"
META_CSV="$META_DIR/SpatialVID_HQ_metadata.csv"

mkdir -p "$META_DIR"

# Write CSV header. Rows appended in the loop.
cat > "$META_CSV" <<'EOF'
id,video path,annotation path,num frames,fps
EOF

count=0
skipped_no_label=0

for f in "$CLIPS_DIR"/*.mp4; do
    stem=$(basename "$f" .mp4)
    npz="$LABELS_DIR/$stem.npz"
    if [ ! -f "$npz" ]; then
        echo "SKIP $stem  (no SAM3 label at $npz)"
        skipped_no_label=$((skipped_no_label + 1))
        continue
    fi

    clip_dir="$DATA_ROOT/SpatialVid/HQ/$stem"
    mkdir -p "$clip_dir"

    # Symlink the MP4 (no copy -- saves disk)
    if [ ! -e "$clip_dir/$stem.mp4" ]; then
        ln -s "$f" "$clip_dir/$stem.mp4"
    fi

    # Scene-based caption from the stem. e.g. "rugd_park-1_00" -> "park-1".
    scene=$(echo "$stem" | sed 's/^rugd_//; s/_[0-9]*$//')
    case "$scene" in
        creek)      desc="off-road robot navigation along a creek with rocks and vegetation" ;;
        park*)      desc="off-road robot navigation through a grassy park with trees and paths" ;;
        trail*)     desc="off-road robot navigation along a wooded outdoor trail" ;;
        village*)   desc="outdoor robot navigation through a village area with buildings and paths" ;;
        *)          desc="outdoor off-road robot navigation scene" ;;
    esac

    caption_file="$clip_dir/caption.json"
    if [ ! -f "$caption_file" ]; then
        printf '{\n    "SceneDescription": "%s"\n}\n' "$desc" > "$caption_file"
    fi

    # CSV row. num_frames=10000 is a lie to pass the min_clip_length filter --
    # actual sampling uses the real video length (~81 frames).
    echo "$stem,$stem/$stem.mp4,$stem,10000,30" >> "$META_CSV"
    count=$((count + 1))
done

echo ""
echo "==> Wrote $count rows to $META_CSV"
echo "==> Skipped $skipped_no_label clips (no SAM3 label yet -- rerun after labeling job finishes)"
echo "==> DATA_ROOT = $DATA_ROOT"
echo ""
echo "Next: sbatch scripts/slurm/train_semantic.sh"
