#!/usr/bin/env bash
# Build the SpatialVID-shaped training dataset combining RUGD (off-road) + Cityscapes (urban).
#
# Prereqs:
#   * RUGD clips at /scratch/.../data/rugd_clips/rugd_*.mp4
#   * Cityscapes clips at /scratch/.../data/cityscapes_clips/cityscapes_*.mp4
#   * SAM 3 labels at outputs/sam3_labels/{rugd,cityscapes}_*.npz
#
# Produces at /scratch/.../combined_train_data/:
#   data/train/SpatialVID_HQ_metadata.csv        (one row per clip)
#   SpatialVid/HQ/<clip>/<clip>.mp4               (symlink)
#   SpatialVid/HQ/<clip>/caption.json             (scene-based description)
#
# Idempotent: rewrites the CSV every run, symlinks + captions skipped if present.

set -euo pipefail

SCRATCH=/scratch/m000204-pm06b/joana
LABELS_DIR="$SCRATCH/NeoVerse/outputs/sam3_labels"
DATA_ROOT="$SCRATCH/combined_train_data"
META_DIR="$DATA_ROOT/data/train"
META_CSV="$META_DIR/SpatialVID_HQ_metadata.csv"

mkdir -p "$META_DIR"

cat > "$META_CSV" <<'EOF'
id,video path,annotation path,num frames,fps
EOF

count=0
skipped_no_label=0

# Walk BOTH RUGD and Cityscapes clip directories
for clips_dir in "$SCRATCH/data/rugd_clips" "$SCRATCH/data/cityscapes_clips"; do
    [ ! -d "$clips_dir" ] && continue

    for f in "$clips_dir"/*.mp4; do
        [ ! -f "$f" ] && continue
        stem=$(basename "$f" .mp4)
        npz="$LABELS_DIR/$stem.npz"
        if [ ! -f "$npz" ]; then
            echo "SKIP $stem  (no SAM3 label at $npz)"
            skipped_no_label=$((skipped_no_label + 1))
            continue
        fi

        clip_dir="$DATA_ROOT/SpatialVid/HQ/$stem"
        mkdir -p "$clip_dir"
        if [ ! -e "$clip_dir/$stem.mp4" ]; then
            ln -s "$f" "$clip_dir/$stem.mp4"
        fi

        # Caption per source + scene family
        case "$stem" in
            rugd_creek*)      desc="off-road robot navigation along a creek with rocks and vegetation" ;;
            rugd_park*)       desc="off-road robot navigation through a grassy park with trees and paths" ;;
            rugd_trail*)      desc="off-road robot navigation along a wooded outdoor trail" ;;
            rugd_village*)    desc="off-road robot navigation through a small village area" ;;
            rugd_*)           desc="outdoor off-road robot navigation scene" ;;
            cityscapes_*)     desc="urban driving scene through Stuttgart streets with buildings, roads, sidewalks, cars, and pedestrians" ;;
            *)                desc="outdoor navigation scene" ;;
        esac

        caption_file="$clip_dir/caption.json"
        if [ ! -f "$caption_file" ]; then
            printf '{\n    "SceneDescription": "%s"\n}\n' "$desc" > "$caption_file"
        fi

        # CSV row -- num_frames=10000 lies past the min_clip_length filter; actual sampling
        # uses decord's real video length (~81 frames).
        echo "$stem,$stem/$stem.mp4,$stem,10000,30" >> "$META_CSV"
        count=$((count + 1))
    done
done

echo ""
echo "==> Wrote $count rows to $META_CSV"
echo "==> Skipped $skipped_no_label clips (no SAM3 label)"
echo "==> DATA_ROOT = $DATA_ROOT"
echo ""
echo "Next: sbatch scripts/slurm/train_semantic.sh"
echo "(and edit train_semantic.yaml: change ROOT to $DATA_ROOT, adjust epochs/output paths/wandb run name)"
