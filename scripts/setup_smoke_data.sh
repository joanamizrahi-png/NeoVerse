#!/usr/bin/env bash
# One-time setup for the semantic-finetune smoke test.
#
# Creates a 1-clip synthetic SpatialVID-shaped dataset that reuses
# examples/videos/driving.mp4, then rerun's SAM 3 labeling with
# --static_scene so we get labels for EVERY frame of the clip (which is
# what the training dataloader expects to index into by sample_index).
#
# Idempotent -- safe to rerun.
#
# Prereqs:
#   * sam3 conda env active AND on a GPU node (SAM 3 forward pass needs cuda)
#   * SAM 3 weights already cached in ~/.cache/huggingface/ (or will download)
#   * NeoVerse repo checked out at /scratch/m000204-pm06b/joana/NeoVerse

set -euo pipefail

SCRATCH=/scratch/m000204-pm06b/joana
SMOKE_ROOT="$SCRATCH/smoke_data"
NEOVERSE="$SCRATCH/NeoVerse"
LABELS_DIR="$NEOVERSE/outputs/sam3_labels"
CLIP_ID="driving"
CLIP_SRC="$NEOVERSE/examples/videos/driving.mp4"
CLIP_DIR="$SMOKE_ROOT/SpatialVid/HQ/$CLIP_ID"
META_DIR="$SMOKE_ROOT/data/train"
META_CSV="$META_DIR/SpatialVID_HQ_metadata.csv"

echo "==> smoke dataset root: $SMOKE_ROOT"
mkdir -p "$CLIP_DIR" "$META_DIR"

# --- 1. Symlink the video into the SpatialVID-shaped layout ---
if [ ! -L "$CLIP_DIR/$CLIP_ID.mp4" ] && [ ! -f "$CLIP_DIR/$CLIP_ID.mp4" ]; then
    ln -s "$CLIP_SRC" "$CLIP_DIR/$CLIP_ID.mp4"
    echo "    symlinked $CLIP_SRC -> $CLIP_DIR/$CLIP_ID.mp4"
else
    echo "    video already in place: $CLIP_DIR/$CLIP_ID.mp4"
fi

# --- 2. caption.json -- SpatialVID dataloader reads SceneDescription ---
CAPTION="$CLIP_DIR/caption.json"
if [ ! -f "$CAPTION" ]; then
    cat > "$CAPTION" <<'EOF'
{
    "SceneDescription": "urban driving scene with roads, sidewalks, cars, and buildings"
}
EOF
    echo "    wrote $CAPTION"
else
    echo "    caption already in place: $CAPTION"
fi

# --- 3. metadata CSV (columns spatialvid.py reads: id, video path, annotation path, num frames, fps) ---
# `num frames` is only used by SpatialVID._load_data to filter out too-short clips:
#     min_clip_length = int(0.2 * fps) * (num_views - 1) * min_interval + 1
# With num_views=81, fps=30, min_interval=1 that filter requires >= 481 frames.
# Actual per-batch sampling uses decord's real video length (81 frames here) and works
# fine, so this field can be a lie -- we just need it above the filter threshold.
# Use 10000 to survive any reasonable num_views setting; ffprobe values would fail.
NFRAMES=10000
FPS=30

# Always overwrite -- it's a tiny file and the previous incarnation may have had a
# ffprobe-derived num_frames that gets us filtered out by the min_clip_length gate.
cat > "$META_CSV" <<EOF
id,video path,annotation path,num frames,fps
$CLIP_ID,$CLIP_ID/$CLIP_ID.mp4,$CLIP_ID,$NFRAMES,$FPS
EOF
echo "    wrote $META_CSV  (num_frames=$NFRAMES, fps=$FPS)"

# --- 4. Full-video SAM3 labeling (--static_scene) -- REQUIRED for training alignment ---
mkdir -p "$LABELS_DIR"
if [ -f "$LABELS_DIR/$CLIP_ID.npz" ]; then
    echo ""
    echo "==> $LABELS_DIR/$CLIP_ID.npz already exists."
    echo "    if this was made WITHOUT --static_scene (i.e., only 81 frames),"
    echo "    delete it and rerun this script to get per-frame labels."
    echo "    check with: python -c \"import numpy as np; d = np.load('$LABELS_DIR/$CLIP_ID.npz'); print('labels shape:', d['labels'].shape)\""
else
    echo ""
    echo "==> Running SAM 3 on ALL $NFRAMES frames of $CLIP_ID (this takes ~$((NFRAMES * 3 / 60)) min at ~3 sec/frame)"
    cd "$NEOVERSE"
    python sam3_precompute_labels.py \
        --input_path "$CLIP_SRC" \
        --static_scene \
        --overlay_every 32
fi

echo ""
echo "==> Done. Config: $NEOVERSE/training/configs/smoke_semantic.yaml"
echo "==> Launch training: sbatch $NEOVERSE/scripts/slurm/smoke_semantic.sh"
