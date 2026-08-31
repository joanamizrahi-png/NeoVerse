#!/usr/bin/env bash
#SBATCH --job-name=pano-pipe
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclude=n04,n06,n13,n14,n17,n21,n24,n26,n30,n31
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-pano-pipe-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-pano-pipe-%j.err

# PANO SCENE PIPELINE (2026-08-31, true-360 track): one job takes an extracted
# pano scene (main mp4 + _pano_yaw{090,270}.mp4 from prepare_rosbag_clips
# --pano_topic) all the way to nav-ready: SAM3 labels for all three views ->
# v14 remap -> reconstructor poses for the main clip. After it finishes,
# spin-certify with drive_preview SPIN=1 and compare the cov curve against a
# forward-only twin — cov improvement at the flanks = the pano views working.
# Knobs: SCENE (default gnd_AUpano01), CAM_H (GND rail robot ZED ~0.5-0.6).

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
export HF_HUB_DISABLE_PROGRESS_BARS=1

S=${SCENE:-gnd_AUpano01}
CLIP_DIR=/scratch/m000204-pm06b/joana/data/rugd_clips

cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

for V in "" _pano_yaw090 _pano_yaw270; do
    test -f "$CLIP_DIR/${S}${V}.mp4" \
        || { echo "[sanity] FATAL: missing $CLIP_DIR/${S}${V}.mp4"; exit 1; }
done

for V in "" _pano_yaw090 _pano_yaw270; do
    echo "==> SAM3 labeling ${S}${V}"
    python sam3_precompute_labels.py --input_path "$CLIP_DIR/${S}${V}.mp4"
done

echo "==> remap to v14"
python scripts/remap_labels_to_v14.py --dirs outputs/sam3_labels

for V in "" _pano_yaw090 _pano_yaw270; do
    test -f "outputs/sam3_labels_v14/${S}${V}.npz" \
        || { echo "[sanity] FATAL: v14 labels missing for ${S}${V}"; exit 1; }
done

echo "==> reconstructor poses for ${S}"
cd /scratch/m000204-pm06b/joana/nav-rl
python scripts/extract_poses.py \
    --videos "$CLIP_DIR/${S}.mp4" \
    --output_dir /scratch/m000204-pm06b/joana/outputs/poses \
    --reconstructor_path /scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
    --num_frames 81 --width 560 --height 336 \
    --camera_height_m "${CAM_H:-0.6}"

echo "==> pano scene ${S} nav-ready. Next: SPIN-certify:"
echo "    env SCENE=${S} START=40 FRAMES=33 SPIN=1 SPINDEG=240 HEIGHT=224 WIDTH=336 sbatch scripts/slurm/drive_preview.sh"
