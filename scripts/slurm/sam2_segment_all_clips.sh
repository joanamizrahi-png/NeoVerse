#!/usr/bin/env bash
#SBATCH --job-name=sam2-segments
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-sam2-segments-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-sam2-segments-%j.err
#SBATCH --exclude=n04,n13,n17,n24

# v8 Change 3 preprocessing: SAM2 class-agnostic segments for every training
# clip. Idempotent (per-clip skip if npz exists) — safe to resubmit after a
# timeout. Runs in the sam3 conda env (transformers), like the SAM3 labeler.
# Budget: ~3-7 min/clip on H100 -> ~2.5-5 h for ~44 clips.

set -euo pipefail

module load conda/24.3.0-0
export PATH=/users/jmizrahi/.conda/envs/sam3/bin:$PATH
export PYTHONNOUSERSITE=1
# Compute nodes have no internet AND the shell has a corrupt HF token (401s on
# public repos). Offline mode makes transformers read the login-node-downloaded
# cache and never attempt a request. PREREQ (once, login node):
#   python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam2.1-hiera-large', token=False)"
export HF_HUB_OFFLINE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

# combined_train_data nests clips under subdirs (data/, SpatialVid/, train/) —
# a flat glob finds nothing (job 406381's 80-second failure). Recursive find
# works regardless of layout; clip stem must equal the dataloader's scene id,
# which holds because the same mp4 basenames feed the SAM3 labeler.
# CLIPS_ROOT is a knob (2026-09-01). The default is the RUGD root, which is
# why the 287 SANPO campus clips have no segments — and therefore why
# semantic_seg_loss never fires on the campus-only runs (v26/v28) while it does
# on every RUGD run. Point this at data/sanpo_v26 to fill that gap.
# SHARD/NSHARD split the list across parallel jobs (287 clips x ~5 min is a
# GPU-day on one node). Per-clip skip makes every shard safely resumable.
CLIPS_ROOT=${CLIPS_ROOT:-/scratch/m000204-pm06b/joana/combined_train_data}
mapfile -t ALL < <(find "$CLIPS_ROOT" -name "*.mp4" | sort)
SHARD=${SHARD:-0}
NSHARD=${NSHARD:-1}
CLIPS=()
for i in "${!ALL[@]}"; do
    if [ $((i % NSHARD)) -eq "$SHARD" ]; then CLIPS+=("${ALL[$i]}"); fi
done
echo "shard $SHARD/$NSHARD -> ${#CLIPS[@]} of ${#ALL[@]} clips"
echo "root: $CLIPS_ROOT"
if [ "${#CLIPS[@]}" -eq 0 ]; then
    echo "FATAL: no .mp4 under $CLIPS_ROOT"; exit 1
fi
N=0
for CLIP_PATH in "${CLIPS[@]}"; do
    N=$((N + 1))
    echo "=== [$N/${#CLIPS[@]}] $(basename "$CLIP_PATH") ==="
    python sam2_precompute_segments.py --input_path "$CLIP_PATH"
done

echo "==> sam2 segments done: $N clips -> outputs/sam2_segments/"
ls outputs/sam2_segments/*.npz | wc -l
