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
CLIPS_ROOT=/scratch/m000204-pm06b/joana/combined_train_data
mapfile -t CLIPS < <(find "$CLIPS_ROOT" -name "*.mp4" | sort)
echo "found ${#CLIPS[@]} clips under $CLIPS_ROOT"
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
