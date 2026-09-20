#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v35_p6_fused
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=05:00:00
#SBATCH --exclude=n04,n13,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v35_p6_fused-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v35_p6_fused-%j.err
# v35 (2026-09-20): v33 epoch 9 continued for 3 epochs with movers FUSED into the
# static scene (static_movers []), after the hint audit confirmed that the per-frame
# setting teaches "paint a person where the hint shows ground" (see the config header).
# Same torchrun launch as v33/v34: one process per GPU. v33 ran ~33 min per epoch on
# 4 GPUs -> 3 epochs ~ 2 h; 5 h limit.
#   sbatch scripts/slurm/train_semantic_v35_p6_fused.sh
set -euo pipefail
NGPU=${NGPU:-4}
CONFIG=${CONFIG:-training/configs/train_semantic_v35_p6_fused.yaml}
mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v35_p6_fused
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"
nvidia-smi -L
NSEEN=$(nvidia-smi -L | wc -l)
[ "$NSEEN" -ge "$NGPU" ] || { echo "REFUSED: NGPU=$NGPU but the job has $NSEEN GPU(s)"; exit 2; }
CSV=/scratch/m000204-pm06b/joana/data/sanpo_v26/combined_train_data_v21/data/train/SpatialVID_HQ_metadata.csv
test -f "$CSV" || { echo "[sanity] FATAL: v26 dataset missing"; exit 1; }
echo "[sanity] training clips: $(( $(wc -l < "$CSV") - 1 ))"
grep -q "semantic_palette_version: 6" "$CONFIG" || { echo "[sanity] FATAL: config is not palette 6"; exit 1; }
grep -q "static_movers: \[\]" "$CONFIG" || { echo "[sanity] FATAL: config does not fuse the movers (static_movers must be [])"; exit 1; }
grep -q "^num_epochs: 3" "$CONFIG" || { echo "[sanity] FATAL: num_epochs is not 3"; exit 1; }
grep -q "^debug_save_root: null" "$CONFIG" || { echo "[sanity] FATAL: debug dump is on -- this would be a dump run, not a training run"; exit 1; }
PRE=$(grep -o "pretrained_path: .*" "$CONFIG" | awk '{print $2}')
test -f "$PRE" || { echo "[sanity] FATAL: warm-start weights missing: $PRE"; exit 1; }
case "$PRE" in *train_semantic_v33_p6_multi/checkpoint-epoch-9.safetensors) ;; *) echo "[sanity] FATAL: warm start is not v33 epoch 9: $PRE"; exit 1;; esac
echo "[sanity] warm start from $PRE ; processes: $NGPU ; movers fused"
python -m torch.distributed.run --standalone --nproc_per_node="$NGPU" train.py "$CONFIG"
echo "==> v35 done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v35_p6_fused/"
