#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v35b_p6_ppl5
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=05:00:00
#SBATCH --exclude=n04,n13,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v35b_p6_ppl5-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v35b_p6_ppl5-%j.err
# v35b (2026-09-20): v33 epoch 9 continued for 3 epochs on SANPO WITHOUT the people-heavy
# clips (person+vehicle share > 5% of GT pixels dropped: 104 of 287). Movers stay per frame
# as in v33; only the training set changes. See the config header for the why.
# Build the filtered root first (login node, ~1 min):
#   python scripts/filter_sanpo_by_person_share.py --root .../sanpo_v26/combined_train_data_v21 \
#       --gt_dir .../sanpo_v26/gt_labels_v21 --max_share 0.05 --out .../sanpo_v26/combined_train_data_v21_ppl5
# Same torchrun launch as v33; ~21 min per epoch on 4 GPUs at 183 clips -> ~1 h.
#   sbatch scripts/slurm/train_semantic_v35b_p6_ppl5.sh
set -euo pipefail
NGPU=${NGPU:-4}
CONFIG=${CONFIG:-training/configs/train_semantic_v35b_p6_ppl5.yaml}
mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v35b_p6_ppl5
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
ROOT=/scratch/m000204-pm06b/joana/data/sanpo_v26/combined_train_data_v21_ppl5
CSV=$ROOT/data/train/SpatialVID_HQ_metadata.csv
test -f "$CSV" || { echo "[sanity] FATAL: filtered root missing ($CSV) -- run filter_sanpo_by_person_share.py first"; exit 1; }
test -e "$ROOT/SpatialVid/HQ" || { echo "[sanity] FATAL: $ROOT/SpatialVid is not linked to the videos"; exit 1; }
NROWS=$(( $(wc -l < "$CSV") - 1 ))
echo "[sanity] training clips: $NROWS (full set is 287)"
[ "$NROWS" -ge 150 ] && [ "$NROWS" -le 220 ] || { echo "[sanity] FATAL: $NROWS clips -- the 5% filter should keep ~183"; exit 1; }
grep -q "combined_train_data_v21_ppl5" "$CONFIG" || { echo "[sanity] FATAL: config does not point at the filtered root"; exit 1; }
grep -q "semantic_palette_version: 6" "$CONFIG" || { echo "[sanity] FATAL: config is not palette 6"; exit 1; }
grep -q "static_movers: \[12, 13\]" "$CONFIG" || { echo "[sanity] FATAL: config must keep movers per frame ([12, 13]) like v33"; exit 1; }
grep -q "^num_epochs: 3" "$CONFIG" || { echo "[sanity] FATAL: num_epochs is not 3"; exit 1; }
grep -q "^debug_save_root: null" "$CONFIG" || { echo "[sanity] FATAL: debug dump is on -- this would be a dump run, not a training run"; exit 1; }
PRE=$(grep -o "pretrained_path: .*" "$CONFIG" | awk '{print $2}')
test -f "$PRE" || { echo "[sanity] FATAL: warm-start weights missing: $PRE"; exit 1; }
case "$PRE" in *train_semantic_v33_p6_multi/checkpoint-epoch-9.safetensors) ;; *) echo "[sanity] FATAL: warm start is not v33 epoch 9: $PRE"; exit 1;; esac
echo "[sanity] warm start from $PRE ; processes: $NGPU ; movers per frame, people-heavy clips dropped"
python -m torch.distributed.run --standalone --nproc_per_node="$NGPU" train.py "$CONFIG"
echo "==> v35b done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v35b_p6_ppl5/"
