#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v33_p6_multi
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=40:00:00
#SBATCH --exclude=n04,n13,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v33_p6_multi-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v33_p6_multi-%j.err
# v33: v32's palette-6 from-scratch line continued from its epoch-3 weights on
# several GPUs (2026-09-16). One process per GPU via torchrun; the Accelerator
# inside train.py picks the distributed setup up from the environment.
#   sbatch scripts/slurm/train_semantic_v33_p6_multi.sh
#   NGPU=2 sbatch --gres=gpu:2 --mem=96G --cpus-per-task=8 --time=00:30:00 ...   (smoke)
set -euo pipefail
NGPU=${NGPU:-4}
CONFIG=${CONFIG:-training/configs/train_semantic_v33_p6_multi.yaml}
mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v33_p6_multi
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
grep -q "static_movers: \[12, 13\]" "$CONFIG" || { echo "[sanity] FATAL: config has no static movers"; exit 1; }
PRE=$(grep -o "pretrained_path: .*" "$CONFIG" | awk '{print $2}')
test -f "$PRE" || { echo "[sanity] FATAL: warm-start weights missing: $PRE"; exit 1; }
echo "[sanity] warm start from $PRE ; processes: $NGPU"
python -m torch.distributed.run --standalone --nproc_per_node="$NGPU" train.py "$CONFIG"
echo "==> v33 done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v33_p6_multi/"
