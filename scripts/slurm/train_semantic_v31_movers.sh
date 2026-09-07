#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v31_movers
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=n04,n13,n14,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v31_movers-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v31_movers-%j.err

# v31_movers: v30_static plus per-frame movers (person/vehicle Gaussians are
# not fused into the constant set). See the config header.

set -euo pipefail
mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v31_movers
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

# Guards: campus-only dataset must EXIST and must contain no rugd clips.
CSV=/scratch/m000204-pm06b/joana/data/sanpo_v26/combined_train_data_v21/data/train/SpatialVID_HQ_metadata.csv
test -f "$CSV" || { echo "[sanity] FATAL: v26 dataset missing (is 459173 done?)"; exit 1; }
if grep -q "^rugd" "$CSV"; then
    echo "[sanity] FATAL: rugd clips leaked into the campus-only dataset"; exit 1
fi
echo "[sanity] campus-only clips: $(( $(wc -l < "$CSV") - 1 ))"
grep -q "sanpo_v26" training/configs/train_semantic_v31_movers.yaml \
    || { echo "[sanity] FATAL: config not pointing at the v26 roots"; exit 1; }
grep -q "static_scene=True" training/configs/train_semantic_v31_movers.yaml \
    || { echo "[sanity] FATAL: config is not static"; exit 1; }
grep -q "static_movers: \[12, 13\]" training/configs/train_semantic_v31_movers.yaml \
    || { echo "[sanity] FATAL: config has no static movers"; exit 1; }
grep -q "dynamic_label_ids" diffsynth/auxiliary_models/worldmirror/models/models/rasterization.py \
    || { echo "[sanity] FATAL: rasterizer without static movers (pull)"; exit 1; }
test -f /scratch/m000204-pm06b/joana/runs/train_semantic_v26_campus/checkpoint-epoch-10.safetensors \
    || { echo "[sanity] FATAL: v26 e10 checkpoint missing"; exit 1; }

python train.py training/configs/train_semantic_v31_movers.yaml
echo "==> v31_movers done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v31_movers/"
