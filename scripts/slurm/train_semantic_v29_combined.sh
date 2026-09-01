#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v29_combined
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --exclude=n04,n13,n14,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v29_combined-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v29_combined-%j.err

# v29_combined: campus-only semantics (SANPO, no RUGD, ~2x clips). See the config
# header for why. Needs the v26 dataset build (job 459173) to have finished.

set -euo pipefail
mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v29_combined
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

# Guards: campus-only dataset must EXIST and must contain no rugd clips.
CSV=/scratch/m000204-pm06b/joana/data/sanpo_v29/combined_train_data_v21/data/train/SpatialVID_HQ_metadata.csv
test -f "$CSV" || { echo "[sanity] FATAL: v26 dataset missing (is 459173 done?)"; exit 1; }
if ! grep -q "^rugd" "$CSV"; then
    echo "[sanity] FATAL: v29 is the COMBINED diet but has no rugd clips"; exit 1
fi
if ! awk -F, 'NR>1 && $1 ~ /^sanpo/ && $5 <= 9 {found=1} END{exit !found}' "$CSV"; then
    echo "[sanity] FATAL: sanpo rows still declare fps > 9 — the length filter"
    echo "                will drop them again (see project_sanpo_never_trained)"; exit 1
fi
echo "[sanity] campus-only clips: $(( $(wc -l < "$CSV") - 1 ))"
grep -q "sanpo_v29" training/configs/train_semantic_v29_combined.yaml \
    || { echo "[sanity] FATAL: config not pointing at the v26 roots"; exit 1; }

python train.py training/configs/train_semantic_v29_combined.yaml
echo "==> v29_combined done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v29_combined/"
