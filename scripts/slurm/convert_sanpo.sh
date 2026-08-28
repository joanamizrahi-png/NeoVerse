#!/usr/bin/env bash
#SBATCH --job-name=sanpo-conv
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-sanpo-conv-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-sanpo-conv-%j.err

# SANPO -> v21 flashcards + dataset roots (CPU only). Resumable.

set -euo pipefail
module load conda/24.3.0-0
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm06b/joana/NeoVerse

python scripts/convert_sanpo_clips.py \
    --sanpo /scratch/m000204-pm06b/joana/data/sanpo \
    --out /scratch/m000204-pm06b/joana/data/sanpo_v21 \
    --v15_root /scratch/m000204-pm06b/joana/combined_train_data_v15 \
    --sam3_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels_v14 \
    --gt_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/gt_labels_v18 \
    --clips_per_session "${CLIPS_PER_SESSION:-3}"

echo "==> sanpo-conv done"
