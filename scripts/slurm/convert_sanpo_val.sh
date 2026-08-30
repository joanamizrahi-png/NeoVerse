#!/usr/bin/env bash
#SBATCH --job-name=sanpo-val
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-sanpo-val-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-sanpo-val-%j.err

# SANPO -> v21 flashcards + dataset roots (CPU only). Resumable.

set -euo pipefail
module load conda/24.3.0-0
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm06b/joana/NeoVerse

python scripts/convert_sanpo_val.py \
    --sanpo /scratch/m000204-pm06b/joana/data/sanpo \
    --out /scratch/m000204-pm06b/joana/data/sanpo_val

echo "==> sanpo-val done"
