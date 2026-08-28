#!/usr/bin/env bash
#SBATCH --job-name=sanpo-dl
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-sanpo-dl-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-sanpo-dl-%j.err

# SANPO labeled-slice download (CPU only, no GPU). Resumable: rerun freely.
# MAX_SESSIONS: labeled sessions to pull (default 60).

set -euo pipefail
module load conda/24.3.0-0
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm06b/joana/NeoVerse

python scripts/download_sanpo.py \
    --out /scratch/m000204-pm06b/joana/data/sanpo \
    --max_sessions "${MAX_SESSIONS:-60}" \
    --workers 8

echo "==> sanpo-dl done"
