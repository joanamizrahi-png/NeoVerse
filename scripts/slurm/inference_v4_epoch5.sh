#!/usr/bin/env bash
#SBATCH --job-name=inf-v4e5
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/m000204-pm06b/joana/inference_v4_epoch5_rugdtrail/slurm-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/inference_v4_epoch5_rugdtrail/slurm-%j.err

# One-shot inference on v4 checkpoint-epoch-5, static trajectory, RUGD trail clip.
# Sanity-check whether the fix v4 applied (unfreeze semantic slots) is producing
# structured semantic output instead of palette noise.

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/inference_v4_epoch5_rugdtrail

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

python inference_semantic.py \
    --input_path /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_trail_00.mp4 \
    --checkpoint /scratch/m000204-pm06b/joana/runs/train_semantic_v4/checkpoint-epoch-5.safetensors \
    --output_dir /scratch/m000204-pm06b/joana/inference_v4_epoch5_rugdtrail \
    --model_path /scratch/m000204-pm06b/joana/NeoVerse/models \
    --reconstructor_path /scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
    --trajectory static

echo "==> inference done; outputs in /scratch/m000204-pm06b/joana/inference_v4_epoch5_rugdtrail/"
