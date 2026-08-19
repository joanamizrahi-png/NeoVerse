#!/usr/bin/env bash
#SBATCH --job-name=inf-v5e20
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/inference_v5_epoch20_rugdtrail/slurm-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/inference_v5_epoch20_rugdtrail/slurm-%j.err
#SBATCH --exclude=n04,n13,n17,n24

# Inference on v5 final checkpoint (epoch 20), static trajectory, RUGD trail clip.
# v5 used LoRA on head+patch_embedding from scratch (no warmstart). Final training
# semantic_loss ~1.56, rgb_loss ~0.17. This eyeballs whether the semantic output
# is actually recognizable class maps (vs palette noise).

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/inference_v5_epoch20_rugdtrail

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

python inference_semantic.py \
    --input_path /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_trail_00.mp4 \
    --checkpoint /scratch/m000204-pm06b/joana/runs/train_semantic_v5/checkpoint-epoch-20.safetensors \
    --output_dir /scratch/m000204-pm06b/joana/inference_v5_epoch20_rugdtrail \
    --model_path /scratch/m000204-pm06b/joana/NeoVerse/models \
    --reconstructor_path /scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
    --trajectory static

echo "==> inference done; outputs in /scratch/m000204-pm06b/joana/inference_v5_epoch20_rugdtrail/"
