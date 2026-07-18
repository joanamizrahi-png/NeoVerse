#!/usr/bin/env bash
#SBATCH --job-name=inf-v6e20
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/inference_v6_epoch20_rugdtrail/slurm-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/inference_v6_epoch20_rugdtrail/slurm-%j.err

# v6 inference. Must match training-time config exactly:
#   - semantic_expansion_version 2 (parallel _sem submodules)
#   - lora_rank 16 (v6 used rank 16, not v5's 32)
#   - lora_target_modules q,k,v,o,ffn.0,ffn.2 (v6 does NOT LoRA patch_embedding /
#     head.head — those are handled by full-trainable .sem sub-modules)

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/inference_v6_epoch20_rugdtrail

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

python inference_semantic.py \
    --input_path /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_trail_00.mp4 \
    --checkpoint /scratch/m000204-pm06b/joana/runs/train_semantic_v6/checkpoint-epoch-20.safetensors \
    --output_dir /scratch/m000204-pm06b/joana/inference_v6_epoch20_rugdtrail \
    --model_path /scratch/m000204-pm06b/joana/NeoVerse/models \
    --reconstructor_path /scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
    --trajectory static \
    --semantic_expansion_version 2 \
    --lora_rank 16 \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2"

echo "==> v6 inference done; outputs in /scratch/m000204-pm06b/joana/inference_v6_epoch20_rugdtrail/"
