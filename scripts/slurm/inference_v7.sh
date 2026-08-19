#!/usr/bin/env bash
#SBATCH --job-name=inf-v7
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-inf-v7-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-inf-v7-%j.err
#SBATCH --exclude=n04,n13,n17,n24

# v7 inference (dense GT targets, trunk LoRA rank 8). Runs epoch 30 AND epoch 15
# in one allocation (model loads once per run; diffusion itself is seconds).
# Must match train_semantic_v7.yaml: expansion v2, lora_rank 8, same targets.

set -euo pipefail

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse
RUNS=/scratch/m000204-pm06b/joana/runs/train_semantic_v7

for EPOCH in 30 15; do
    CKPT="$RUNS/checkpoint-epoch-${EPOCH}.safetensors"
    if [[ ! -f "$CKPT" ]]; then echo "==> SKIP missing $CKPT (ls $RUNS)"; continue; fi
    OUT="/scratch/m000204-pm06b/joana/inference_v7_e${EPOCH}_rugdtrail"
    mkdir -p "$OUT"
    echo "==> v7 epoch ${EPOCH}"
    python inference_semantic.py \
        --input_path /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_trail_00.mp4 \
        --checkpoint "$CKPT" \
        --output_dir "$OUT" \
        --model_path /scratch/m000204-pm06b/joana/NeoVerse/models \
        --reconstructor_path /scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
        --trajectory static \
        --semantic_expansion_version 2 \
        --lora_rank 8 \
        --lora_target_modules "q,k,v,o,ffn.0,ffn.2"
done

echo "==> v7 inference done: inference_v7_e30_rugdtrail / inference_v7_e15_rugdtrail"
