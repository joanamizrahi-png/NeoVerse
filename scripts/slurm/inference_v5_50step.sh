#!/usr/bin/env bash
#SBATCH --job-name=inf-v5-50step
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-inf-v5-50step-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-inf-v5-50step-%j.err

# 50-step eval of v5 WITHOUT the lightx2v 4-step distill LoRA (--disable_lora).
#
# Why: v5 was trained on the full 1000-timestep schedule with NO distill LoRA
# loaded. The default inference path merges the distill LoRA and samples 4 steps
# — a regime the semantic pathway (and 20 epochs of control_branch/attention
# drift) never saw. This run removes that confound: if outputs are clean here,
# v5 is fine and only the sampling regime was wrong; if still mushy, the
# training-side architecture (rank-32 semantic bottleneck) is the problem -> v6.
#
# Runs epoch 20 AND epoch 10: later epochs drift hardest against the distill
# LoRA and may overfit the 46 clips, so a mid-training checkpoint is a useful
# second data point. Each 50-step run is ~10x slower than 4-step, hence 4h.

set -euo pipefail

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

RUNS_DIR=/scratch/m000204-pm06b/joana/runs/train_semantic_v5

for EPOCH in 20 10; do
    CKPT="${RUNS_DIR}/checkpoint-epoch-${EPOCH}.safetensors"
    if [[ ! -f "$CKPT" ]]; then
        echo "==> SKIP: $CKPT not found (ls ${RUNS_DIR} and fix the epoch list)"
        continue
    fi
    OUT="/scratch/m000204-pm06b/joana/inference_v5_epoch${EPOCH}_rugdtrail_50step"
    mkdir -p "$OUT"
    echo "==> epoch ${EPOCH}, 50-step, output -> $OUT"
    python inference_semantic.py \
        --input_path /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_trail_00.mp4 \
        --checkpoint "$CKPT" \
        --output_dir "$OUT" \
        --model_path /scratch/m000204-pm06b/joana/NeoVerse/models \
        --reconstructor_path /scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
        --trajectory static \
        --disable_lora
done

echo "==> done. Compare semantic_raw.mp4 (pre-snap) vs semantic.mp4 (snapped) per dir."
