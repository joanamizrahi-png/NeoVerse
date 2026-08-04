#!/usr/bin/env bash
#SBATCH --job-name=inf-v8-stage
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-inf-v8-stage-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-inf-v8-stage-%j.err

# v8 stage-1 validation inference: the epoch-5 checkpoint of the x0-prediction
# run. CRITICAL: --semantic_x0_prediction must be set — v8 checkpoints output
# the clean latent for the sem half; without the flag the sampler treats it as
# velocity and produces garbage BY CONSTRUCTION (not a model failure).
# Read: recognizable rough class regions after only 5 epochs = pass.
# Compare against inference_v7_e15/e30 outputs (same clip, same trajectory).

set -euo pipefail

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse
RUN_NAME=${RUN_NAME:?set RUN_NAME=train_semantic_v8_stage2_val5 etc via --export}
echo "commit: $(git log --oneline -1)"
RUNS=/scratch/m000204-pm06b/joana/runs/${RUN_NAME}

# EPOCH env selects the checkpoint; unset -> newest by mtime (the final one).
if [[ -n "${EPOCH:-}" ]]; then
    CKPT="$RUNS/checkpoint-epoch-${EPOCH}.safetensors"
else
    CKPT=$(ls -t "$RUNS"/checkpoint-epoch-*.safetensors | head -1)
fi
[[ -f "$CKPT" ]] || { echo "==> no checkpoint ($CKPT); contents:"; ls "$RUNS"; exit 1; }
echo "==> rendering $CKPT"
OUT="/scratch/m000204-pm06b/joana/inference_${RUN_NAME}_rugdtrail"
mkdir -p "$OUT"
python inference_semantic.py \
    --input_path /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_trail_00.mp4 \
    --checkpoint "$CKPT" \
    --output_dir "$OUT" \
    --model_path /scratch/m000204-pm06b/joana/NeoVerse/models \
    --reconstructor_path /scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
    --trajectory static \
    --semantic_expansion_version 2 \
    --lora_rank 8 \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --semantic_x0_prediction

echo "==> v8 val5 inference done: $OUT"
