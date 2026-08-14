#!/usr/bin/env bash
#SBATCH --job-name=inf-sem
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-inf-sem-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-inf-sem-%j.err

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
RUN_NAME=${RUN_NAME:?set RUN_NAME=... via --export}
CLIP=${CLIP:-rugd_trail_00}
TRAJ=${TRAJ:-static}   # move_left / pan_left / orbit_left... for off-trajectory probes
MAG=""
TRAJARGS=""
if [ -n "${ANGLE:-}" ]; then TRAJARGS="$TRAJARGS --traj_angle $ANGLE"; MAG="${MAG}_a${ANGLE}"; fi
if [ -n "${DIST:-}" ]; then TRAJARGS="$TRAJARGS --traj_distance $DIST"; MAG="${MAG}_d${DIST}"; fi
EXTRA=""
DECSUF=""
if [ "${HEAD_DECODE:-0}" = "1" ]; then EXTRA="--decode_with_head"; DECSUF="_head"; fi
NUM_CLASSES=${NUM_CLASSES:-30}   # 14 for v9+ checkpoints
if [ "$NUM_CLASSES" = "14" ]; then
    LABELS=outputs/sam3_labels_v14/${CLIP}.npz    # v14 hints for v14 models
else
    LABELS=outputs/sam3_labels/${CLIP}.npz
fi
echo "commit: $(git log --oneline -1)"
RUNS=/scratch/m000204-pm06b/joana/runs/${RUN_NAME}

# EPOCH env selects the checkpoint; unset -> newest by mtime (the final one).
# With EPOCH set, the output dir gets an _e${EPOCH} suffix so epoch probes
# never overwrite the final-checkpoint render of the same clip/trajectory.
EPSUF=""
if [[ -n "${EPOCH:-}" ]]; then
    CKPT="$RUNS/checkpoint-epoch-${EPOCH}.safetensors"
    EPSUF="_e${EPOCH}"
else
    CKPT=$(ls -t "$RUNS"/checkpoint-epoch-*.safetensors 2>/dev/null | head -1 || true)
fi
[[ -f "$CKPT" ]] || { echo "==> no checkpoint ($CKPT); contents:"; ls "$RUNS"; exit 1; }
echo "==> rendering $CKPT"
# ANCHOR=1 anchors the RGB latents to the matching vanilla run's saved trajectory
# (run inference_vanilla_traj.sh with SAVE_TRAJ=1 and the same CLIP/TRAJ/ANGLE/DIST first).
ANCSUF=""
if [ "${ANCHOR:-0}" = "1" ]; then
    TRAJPT=/scratch/m000204-pm06b/joana/inference_VANILLA_${CLIP}_${TRAJ}${MAG}/rgb_latent_traj.pt
    [[ -f "$TRAJPT" ]] || { echo "==> no anchor trajectory at $TRAJPT — run the vanilla pass with SAVE_TRAJ=1 first"; exit 1; }
    EXTRA="$EXTRA --anchor_traj $TRAJPT"
    ANCSUF="_ANCHORED"
fi
OUT="/scratch/m000204-pm06b/joana/inference_${RUN_NAME}_${CLIP}_${TRAJ}${MAG}${DECSUF}${ANCSUF}${EPSUF}"
mkdir -p "$OUT"
python inference_semantic.py \
    --input_path /scratch/m000204-pm06b/joana/data/rugd_clips/${CLIP}.mp4 \
    --checkpoint "$CKPT" \
    --output_dir "$OUT" \
    --model_path /scratch/m000204-pm06b/joana/NeoVerse/models \
    --reconstructor_path /scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
    --trajectory "$TRAJ" \
    --semantic_expansion_version 2 \
    --lora_rank 8 \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --semantic_labels "$LABELS" \
    --num_semantic_classes $NUM_CLASSES \
    --semantic_x0_prediction $EXTRA $TRAJARGS

echo "==> v8 val5 inference done: $OUT"
