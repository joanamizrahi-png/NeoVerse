#!/usr/bin/env bash
#SBATCH --job-name=chain2
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=n04,n17
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-chain2-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-chain2-%j.err

# CHAIN v2: dream-arc seeding. v1 seeded the PREFIX (real-region frames,
# fighting the raster hint; agreement got WORSE). v2 seeds latent frames 6-16
# (video frames ~21-61 = headings ~93-271 deg — the pure-dream arc), so the
# previous spot's dream directly conditions the region that needs it.
# Same gates as v1: seam integrity, dream agreement vs 68-84% baseline, drift.

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

RUNS=/scratch/m000204-pm06b/joana/runs/train_semantic_v10
CKPT=$(ls -t "$RUNS"/checkpoint-epoch-*.safetensors 2>/dev/null | head -1 || true)
[ -f "$CKPT" ] || { echo "FATAL: no v10 checkpoint under $RUNS"; exit 1; }
TRAJ=/scratch/m000204-pm06b/joana/outputs/ribbon_traj_spin/rugd_trail_00
OUTROOT=/scratch/m000204-pm06b/joana/outputs/chain2_pilot

PREV=""
for SPIN in spin_f40_lat+0.00 spin_f42_lat+0.00 spin_f44_lat+0.00; do
    EXTRA=()
    [ -n "$PREV" ] && EXTRA=(--chain_seed_dir "$OUTROOT/$PREV" --chain_seed_lat_range 6-16)
    python inference_semantic.py \
        --input_path /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_trail_00.mp4 \
        --checkpoint "$CKPT" \
        --output_dir "$OUTROOT/$SPIN" \
        --trajectory_file "$TRAJ/$SPIN.json" \
        "${EXTRA[@]}" \
        --model_path /scratch/m000204-pm06b/joana/NeoVerse/models \
        --reconstructor_path /scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
        --semantic_expansion_version 2 --lora_rank 8 \
        --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
        --semantic_labels outputs/sam3_labels_v14/rugd_trail_00.npz \
        --num_semantic_classes 14 --semantic_x0_prediction --decode_with_head
    PREV=$SPIN
done
echo "==> chain2 pilot done: $OUTROOT"
