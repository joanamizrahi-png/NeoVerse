#!/usr/bin/env bash
#SBATCH --job-name=dreamlift
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --exclude=n04,n17
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-dreamlift-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-dreamlift-%j.err

# DREAM LIFTING pilot (docs/DREAM_CONSISTENCY_DESIGNS.md, design 1).
# Appends the generated spin_f40 sweep to the reconstruction views, then:
#   render 1: spin_f40's own trajectory  -> GATE B (dream committed: backward
#             alpha ~0% -> >50%, rough render shows the dream)
#   render 2: spin_f42's trajectory      -> GATE C (neighbor consistency:
#             formerly-dream regions should now match f40's committed dream)
# Compare against the ORIGINAL ribbon_cache_spin/spin_f40 & f42 renders.

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

RUNS=/scratch/m000204-pm06b/joana/runs/train_semantic_v10
CKPT=$(ls -t "$RUNS"/checkpoint-epoch-*.safetensors | head -1)
DREAM=/scratch/m000204-pm06b/joana/outputs/ribbon_cache_spin/rugd_trail_00/spin_f40_lat+0.00
TRAJ=/scratch/m000204-pm06b/joana/outputs/ribbon_traj_spin/rugd_trail_00
OUTROOT=/scratch/m000204-pm06b/joana/outputs/dreamlift_pilot
[ -d "$DREAM" ] || { echo "FATAL: no dream sweep at $DREAM"; exit 1; }

for SPIN in spin_f40_lat+0.00 spin_f42_lat+0.00; do
python inference_semantic.py \
    --input_path /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_trail_00.mp4 \
    --checkpoint "$CKPT" \
    --output_dir "$OUTROOT/$SPIN" \
    --trajectory_file "$TRAJ/$SPIN.json" \
    --append_views_dir "$DREAM" \
    --append_views_timestamp 40 \
    --model_path /scratch/m000204-pm06b/joana/NeoVerse/models \
    --reconstructor_path /scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
    --semantic_expansion_version 2 --lora_rank 8 \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --semantic_labels outputs/sam3_labels_v14/rugd_trail_00.npz \
    --num_semantic_classes 14 --semantic_x0_prediction --decode_with_head
done
echo "==> dreamlift pilot done: $OUTROOT"
