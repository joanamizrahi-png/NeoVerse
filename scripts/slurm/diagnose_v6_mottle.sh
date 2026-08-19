#!/usr/bin/env bash
#SBATCH --job-name=diag-v6-mottle
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-diag-v6-mottle-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-diag-v6-mottle-%j.err
#SBATCH --exclude=n04,n13,n17,n24

# Attributes the v6 mottle (identical high-freq noise in RGB AND semantic).
# Hypothesis: the shared rank-16 trunk LoRA drifted — it is the only trained
# component that touches the RGB path. Three runs on one GPU allocation
# (model loads once per run; diffusion itself is ~8 s):
#
#   A. epoch-20 + --zero_trunk_lora  -> trunk = pristine base.
#        RGB MUST be vanilla-clean here. Semantic quality shows how much the
#        sem pathway depends on trunk routing.
#   B. epoch-10                      -> half the drift. Mottle should be visibly
#        weaker if drift is the cause (it grows with training).
#   C. epoch-5                       -> quarter the drift.
#
# Read-out:
#   A clean-RGB + decent semantic  => v8 drops/tames trunk LoRA; speckle solved
#   A clean-RGB + collapsed sem    => trunk routing needed; v8 = lower LoRA LR/rank
#   A still mottled                => hypothesis WRONG (something else moves RGB);
#                                     escalate — check control_branch really froze.

set -euo pipefail

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse
RUNS=/scratch/m000204-pm06b/joana/runs/train_semantic_v6

run_one () {  # $1 = checkpoint epoch, $2 = output suffix, $3 = extra args
    local CKPT="$RUNS/checkpoint-epoch-$1.safetensors"
    if [[ ! -f "$CKPT" ]]; then echo "==> SKIP missing $CKPT"; return; fi
    local OUT="/scratch/m000204-pm06b/joana/inference_v6_$2_rugdtrail"
    mkdir -p "$OUT"
    echo "==> [$2] epoch $1 $3"
    python inference_semantic.py \
        --input_path /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_trail_00.mp4 \
        --checkpoint "$CKPT" \
        --output_dir "$OUT" \
        --model_path /scratch/m000204-pm06b/joana/NeoVerse/models \
        --reconstructor_path /scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
        --trajectory static \
        --semantic_expansion_version 2 --lora_rank 16 \
        --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
        $3
}

run_one 20 "e20_lorazero" "--zero_trunk_lora"
run_one 10 "e10" ""
run_one 5  "e5" ""

echo "==> diagnosis runs done. Compare rgb.mp4 + semantic_raw.mp4 across:"
echo "    inference_v6_e20_lorazero_rugdtrail / _e10_ / _e5_ / and the existing epoch20 run"
