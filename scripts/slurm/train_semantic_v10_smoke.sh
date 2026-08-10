#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v10-smoke
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v10-smoke-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v10-smoke-%j.err

# v10 attribution smokes: 5 epochs of the v9 recipe + ONE candidate change each.
#   sbatch --export=VARIANT=pres   scripts/slurm/train_semantic_v10_smoke.sh
#   VARIANT: pres (RGB-preservation loss) / snr (min-SNR weighting) /
#            both / cegate (CE+SAM2 at all timesteps)
# Judge on wandb split losses (rgb should drop vs v9; semantic must not rise)
# + a rendered clip per winner. Winner(s) define the full v10 config.
# NOTE: pres/both run a second frozen forward per step (~1.6x step time) —
# hence the 8h wall.

set -euo pipefail

VARIANT=${VARIANT:?set VARIANT=pres|snr|both|cegate via --export}
CFG=training/configs/train_semantic_v10smoke_${VARIANT}.yaml

mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v10smoke_${VARIANT}

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

echo "hostname: $(hostname)"
echo "commit: $(git log --oneline -1)"
echo "variant: $VARIANT  config: $CFG"
python -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')"

[ -f "$CFG" ] || { echo "[sanity] FATAL: $CFG missing — stale checkout?"; exit 1; }

# Stale-code guard: the loss code must actually carry the v10 knobs.
grep -q "rgb_preservation_weight" diffsynth/pipelines/wan_video_neoverse.py \
    || { echo "[sanity] FATAL: pipeline lacks rgb_preservation_weight — pull the v10 commit"; exit 1; }

case "$VARIANT" in
  pres)   grep -q "rgb_preservation_weight: 1.0" "$CFG" || { echo "[sanity] FATAL: config/variant mismatch"; exit 1; } ;;
  snr)    grep -q "snr_gamma: 5.0" "$CFG" || { echo "[sanity] FATAL: config/variant mismatch"; exit 1; } ;;
  both)   grep -q "rgb_preservation_weight: 1.0" "$CFG" && grep -q "snr_gamma: 5.0" "$CFG" || { echo "[sanity] FATAL: config/variant mismatch"; exit 1; } ;;
  cegate) grep -q "semantic_ce_sigma_max: 1.0" "$CFG" || { echo "[sanity] FATAL: config/variant mismatch"; exit 1; } ;;
  *) echo "[sanity] FATAL: unknown VARIANT=$VARIANT"; exit 1 ;;
esac

python train.py "$CFG"

echo "==> v10 smoke ($VARIANT) done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v10smoke_${VARIANT}/"
