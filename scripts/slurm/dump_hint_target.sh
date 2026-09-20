#!/usr/bin/env bash
#SBATCH --job-name=hint-dump
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=00:30:00
#SBATCH --exclude=n04,n13,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-hint-dump-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-hint-dump-%j.err
# HINT/TARGET DUMP (2026-09-19): does the training hint contain the people and
# vehicles the target contains? Runs the training loop for ~20 minutes with
# debug_save_root set, one process, one clip per step, and lets the time limit
# kill it. CONFIG=training/configs/dump_v33_movers_perframe.yaml (v33's setting)
# or dump_v33_movers_fused.yaml (the v35 proposal). Then:
#   python scripts/hint_mover_audit.py /scratch/.../outputs/hint_dump/<name>
set -euo pipefail
CONFIG=${CONFIG:?set CONFIG=training/configs/dump_v33_movers_*.yaml}
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"; nvidia-smi -L
grep -q "debug_save_root: /scratch" "$CONFIG" || { echo "[sanity] FATAL: config has no debug_save_root"; exit 1; }
PRE=$(grep -o "pretrained_path: .*" "$CONFIG" | awk '{print $2}'); test -f "$PRE" || { echo "[sanity] FATAL: weights missing: $PRE"; exit 1; }
echo "[sanity] $(grep static_movers "$CONFIG") ; dump -> $(grep -o 'debug_save_root: .*' "$CONFIG" | awk '{print $2}')"
python -m torch.distributed.run --standalone --nproc_per_node=1 train.py "$CONFIG"
echo "==> dump done (or killed by the time limit, which is expected)"
