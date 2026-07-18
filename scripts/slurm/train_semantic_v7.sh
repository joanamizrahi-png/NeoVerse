#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v7
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v7-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v7-%j.err

# v7: dense RUGD GT targets (Option B hybrid) + trunk LoRA rank 8, 30 epochs.
# Logs go to the joana/ root (always exists) — SLURM won't create log dirs.
# PREREQ (run once, login node is fine — CPU only):
#   python scripts/prepare_rugd_gt_labels.py \
#     --annotations_root /scratch/m000204-pm06b/joana/data/rugd/RUGD_annotations \
#     --colormap /scratch/m000204-pm06b/joana/data/rugd/RUGD_annotations/RUGD_annotation-colormap.txt \
#     --clips_dir /scratch/m000204-pm06b/joana/data/rugd_clips \
#     --out_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/rugd_gt_labels
# (adjust --annotations_root to wherever the RUGD annotation masks live)

set -euo pipefail

mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v7

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

cd /scratch/m000204-pm06b/joana/NeoVerse

echo "hostname: $(hostname)"
python -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')"

GT_DIR=/scratch/m000204-pm06b/joana/NeoVerse/outputs/rugd_gt_labels
SAM3_DIR=/scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels

# Sanity: GT labels must exist and cover the RUGD clips, else v7 silently
# degenerates into a re-run of v6 (SAM3 fallback for every clip).
N_GT=$(ls "$GT_DIR"/*.npz 2>/dev/null | wc -l)
N_SAM3=$(ls "$SAM3_DIR"/*.npz 2>/dev/null | wc -l)
echo "[sanity] $N_GT GT label files, $N_SAM3 SAM3 label files"
if [ "$N_GT" -lt 10 ]; then
    echo "[sanity] FATAL: <10 GT label npz in $GT_DIR — run prepare_rugd_gt_labels.py first"
    exit 1
fi
python - <<'PY'
import numpy as np, glob
files = sorted(glob.glob("/scratch/m000204-pm06b/joana/NeoVerse/outputs/rugd_gt_labels/*.npz"))
voids = []
for f in files[:5] + files[-5:]:
    lab = np.load(f)["labels"]
    voids.append((lab == 0).mean())
    assert lab.shape[1:] == (336, 560), f"{f}: wrong shape {lab.shape}"
    assert lab.max() <= 29, f"{f}: class id out of range {lab.max()}"
print(f"[sanity] GT spot-check OK; mean void fraction {np.mean(voids):.1%} "
      f"(should be FAR below SAM3's — that's the point)")
PY

python train.py training/configs/train_semantic_v7.yaml

echo "==> v7 done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v7/"
