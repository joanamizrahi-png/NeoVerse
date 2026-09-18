#!/usr/bin/env bash
#SBATCH --job-name=train-sem-v34_p6_rugd
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --exclude=n04,n13,n17,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-train-sem-v34_p6_rugd-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-train-sem-v34_p6_rugd-%j.err
# v34: v33 continued on SANPO + RUGD (rough / trail vocabulary), see the
# config header. Same torchrun launch as v33: one process per GPU, ~100 GB
# host RAM per process, so --mem is sized for 4 processes. 10 epochs at
# ~33 min each on 4 GPUs (v33's rate on 287 clips; 319 clips here).
#
# Build the dataset first (login node, ~1 h CPU):
#   cp -r outputs/sam3_labels outputs/sam3_labels_m2
#   cp -r outputs/rugd_gt_labels outputs/rugd_gt_labels_m2
#   python scripts/remap_labels_to_v14.py --dirs outputs/sam3_labels_m2 outputs/rugd_gt_labels_m2
#   python scripts/convert_sanpo_clips.py --sanpo .../data/sanpo --out .../data/sanpo_v34 \
#       --v15_root .../combined_train_data_v15 --sam3_dir outputs/sam3_labels_m2_v14 \
#       --gt_dir outputs/rugd_gt_labels_m2_v14 --clips_per_session 6
# (the _m2 copies keep the old *_v14 label dirs, which older dataset roots
# symlink into, untouched by the new merge).
set -euo pipefail
NGPU=${NGPU:-4}
CONFIG=${CONFIG:-training/configs/train_semantic_v34_p6_rugd.yaml}
mkdir -p /scratch/m000204-pm06b/joana/runs/train_semantic_v34_p6_rugd

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"
nvidia-smi -L
NSEEN=$(nvidia-smi -L | wc -l)
[ "$NSEEN" -ge "$NGPU" ] || { echo "REFUSED: NGPU=$NGPU but the job has $NSEEN GPU(s)"; exit 2; }

# --- sanity guards: every one of these has caught a real launch bug before ---
ROOT=/scratch/m000204-pm06b/joana/data/sanpo_v34
CSV=$ROOT/combined_train_data_v21/data/train/SpatialVID_HQ_metadata.csv
test -f "$CSV" || { echo "[sanity] FATAL: v34 dataset missing ($CSV) -- run the build in the header"; exit 1; }
NROWS=$(( $(wc -l < "$CSV") - 1 ))
NRUGD=$(grep -c "^rugd_" "$CSV" || true)
echo "[sanity] training clips: $NROWS (rugd: $NRUGD)"
[ "$NRUGD" -ge 30 ] || { echo "[sanity] FATAL: only $NRUGD rugd rows -- the run would have no rough/trail source"; exit 1; }
[ "$NROWS" -ge 300 ] || { echo "[sanity] FATAL: only $NROWS rows -- SANPO clips missing"; exit 1; }
# the RUGD targets must carry the new merge: some pixels labelled rough (4)
python - <<'EOF' || exit 1
import numpy as np, glob
fs = sorted(glob.glob("/scratch/m000204-pm06b/joana/data/sanpo_v34/gt_labels_v21/rugd_*.npz"))[:8]
assert fs, "[sanity] no rugd GT in gt_labels_v21"
tot = np.zeros(14);
for f in fs:
    lab = np.load(f)["labels"]; tot += np.bincount(lab.ravel().astype(np.int64), minlength=14)[:14]
share = tot / tot.sum()
print(f"[sanity] rugd GT class shares (8 clips): rough {share[4]:.3f} trail {share[2]:.3f} grass {share[3]:.3f}")
assert share[4] > 0.01, "[sanity] FATAL: rough is (almost) absent from the RUGD targets -- was remap_labels_to_v14 run after the merge commit?"
EOF
grep -q "semantic_palette_version: 6" "$CONFIG" || { echo "[sanity] FATAL: config is not palette 6"; exit 1; }
grep -q "static_movers: \[12, 13\]" "$CONFIG" || { echo "[sanity] FATAL: config has no static movers"; exit 1; }
grep -q "sanpo_v34" "$CONFIG" || { echo "[sanity] FATAL: config does not point at sanpo_v34"; exit 1; }
PRE=$(grep -o "pretrained_path: .*" "$CONFIG" | awk '{print $2}')
test -f "$PRE" || { echo "[sanity] FATAL: warm-start weights missing: $PRE"; exit 1; }
echo "[sanity] warm start from $PRE ; processes: $NGPU"

python -m torch.distributed.run --standalone --nproc_per_node="$NGPU" train.py "$CONFIG"
echo "==> v34 done; checkpoints in /scratch/m000204-pm06b/joana/runs/train_semantic_v34_p6_rugd/"
