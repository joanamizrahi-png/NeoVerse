#!/usr/bin/env bash
#SBATCH --job-name=segf-labels
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-segf-labels-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-segf-labels-%j.err
#SBATCH --exclude=n04,n13,n17,n21,n24,n26,n31

# v18 pseudo-GT: Cityscapes-trained SegFormer over the campus clips (SCAND,
# GND, Stuttgart, Go2W). Output npzs land in outputs/segformer_gt_labels_v14
# — same format as rugd_gt_labels_v14, consumed as DENSE GT by training
# (the campus-class supervisor v10 never had: person/sidewalk/road/vehicle).
#   sbatch scripts/slurm/segmenter_labels.sh

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm06b/joana/NeoVerse
echo "commit: $(git log --oneline -1)"

python -c "import transformers" 2>/dev/null || python -m pip install --quiet transformers

# MODEL / OUT_DIR knobs drive the supervisor bake-off:
#   default            -> Mask2Former-Cityscapes  -> outputs/segformer_gt_labels_v14
#   MODEL=...vistas... -> Mask2Former-Mapillary   -> OUT_DIR=outputs/m2f_vistas_labels_v14
MODEL=${MODEL:-facebook/mask2former-swin-large-cityscapes-semantic}
OUT_DIR=${OUT_DIR:-outputs/segformer_gt_labels_v14}
D=/scratch/m000204-pm06b/joana/data
python scripts/segmenter_labels.py \
    --videos "$D"/scand_clips/*.mp4 "$D"/gnd_clips/gnd_*.mp4 \
             "$D"/cityscapes_clips/*.mp4 "$D"/go2w_clips/go2w_*.mp4 \
    --model "$MODEL" \
    --out_dir "$OUT_DIR"
echo "==> segmenter labels done: $OUT_DIR"
