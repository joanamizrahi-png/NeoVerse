#!/usr/bin/env bash
#SBATCH --job-name=test-backend
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm_logs/backend_test-slurm-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm_logs/backend_test-slurm-%j.err
#SBATCH --exclude=n04,n13,n17,n24

# Integration test for RealWorldBackend: renders 4 poses of rugd_park-1_00
# via NeoVerse's pipeline through the WorldBackend abstraction. If images
# look sensible, our coord conversions are correct and Milestone B is unblocked.

set -euo pipefail

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

# nav-rl code — clone if missing, pull if present. nav-rl is now public so this
# works from a compute node without credentials (same as NeoVerse).
NAVRL_ROOT=/scratch/m000204-pm06b/joana/nav-rl
if [ ! -d "$NAVRL_ROOT/.git" ]; then
    [ -d "$NAVRL_ROOT" ] && rm -rf "$NAVRL_ROOT"
    git clone https://github.com/joanamizrahi-png/nav-rl.git "$NAVRL_ROOT"
else
    (cd "$NAVRL_ROOT" && git pull)
fi

cd "$NAVRL_ROOT"
mkdir -p "$NAVRL_ROOT/outputs/backend_test"

python scripts/test_real_backend.py \
    --input_path /scratch/m000204-pm06b/joana/data/rugd_clips/rugd_park-1_00.mp4 \
    --output_dir /scratch/m000204-pm06b/joana/nav-rl/outputs/backend_test \
    --render_mode rasterizer_plus_diffusion \
    --num_frames 16

echo "==> RealWorldBackend integration test done"
