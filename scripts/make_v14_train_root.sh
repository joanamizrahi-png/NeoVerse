#!/usr/bin/env bash
# Build combined_train_data_v14: symlink copy of combined_train_data MINUS the
# held-out clips, so v9 trains with a real test set by construction.
set -euo pipefail
SRC=/scratch/m000204-pm06b/joana/combined_train_data
DST=/scratch/m000204-pm06b/joana/combined_train_data_v14
HELD_OUT="rugd_trail-6_01 rugd_park-1_02"
rm -rf "$DST"
cp -as "$SRC" "$DST"
N=0
for H in $HELD_OUT; do
    while IFS= read -r f; do rm -rf "$f"; N=$((N+1)); done \
        < <(find "$DST" -name "${H}*")
done
echo "built $DST (removed $N held-out entries: $HELD_OUT)"
find "$DST" -name '*.mp4' | wc -l
