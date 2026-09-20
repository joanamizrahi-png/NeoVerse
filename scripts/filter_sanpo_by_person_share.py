"""Build a training-set root with the people-heavy SANPO clips removed (2026-09-20).

Why: SANPO is a head camera in crowds (5.6% of training pixels are person/vehicle, single
clips up to 57%); our campus recordings are 0.7%. The semantic model carried that crowd
prior into thin campus views and invented people on pavement (the phantom crashes). The
hint audit showed neither mover mode fixes the person class, so v35b keeps v33's recipe and
matches the prior instead: drop every clip whose person+vehicle share of the GT labels is
above --max_share.

The new root is a directory beside the old one: everything symlinked, only
data/train/SpatialVID_HQ_metadata.csv rewritten with the kept rows.

    python scripts/filter_sanpo_by_person_share.py \
        --root /scratch/m000204-pm06b/joana/data/sanpo_v26/combined_train_data_v21 \
        --gt_dir /scratch/m000204-pm06b/joana/data/sanpo_v26/gt_labels_v21 \
        --max_share 0.05 --out /scratch/m000204-pm06b/joana/data/sanpo_v26/combined_train_data_v21_ppl5
"""
import argparse, csv, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--gt_dir", required=True)
ap.add_argument("--max_share", type=float, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--classes", default="12,13")
a = ap.parse_args()
cls = tuple(int(c) for c in a.classes.split(","))
csv_rel = "data/train/SpatialVID_HQ_metadata.csv"
rows = list(csv.DictReader(open(os.path.join(a.root, csv_rel))))
keep, drop = [], []
for r in rows:
    f = os.path.join(a.gt_dir, r["id"] + ".npz")
    if not os.path.exists(f):
        drop.append((r["id"], None)); continue
    lab = np.load(f)["labels"][::4, ::4, ::4]
    share = float(np.isin(lab, cls).mean())
    (keep if share <= a.max_share else drop).append((r["id"], share))
os.makedirs(os.path.join(a.out, "data/train"), exist_ok=True)
for name in os.listdir(a.root):
    if name == "data":
        continue
    dst = os.path.join(a.out, name)
    if not os.path.lexists(dst):
        os.symlink(os.path.join(a.root, name), dst)
for name in os.listdir(os.path.join(a.root, "data")):
    if name == "train":
        continue
    dst = os.path.join(a.out, "data", name)
    if not os.path.lexists(dst):
        os.symlink(os.path.join(a.root, "data", name), dst)
kept_ids = {k for k, _ in keep}
with open(os.path.join(a.out, csv_rel), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows:
        if r["id"] in kept_ids:
            w.writerow(r)
with open(os.path.join(a.out, "dropped_clips.txt"), "w") as fh:
    for k, s in sorted(drop, key=lambda x: -(x[1] or 0)):
        fh.write(f"{k}\t{'no GT' if s is None else f'{100 * s:.1f}%'}\n")
ks = np.array([s for _, s in keep])
print(f"kept {len(keep)} / {len(rows)} clips (dropped {len(drop)}: share > {100 * a.max_share:.0f}% or no GT)")
print(f"remaining person+vehicle share: mean {100 * ks.mean():.2f}%  max {100 * ks.max():.1f}%")
print(f"wrote {os.path.join(a.out, csv_rel)} and dropped_clips.txt")
