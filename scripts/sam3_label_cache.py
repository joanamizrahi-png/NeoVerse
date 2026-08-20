"""Re-label a ribbon cache with SAM3 run on the cache's own generated RGB.

Why (her 2026-08-20 challenge): the cache is pre-rendered, so semantics for the
reward do NOT have to come from the diffusion model's semantic half — SAM3 can
segment the generated frames offline. This builds a sibling cache where rgb /
alpha / manifest are symlinks to the original and only semantic_labels.npz is
replaced, so any OBS_CACHE=<sibling> run is an exact A/B of
  co-generated semantics   vs.   segment-the-generated-image.

Runs SAM3 once per cell (~1 min), so a 117-cell scene is ~2 h on one GPU.

Usage (via scripts/slurm/sam3_label_cache.sh):
  python scripts/sam3_label_cache.py --cache ribbon_cache_fan \
      --scene rugd_trail_00 --out_tag sam3 [--cells 0-38]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRATCH = Path("/scratch/m000204-pm06b/joana")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="cache dir name under outputs/")
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--out_tag", default="sam3",
                    help="sibling cache is <cache>_<out_tag>")
    ap.add_argument("--cells", default=None, help='"a-b" index range into the manifest')
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    src = SCRATCH / "outputs" / args.cache / args.scene
    dst = SCRATCH / "outputs" / f"{args.cache}_{args.out_tag}" / args.scene
    dst.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text())
    if not (dst / "manifest.json").exists():
        (dst / "manifest.json").write_text(json.dumps(manifest))

    cells = [sw["file"][:-5] for sw in manifest["sweeps"]]
    if args.cells:
        lo, hi = (int(x) for x in args.cells.split("-"))
        cells = cells[lo:hi + 1]
    print(f"labelling {len(cells)} cells of {src}", flush=True)

    for i, name in enumerate(cells):
        s_dir, d_dir = src / name, dst / name
        if not (s_dir / "rgb.mp4").exists():
            print(f"  skip {name}: no rgb.mp4", flush=True)
            continue
        d_dir.mkdir(parents=True, exist_ok=True)
        # rgb + alpha are shared with the source cache (symlinks, no copies)
        for f in ("rgb.mp4", "alpha.npz"):
            link = d_dir / f
            if not link.exists():
                link.symlink_to(s_dir / f)
        if (d_dir / "semantic_labels.npz").exists():
            continue
        # SAM3 writes outputs/sam3_labels/<stem>.npz, keyed by the video stem —
        # give each cell a unique stem so cells never collide.
        stem = f"__cachecell_{args.scene}_{name}"
        tmp_mp4 = Path("/tmp") / f"{stem}.mp4"
        if tmp_mp4.exists():
            tmp_mp4.unlink()
        tmp_mp4.symlink_to(s_dir / "rgb.mp4")
        subprocess.run(
            [sys.executable, "sam3_precompute_labels.py", "--input_path", str(tmp_mp4),
             "--static_scene", "--conf", str(args.conf), "--overlay_every", "999"],
            check=True, cwd=str(SCRATCH / "NeoVerse"))
        raw = SCRATCH / "NeoVerse/outputs/sam3_labels" / f"{stem}.npz"
        subprocess.run(
            ["/users/jmizrahi/.conda/envs/neoverse/bin/python",
             "scripts/remap_labels_to_v14.py", "--dirs", "outputs/sam3_labels"],
            check=True, cwd=str(SCRATCH / "NeoVerse"))
        v14 = SCRATCH / "NeoVerse/outputs/sam3_labels_v14" / f"{stem}.npz"
        lab = np.load(v14)["labels"].astype(np.int8)
        np.savez_compressed(d_dir / "semantic_labels.npz", labels=lab)
        for p in (raw, v14, tmp_mp4):
            if p.exists():
                p.unlink()
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(cells)} cells", flush=True)
    print(f"==> SAM3-labelled cache at {dst}")


if __name__ == "__main__":
    main()
