"""Resample a SAM3 label npz to N frames (nearest-index).

Why: inference_semantic asserts label frames == video frames. Rendering a cache
cell longer than the 81-frame default (--num_frames 153, the "supercell" test:
one generation covering 17 positions x 9 headings instead of 9 x 9) therefore
needs labels at that length too. The source clip has only ~81 real frames, so
frames are repeated by nearest index — fine, because the labels are a
conditioning hint, not a target.

    python scripts/resample_labels.py --src outputs/sam3_labels_v14/rugd_trail_00.npz \
        --n 153 --out outputs/sam3_labels_v14/rugd_trail_00_n153.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--n", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    d = np.load(args.src)
    key = [k for k in d.files if "lab" in k.lower()][0]
    lab = d[key]
    idx = np.clip(np.round(np.linspace(0, len(lab) - 1, args.n)).astype(int),
                  0, len(lab) - 1)
    out = lab[idx]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, labels=out.astype(np.int8))
    print(f"{args.src.name}: {len(lab)} -> {len(out)} frames  ->  {args.out}")


if __name__ == "__main__":
    main()
