"""What does SANPO actually call 'crosswalk', 'curb', 'terrain', 'paved
trail'...? For each class of interest, find the frames where it covers the
most pixels and tint those pixels on the real RGB — the ground-truth-by-eyes
gate for the 31->14 mapping decisions.

Usage:
    python scripts/sanpo_class_gallery.py \
        --root /scratch/m000204-pm06b/joana/data/sanpo \
        --out /scratch/m000204-pm06b/joana/data/sanpo/inspect
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

CLASSES = {
    "crosswalk": 5, "curb": 2, "terrain": 30, "paved_trail": 6,
    "sidewalk": 3, "road": 1, "other_walkable": 17,
}
TINT = (0, 0, 255)          # red tint (BGR)
PER_CLASS = 3               # example frames per class
SCAN = 500                  # masks sampled for the search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root) / "sanpo-real"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    masks = list(root.glob("*/camera_chest/left/segmentation_masks/*.png"))
    random.seed(0)
    sample = random.sample(masks, min(SCAN, len(masks)))

    shares = {k: [] for k in CLASSES}
    for p in sample:
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        sem = m[..., 2] if m.ndim == 3 else m
        for name, cid in CLASSES.items():
            sh = float((sem == cid).mean())
            if sh > 0.005:
                shares[name].append((sh, p))

    rows = []
    for name, cid in CLASSES.items():
        best = sorted(shares[name], reverse=True)[:PER_CLASS]
        cells = []
        for sh, p in best:
            rgb_p = p.parent.parent / "video_frames" / p.name
            rgb = cv2.imread(str(rgb_p))
            m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if rgb is None or m is None:
                continue
            sem = m[..., 2] if m.ndim == 3 else m
            sel = sem == cid
            over = rgb.copy()
            over[sel] = (0.45 * rgb[sel] + 0.55 * np.array(TINT)).astype(np.uint8)
            edges = cv2.dilate(sel.astype(np.uint8), np.ones((5, 5), np.uint8)) - sel
            over[edges > 0] = (255, 255, 255)
            h = 220
            over = cv2.resize(over, (int(over.shape[1] * h / over.shape[0]), h))
            cv2.putText(over, f"{name} ({sh * 100:.0f}%)", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cells.append(over)
        if not cells:
            print(f"{name}: no examples found above threshold", flush=True)
            continue
        w = max(c.shape[1] for c in cells)
        cells = [cv2.copyMakeBorder(c, 0, 0, 0, w - c.shape[1],
                                    cv2.BORDER_CONSTANT) for c in cells]
        rows.append(np.hstack(cells))
        print(f"{name}: {len(cells)} examples", flush=True)

    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 4, 4, 0, wmax - r.shape[1],
                               cv2.BORDER_CONSTANT) for r in rows]
    cv2.imwrite(str(out / "CLASS_GALLERY.png"), np.vstack(rows))
    print(f"==> {out / 'CLASS_GALLERY.png'}", flush=True)


if __name__ == "__main__":
    main()
