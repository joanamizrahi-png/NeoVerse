"""SANPO frames through the robot's eyes: RGB | v14 palette | traversability.

The coherence gate Joana asked for (2026-08-28): after the 31->14 mapping and
the grass/pavement score change, does a SANPO street read as a sensible cost
surface? Green = preferred ground, yellow = allowed, red = never touch.

Outputs under --out:
  V14_PREVIEW_SHEET.png       ~16 frames across sessions, 3 panels each
  v14_preview_<session>.mp4   one session driven through, 3 panels

Usage:
    python scripts/sanpo_v14_preview.py \
        --sanpo /scratch/m000204-pm06b/joana/data/sanpo \
        --out /scratch/m000204-pm06b/joana/data/sanpo/inspect
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from convert_sanpo_clips import SANPO_TO_V14

# Load the taxonomy FILE directly — importing the diffsynth package drags in
# modelscope etc., unavailable outside the neoverse env (login-node friendly).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "class_taxonomy",
    Path(__file__).resolve().parents[1] / "diffsynth" / "utils" / "class_taxonomy.py")
_tax = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tax)
V14 = _tax.V14

PAL = np.array([c for _, c, _ in V14], dtype=np.uint8)          # RGB
SCORES = np.array([s for _, _, s in V14], dtype=np.float32)


def trav_heat(score: np.ndarray) -> np.ndarray:
    """score in [0,1] -> BGR heat: 0 red, 0.5 yellow, 1 green."""
    s = np.clip(score, 0.0, 1.0)
    r = np.where(s < 0.5, 255, (1.0 - (s - 0.5) * 2) * 255)
    g = np.where(s < 0.5, s * 2 * 255, 255)
    heat = np.stack([np.zeros_like(r), g, r], axis=-1)          # BGR
    return heat.astype(np.uint8)


def three_panel(rgb_bgr: np.ndarray, sem_ids: np.ndarray, h: int) -> np.ndarray:
    v14 = SANPO_TO_V14[sem_ids]
    pal_bgr = PAL[v14][:, :, ::-1]
    heat = trav_heat(SCORES[v14])
    heat = (0.65 * heat + 0.35 * rgb_bgr).astype(np.uint8)      # ghost the scene
    cells = []
    for img, txt in ((rgb_bgr, "rgb"), (pal_bgr, "v14 classes"),
                     (heat, "traversability")):
        img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        img = np.ascontiguousarray(img)
        cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cells.append(img)
    return np.hstack(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanpo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet_frames", type=int, default=16)
    ap.add_argument("--video_frames", type=int, default=80)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    masks = list(Path(args.sanpo).glob(
        "sanpo-real/*/camera_chest/left/segmentation_masks/*.png"))
    random.seed(1)
    rows = []
    for p in random.sample(masks, min(args.sheet_frames, len(masks))):
        rgb_p = p.parent.parent / "video_frames" / p.name
        rgb = cv2.imread(str(rgb_p))
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if rgb is None or m is None:
            continue
        sem = m[..., 2] if m.ndim == 3 else m
        rows.append(three_panel(rgb, sem, h=190))
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 2, 2, 0, wmax - r.shape[1],
                               cv2.BORDER_CONSTANT) for r in rows]
    cv2.imwrite(str(out / "V14_PREVIEW_SHEET.png"), np.vstack(rows))
    print(f"sheet: {out / 'V14_PREVIEW_SHEET.png'}", flush=True)

    sess = sorted({p.parents[3] for p in masks})[0]
    cam = sess / "camera_chest" / "left"
    vw = None
    for mp in sorted((cam / "segmentation_masks").glob("*.png"))[:args.video_frames]:
        rgb = cv2.imread(str(cam / "video_frames" / mp.name))
        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if rgb is None or m is None:
            continue
        sem = m[..., 2] if m.ndim == 3 else m
        fr = three_panel(rgb, sem, h=280)
        if vw is None:
            vw = cv2.VideoWriter(
                str(out / f"v14_preview_{sess.name[:12]}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                (fr.shape[1], fr.shape[0]))
        vw.write(fr)
    if vw is not None:
        vw.release()
        print(f"video: v14_preview_{sess.name[:12]}.mp4", flush=True)
    print("==> v14 preview done", flush=True)


if __name__ == "__main__":
    main()
