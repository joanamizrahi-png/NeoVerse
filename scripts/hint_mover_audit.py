"""Audit of the training signal for movers (2026-09-19, Joana: "can we verify
that that is actually what happens?").

Hypothesis: with people/vehicles kept PER-FRAME in the hint, a person present
in the target frame is often absent from (or void in) the hint, so the model is
taught "thin hint here -> paint a person" -- the phantom-on-pavement failure.

Reads the debug dump of a training run (debug_save_root/<dataset>/<clip>/
gt_semantic_hint.mp4 + gt_semantic_target.mp4, colorized with palette 6),
decodes both back to class ids by nearest palette color, and reports over all
dumped frames:
  base rates:   share of target pixels that are person/vehicle; share of hint pixels void
  given TARGET = person/vehicle (the pixels that teach the model about movers):
     hint = same class   |  hint = void  |  hint = ground/other
  given TARGET = walkable ground:  hint = person/vehicle  (the reverse error)
Compare the two dumps (per-frame vs fused). numpy + cv2 only.

    python scripts/hint_mover_audit.py <dump_root> [<dump_root2> ...]
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys

import cv2
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("class_taxonomy", os.path.join(_root, "diffsynth/utils/class_taxonomy.py"))
_tax = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_tax)
PAL = np.asarray(_tax.V14_V6, np.float32)          # [14,3]
MOVERS = (12, 13)
WALK = (2, 6, 7, 8)
SUB = 4


def decode(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path); frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        # 2026-09-20: every 4th pixel in each direction (the audit reports
        # ratios; 16x fewer pixels made the login-node run minutes, not an hour)
        f = f[::SUB, ::SUB]
        rgb = f[:, :, ::-1].astype(np.float32)
        d = ((rgb[:, :, None, :] - PAL[None, None, :, :]) ** 2).sum(-1)      # [h,w,14]
        frames.append(d.argmin(-1).astype(np.int8))
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 1, 1), np.int8)


def audit(root: str):
    clips = sorted(glob.glob(os.path.join(root, "**", "gt_semantic_target.mp4"), recursive=True))
    tot = {"px": 0, "tgt_mover": 0, "hint_void": 0, "tm_hint_same": 0, "tm_hint_void": 0, "tm_hint_other": 0,
           "tgt_walk": 0, "tw_hint_mover": 0}
    n_clips = 0
    for t in clips:
        h = t.replace("gt_semantic_target.mp4", "gt_semantic_hint.mp4")
        if not os.path.exists(h):
            continue
        T, Hn = decode(t), decode(h)
        n = min(len(T), len(Hn))
        if n == 0:
            continue
        T, Hn = T[:n], Hn[:n]
        if T.shape != Hn.shape:
            Hn = np.stack([cv2.resize(x, (T.shape[2], T.shape[1]), interpolation=cv2.INTER_NEAREST) for x in Hn])
        tm = np.isin(T, MOVERS); tw = np.isin(T, WALK); hv = Hn == 0; hm = np.isin(Hn, MOVERS)
        tot["px"] += T.size; tot["tgt_mover"] += int(tm.sum()); tot["hint_void"] += int(hv.sum())
        tot["tm_hint_same"] += int((tm & hm).sum()); tot["tm_hint_void"] += int((tm & hv).sum())
        tot["tm_hint_other"] += int((tm & ~hm & ~hv).sum())
        tot["tgt_walk"] += int(tw.sum()); tot["tw_hint_mover"] += int((tw & hm).sum())
        n_clips += 1
        print(f"   [{n_clips}/{len(clips)}] {os.path.basename(os.path.dirname(t))}: {n} frames", flush=True)
    return n_clips, tot


def main():
    for root in sys.argv[1:]:
        n, t = audit(root)
        name = os.path.basename(root.rstrip("/"))
        if t["px"] == 0:
            print(f"== {name}: no frames found under {root}"); continue
        tm = max(t["tgt_mover"], 1); tw = max(t["tgt_walk"], 1)
        print(f"== {name}: {n} clips, {t['px'] / 1e6:.1f} M pixels")
        print(f"   target person/vehicle share {100 * t['tgt_mover'] / t['px']:.2f}% | hint void share {100 * t['hint_void'] / t['px']:.1f}%")
        print(f"   given TARGET = person/vehicle: hint same class {100 * t['tm_hint_same'] / tm:.1f}% | hint void {100 * t['tm_hint_void'] / tm:.1f}% | hint other class {100 * t['tm_hint_other'] / tm:.1f}%")
        print(f"   given TARGET = walkable ground: hint says person/vehicle {100 * t['tw_hint_mover'] / tw:.2f}%")


if __name__ == "__main__":
    main()
