"""Several model renders as ONE labelled side-by-side video.

Comparing semantics checkpoints by opening three files and alt-tabbing does not
work — the differences are frame-local and the eye cannot hold them across
windows. One panel, same frame index, model name burned in, is the only way the
comparison is actually made rather than asserted.

Usage:
    python scripts/viz_video_panel.py --out cmp.mp4 \
        v21=/scratch/.../inference_train_semantic_v21_.../rgb.mp4 \
        v26=/scratch/.../inference_train_semantic_v26_campus_..._e5/rgb.mp4 \
        v28=/scratch/.../inference_train_semantic_v28_campus_dino_..._e5/rgb.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=/path/to/video.mp4")
    ap.add_argument("--out", default="panel.mp4")
    ap.add_argument("--width", type=int, default=560,
                    help="per-panel width; total = width * n_videos")
    ap.add_argument("--fps", type=float, default=0.0, help="0 = take from the first input")
    ap.add_argument("--vertical", action="store_true")
    args = ap.parse_args()

    import cv2
    caps, labels = [], []
    for spec in args.runs:
        label, _, p = spec.partition("=")
        if not Path(p).exists():
            print(f"[skip] {label}: missing {p}")
            continue
        c = cv2.VideoCapture(p)
        if not c.isOpened():
            print(f"[skip] {label}: cannot open {p}")
            continue
        # A file can open and still be unreadable. On the login node imageio
        # silently falls back to the TIFF writer when its ffmpeg plugin is
        # missing, producing an "mp4" whose frame count comes back as
        # -9223372036854775808 — the panel then built a column of garbage
        # instead of complaining (2026-09-02).
        nf = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        if nf <= 0:
            print(f"[skip] {label}: frame count {nf} — not a readable video "
                  f"({p}). Rewrite it with the neoverse env python.")
            c.release()
            continue
        caps.append(c)
        labels.append(label)
    if not caps:
        raise SystemExit("no readable videos — check the paths")

    fps = args.fps or (caps[0].get(cv2.CAP_PROP_FPS) or 5.0)
    n = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
    frames = min(x for x in n if x > 0)
    print(f"{len(caps)} videos, {frames} frames each (counts {n}), {fps:.1f} fps")

    writer, bar_h = None, 30
    for _ in range(frames):
        tiles = []
        for c, lab in zip(caps, labels):
            ok, fr = c.read()
            if not ok:
                break
            h = int(fr.shape[0] * args.width / fr.shape[1])
            fr = cv2.resize(fr, (args.width, h))
            bar = np.zeros((bar_h, args.width, 3), dtype=np.uint8)
            cv2.putText(bar, lab, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(np.vstack([bar, fr]))
        if len(tiles) != len(caps):
            break
        panel = np.vstack(tiles) if args.vertical else np.hstack(tiles)
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (panel.shape[1], panel.shape[0]))
        writer.write(panel)
    for c in caps:
        c.release()
    if writer is not None:
        writer.release()
    print(f"==> {args.out}")


if __name__ == "__main__":
    main()
