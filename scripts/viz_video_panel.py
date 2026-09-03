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


def quicktime_safe(path) -> None:
    """Re-encode so QuickTime plays it. OpenCV here has no H.264 encoder and
    falls back to mp4v, which QuickTime shows as a green screen (VLC is fine,
    which is how it hid). imageio_ffmpeg ships a usable binary. -pix_fmt
    yuv420p is required or QuickTime refuses even H.264."""
    import shutil
    import subprocess
    src = Path(path)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
    if not exe or not src.exists():
        return
    tmp = src.with_suffix(".h264.mp4")
    try:
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp)],
                       check=True, timeout=300)
        if tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(src)
    except Exception as e:
        print(f"[quicktime_safe] left as-is ({e})")
        if tmp.exists():
            tmp.unlink()


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
    # Run to the LONGEST clip, not the shortest (2026-09-02). Truncating to the
    # shortest cut the blind-vs-sighted comparison off at the exact moment the
    # blind policy crashed, hiding the 40 remaining steps of the sighted one --
    # which is the half that shows what it does instead of crashing. A clip that
    # has ended holds its last frame, dimmed, with ENDED burned in.
    frames = max(x for x in n if x > 0)
    print(f"{len(caps)} videos, {frames} frames each (counts {n}), {fps:.1f} fps")

    writer, bar_h = None, 30
    last = [None] * len(caps)          # last real frame per clip, for freezing
    done = [False] * len(caps)
    for fi in range(frames):
        tiles = []
        for ci, (c, lab) in enumerate(zip(caps, labels)):
            ok, fr = (False, None) if done[ci] else c.read()
            if not ok:
                done[ci] = True
                # a clip that already ended holds its last frame, dimmed;
                # one that never produced a frame contributes black. Heights
                # are reconciled by the padding below.
                fr = (np.zeros((args.width * 9 // 16, args.width, 3),
                               dtype=np.uint8) if last[ci] is None
                      else (last[ci] * 0.45).astype(np.uint8))
            else:
                h = int(fr.shape[0] * args.width / fr.shape[1])
                fr = cv2.resize(fr, (args.width, h))
                last[ci] = fr
            bar = np.zeros((bar_h, args.width, 3), dtype=np.uint8)
            tag = f"{lab}   ENDED at {n[ci]}" if done[ci] else f"{lab}   {fi + 1}/{n[ci]}"
            cv2.putText(bar, tag, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (140, 140, 140) if done[ci] else (255, 255, 255), 2,
                        cv2.LINE_AA)
            tiles.append(np.vstack([bar, fr]))
        if len({t.shape[1] for t in tiles}) > 1 or len({t.shape[0] for t in tiles}) > 1:
            hmax = max(t.shape[0] for t in tiles)
            tiles = [np.pad(t, ((0, hmax - t.shape[0]), (0, 0), (0, 0)))
                     for t in tiles]
        panel = np.vstack(tiles) if args.vertical else np.hstack(tiles)
        if writer is None:
            # mp4v is MPEG-4 Part 2; QuickTime frequently renders it as a green
            # screen even though VLC and ffmpeg decode it fine (2026-09-02).
            # Try H.264 first and fall back, so the file opens on a Mac.
            size = (panel.shape[1], panel.shape[0])
            for tag in ("avc1", "H264", "mp4v"):
                writer = cv2.VideoWriter(
                    args.out, cv2.VideoWriter_fourcc(*tag), fps, size)
                if writer.isOpened():
                    print(f"codec: {tag}")
                    break
                writer.release()
            if not writer.isOpened():
                raise SystemExit("no usable video codec found")
        writer.write(panel)
    for c in caps:
        c.release()
    if writer is not None:
        writer.release()
        quicktime_safe(args.out)
    print(f"==> {args.out}")


if __name__ == "__main__":
    main()
