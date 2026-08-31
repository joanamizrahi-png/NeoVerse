"""SANPO HELD-OUT VAL SET — the GT-numbers gate (Joana 2026-08-29: "is there
no way to have numbers with actual GT? necessary for the paper").

Takes the LAST labeled 81-frame window of each session — the training
converter took the FIRST clips_per_session windows, so with a >=8-window
guard the val clips are verifiably disjoint from every training diet built
from the front (v21 roots at 3/session; the 'big' diet at 6/session).

Outputs sanpo_val/{clips,labels}/sanpoval_<sess>_<start>.{mp4,npz} and
symlinks each clip into rugd_clips/ + its GT labels into sam3_labels_v14/
so the standard render launcher works on them unchanged. Grade with:
  eval_semantic_accuracy.py <render>/semantic_labels.npz sanpo_val/labels/<clip>.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from convert_sanpo_clips import SANPO_TO_V14, crop_resize, W, H, N_FRAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanpo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_windows", type=int, default=8,
                    help="skip sessions with fewer labeled windows (keeps val "
                         "disjoint from front-picked training diets up to 6)")
    ap.add_argument("--clips_dir",
                    default="/scratch/m000204-pm06b/joana/data/rugd_clips")
    ap.add_argument("--labels_dir",
                    default="/scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels_v14")
    ap.add_argument("--max_sessions", type=int, default=20)
    ap.add_argument("--window_back", type=int, default=1,
                    help="1 = last window per session, 2 = second-to-last, ... "
                         "(2026-08-31: widen the val jury beyond 2 clips; "
                         "training diets take the FIRST 6, so back<=2 with "
                         "min_windows>=8 stays disjoint)")
    args = ap.parse_args()

    out = Path(args.out)
    clips_dir = out / "clips"
    labels_dir = out / "labels"
    clips_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    made = []
    for s in sorted((Path(args.sanpo) / "sanpo-real").iterdir()):
        if len(made) >= args.max_sessions:
            break
        cam = s / "camera_chest" / "left"
        masks = sorted((cam / "segmentation_masks").glob("*.png"))
        idx = [int(p.stem) for p in masks]
        # longest contiguous run (mirror of the training converter's scan)
        runs, start = [], 0
        for k in range(1, len(idx) + 1):
            if k == len(idx) or idx[k] != idx[k - 1] + 1:
                if k - start >= N_FRAMES:
                    runs.append((start, k))
                start = k
        if not runs:
            continue
        r0, r1 = max(runs, key=lambda r: r[1] - r[0])
        if (r1 - r0) < args.min_windows * N_FRAMES:
            continue                          # too short: could collide with
        w0 = r1 - args.window_back * N_FRAMES  # back from the run's end
        if w0 < r0:
            continue
        name = f"sanpoval_{s.name[:10]}_{idx[w0]:06d}"
        clip_p = clips_dir / f"{name}.mp4"
        lab_p = labels_dir / f"{name}.npz"
        if not (clip_p.exists() and lab_p.exists()):
            frames, labs, ok = [], [], True
            for k in range(w0, w0 + N_FRAMES):
                img = cv2.imread(str(cam / "video_frames" / masks[k].name))
                m = cv2.imread(str(masks[k]), cv2.IMREAD_UNCHANGED)
                if img is None or m is None:
                    ok = False
                    break
                sem = m[..., 2] if m.ndim == 3 else m
                frames.append(crop_resize(img, nearest=False))
                labs.append(SANPO_TO_V14[crop_resize(sem, nearest=True)])
            if not ok:
                continue
            vw = cv2.VideoWriter(str(clip_p), cv2.VideoWriter_fourcc(*"mp4v"),
                                 15.0, (W, H))
            for f in frames:
                vw.write(f)
            vw.release()
            np.savez_compressed(lab_p, labels=np.stack(labs).astype(np.int8))
        for src, dst_dir, suff in ((clip_p, Path(args.clips_dir), ".mp4"),
                                   (lab_p, Path(args.labels_dir), ".npz")):
            dst = dst_dir / f"{name}{suff}"
            if not dst.exists():
                dst.symlink_to(src)
        made.append(name)
        print(f"  {name}", flush=True)

    print(f"==> {len(made)} val clips; render with CLIP=<name>, grade vs "
          f"{labels_dir}/<name>.npz", flush=True)


if __name__ == "__main__":
    main()
