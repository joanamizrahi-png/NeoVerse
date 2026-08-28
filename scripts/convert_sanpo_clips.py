"""SANPO -> v21 training flashcards.

Cuts each labeled SANPO chest-camera session into 81-frame clips at 560x336,
translates masks from SANPO's 31-class taxonomy to our v14 ids, and builds
the v21 dataset roots (symlinking the RUGD flashcards alongside) so training
consumes SANPO clips exactly like RUGD ones — as DENSE HUMAN GT.

Mapping locked with Joana (2026-08-28 gallery review):
  curb->obstacle, crosswalk->sidewalk, paved trail->sidewalk,
  other walkable->pavement, terrain->grass, unlabeled->void (loss-masked).
GT-as-hint caveat: SANPO clips use their GT as the gaussian hint too (RUGD
uses SAM3 hints + human GT targets); the 3D projection still degrades the
hint, so the inpaint-and-correct task survives. Honest limitation, noted.

Usage (CPU, ~1h):
    python scripts/convert_sanpo_clips.py \
        --sanpo /scratch/m000204-pm06b/joana/data/sanpo \
        --out /scratch/m000204-pm06b/joana/data/sanpo_v21 \
        --v15_root /scratch/m000204-pm06b/joana/combined_train_data_v15 \
        --sam3_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels_v14 \
        --gt_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/gt_labels_v18 \
        --clips_per_session 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

W, H = 560, 336
N_FRAMES = 81

# SANPO labelmap.json id -> v14 id
SANPO_TO_V14 = np.zeros(256, dtype=np.int8)
SANPO_TO_V14[:31] = [
    0,   # 0  unlabeled        -> void
    7,   # 1  road             -> road
    10,  # 2  curb             -> obstacle (drop edge; do not drive over)
    6,   # 3  sidewalk         -> sidewalk
    10,  # 4  guard rail       -> obstacle
    6,   # 5  crosswalk        -> sidewalk (walk-preferred by function)
    6,   # 6  paved trail      -> sidewalk (pedestrian function)
    10,  # 7  building         -> obstacle
    10,  # 8  wall/fence       -> obstacle
    10,  # 9  hand rail        -> obstacle
    10,  # 10 opening-door     -> obstacle
    10,  # 11 opening-gate     -> obstacle
    12,  # 12 pedestrian       -> person
    12,  # 13 rider            -> person
    12,  # 14 animal           -> person (moving agent)
    9,   # 15 stairs           -> stairs
    5,   # 16 water body       -> water
    8,   # 17 other walkable   -> pavement (function unknown)
    10,  # 18 inaccessible     -> obstacle
    10,  # 19 railway track    -> obstacle
    10,  # 20 obstacle         -> obstacle
    13,  # 21 vehicle          -> vehicle
    10,  # 22 traffic sign     -> obstacle
    10,  # 23 traffic light    -> obstacle
    10,  # 24 pole             -> obstacle
    10,  # 25 bus stop         -> obstacle
    10,  # 26 bike rack        -> obstacle
    1,   # 27 sky              -> sky
    10,  # 28 tree             -> obstacle
    11,  # 29 vegetation       -> vegetation
    3,   # 30 terrain          -> grass
]


def crop_resize(img: np.ndarray, nearest: bool) -> np.ndarray:
    h, w = img.shape[:2]
    tw = int(round(h * W / H))
    if tw <= w:
        x0 = (w - tw) // 2
        img = img[:, x0:x0 + tw]
    else:
        th = int(round(w * H / W))
        y0 = (h - th) // 2
        img = img[y0:y0 + th]
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(img, (W, H), interpolation=interp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanpo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v15_root", required=True)
    ap.add_argument("--sam3_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--clips_per_session", type=int, default=3)
    ap.add_argument("--max_void_frac", type=float, default=0.25,
                    help="skip windows whose mean void share exceeds this")
    args = ap.parse_args()

    out = Path(args.out)
    clips_dir = out / "clips"
    labels_dir = out / "labels"
    clips_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    sessions = sorted((Path(args.sanpo) / "sanpo-real").iterdir())
    made = []
    for s in sessions:
        cam = s / "camera_chest" / "left"
        masks = sorted((cam / "segmentation_masks").glob("*.png"))
        idx = [int(p.stem) for p in masks]
        if len(idx) < N_FRAMES:
            continue
        # contiguous labeled runs
        runs, start = [], 0
        for k in range(1, len(idx) + 1):
            if k == len(idx) or idx[k] != idx[k - 1] + 1:
                if k - start >= N_FRAMES:
                    runs.append((start, k))
                start = k
        windows = []
        for r0, r1 in runs:
            span = r1 - r0
            n = min(args.clips_per_session - len(windows),
                    max(1, span // N_FRAMES))
            for j in range(n):
                w0 = r0 + j * max(N_FRAMES, (span - N_FRAMES) // max(1, n))
                if w0 + N_FRAMES <= r1:
                    windows.append(w0)
            if len(windows) >= args.clips_per_session:
                break

        for w0 in windows[:args.clips_per_session]:
            name = f"sanpo_{s.name[:10]}_{idx[w0]:06d}"
            clip_p = clips_dir / f"{name}.mp4"
            lab_p = labels_dir / f"{name}.npz"
            if clip_p.exists() and lab_p.exists():
                made.append(name)
                continue
            frames, labs = [], []
            ok = True
            for k in range(w0, w0 + N_FRAMES):
                mp = masks[k]
                fp = cam / "video_frames" / mp.name
                img = cv2.imread(str(fp))
                m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
                if img is None or m is None:
                    ok = False
                    break
                sem = m[..., 2] if m.ndim == 3 else m
                frames.append(crop_resize(img, nearest=False))
                labs.append(SANPO_TO_V14[crop_resize(sem, nearest=True)])
            if not ok:
                continue
            labs = np.stack(labs)
            void = float((labs == 0).mean())
            if void > args.max_void_frac:
                print(f"  skip {name}: void {void:.2f}", flush=True)
                continue
            vw = cv2.VideoWriter(str(clip_p), cv2.VideoWriter_fourcc(*"mp4v"),
                                 15.0, (W, H))
            for f in frames:
                vw.write(f)
            vw.release()
            np.savez_compressed(lab_p, labels=labs.astype(np.int8))
            made.append(name)
            print(f"  {name}  void={void:.3f}", flush=True)

    print(f"==> {len(made)} sanpo clips", flush=True)

    # ---- v21 dataset roots: RUGD (dense GT) + SANPO (dense GT), no pseudo ----
    root21 = out / "combined_train_data_v21"
    sam21 = out / "sam3_labels_v21"
    gt21 = out / "gt_labels_v21"
    for d in (root21, sam21, gt21):
        d.mkdir(parents=True, exist_ok=True)

    def link(src: Path, dst: Path):
        if not dst.exists() and src.exists():
            dst.symlink_to(src)

    n_rugd = 0
    for clip in sorted(Path(args.v15_root).glob("rugd_*.mp4")):
        stem = clip.stem
        gt = Path(args.gt_dir) / f"{stem}.npz"
        if not gt.exists():
            continue                      # GT-less rugd clips stay out of v21
        link(clip, root21 / clip.name)
        link(Path(args.sam3_dir) / f"{stem}.npz", sam21 / f"{stem}.npz")
        link(gt, gt21 / f"{stem}.npz")
        n_rugd += 1
    for name in made:
        link(clips_dir / f"{name}.mp4", root21 / f"{name}.mp4")
        link(labels_dir / f"{name}.npz", sam21 / f"{name}.npz")   # GT as hint
        link(labels_dir / f"{name}.npz", gt21 / f"{name}.npz")    # GT target
    print(f"==> v21 roots: {n_rugd} rugd + {len(made)} sanpo clips", flush=True)
    print(f"    {root21}\n    {sam21}\n    {gt21}", flush=True)


if __name__ == "__main__":
    main()
