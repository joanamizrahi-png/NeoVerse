"""Inventory + visualize the downloaded SANPO slice BEFORE building the v21
preprocessing on top of it (gate: Joana's eyes on the data first).

Produces, under --out:
  manifest.json        per-session: frames, masks, human vs propagated counts
  mask_format.txt      empirical mask PNG properties (dtype/channels/values)
  CONTACT_SHEET.png    grid: one labeled frame per session (rgb | mask)
  sample_<i>_<sess>.mp4  3 sessions, rgb | colorized-mask side by side

Masks are colorized with a fixed arbitrary palette (indexed by raw id) —
class NAMES come later from the official taxonomy; here we only need to SEE
whether region structure is clean and how ids are laid out.

Usage:
    python scripts/inspect_sanpo.py \
        --root /scratch/m000204-pm06b/joana/data/sanpo \
        --out /scratch/m000204-pm06b/joana/data/sanpo/inspect
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def palette(n=256):
    rng = np.random.RandomState(7)
    pal = rng.randint(30, 255, size=(n, 3), dtype=np.uint8)
    pal[0] = (0, 0, 0)
    return pal


def read_mask(path: Path):
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return m


def mask_to_ids(m: np.ndarray) -> np.ndarray:
    """SANPO panoptic PNG encoding (verified 2026-08-28): the RED channel is
    the semantic class id (labelmap.json ids 0-30); green/blue carry instance
    bytes. cv2 loads BGR, so semantic = channel 2. Reading channel 0 was the
    bug behind the phantom '82% unlabeled'."""
    if m.ndim == 3:
        m = m[..., 2]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video_sessions", type=int, default=3)
    ap.add_argument("--video_frames", type=int, default=80)
    args = ap.parse_args()
    root = Path(args.root) / "sanpo-real"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pal = palette(4096)

    sessions = sorted([p for p in root.iterdir() if p.is_dir()])
    manifest, fmt_lines = {}, []
    cells = []
    for si, s in enumerate(sessions):
        cam = s / "camera_chest" / "left"
        masks = sorted((cam / "segmentation_masks").glob("*.png"))
        frames = sorted((cam / "video_frames").glob("*.png"))
        ann_file = cam / "frame_segmentation_annotation_type.json"
        ann = {}
        if ann_file.exists():
            ann = json.load(open(ann_file))
        ann_vals = list(ann.values()) if isinstance(ann, dict) else ann
        n_human = sum(1 for v in ann_vals if "human" in str(v).lower())
        manifest[s.name] = {
            "masks": len(masks), "frames": len(frames),
            "human_annotated": n_human,
            "annotation_types": sorted({str(v) for v in ann_vals})[:6],
        }
        if not masks:
            continue

        mid = masks[len(masks) // 2]
        m = read_mask(mid)
        if si < 3 and m is not None:
            ids = mask_to_ids(m)
            fmt_lines.append(
                f"{s.name}: shape={m.shape} dtype={m.dtype} "
                f"uniq[:20]={sorted(np.unique(ids).tolist())[:20]} "
                f"(count={len(np.unique(ids))})")
            if m.dtype == np.uint16:
                fmt_lines.append(
                    f"  as id//1000 uniq={sorted(np.unique(ids // 1000).tolist())[:20]}"
                    f"  as id%1000 uniq[:20]={sorted(np.unique(ids % 1000).tolist())[:20]}")

        rgb_p = cam / "video_frames" / mid.name
        if rgb_p.exists() and m is not None:
            rgb = cv2.imread(str(rgb_p))
            ids = mask_to_ids(m)
            if ids.dtype == np.uint16 and ids.max() >= 1000:
                ids = ids // 1000
            col = pal[np.clip(ids, 0, 4095).astype(int) % 4096]
            h = 168
            rgb_s = cv2.resize(rgb, (int(rgb.shape[1] * h / rgb.shape[0]), h))
            col_s = cv2.resize(col, (rgb_s.shape[1], h),
                               interpolation=cv2.INTER_NEAREST)
            cell = np.hstack([rgb_s, col_s])
            cv2.putText(cell, f"{si:02d} {s.name[:14]}", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cells.append(cell)

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    with open(out / "mask_format.txt", "w") as f:
        f.write("\n".join(fmt_lines) + "\n")
    total_m = sum(v["masks"] for v in manifest.values())
    total_h = sum(v["human_annotated"] for v in manifest.values())
    print(f"{len(sessions)} sessions, {total_m} masked frames "
          f"({total_h} human-annotated)", flush=True)

    if cells:
        w = max(c.shape[1] for c in cells)
        cells = [cv2.copyMakeBorder(c, 0, 0, 0, w - c.shape[1],
                                    cv2.BORDER_CONSTANT) for c in cells]
        rows = [np.hstack(cells[i:i + 2]) if i + 1 < len(cells)
                else np.hstack([cells[i], np.zeros_like(cells[i])])
                for i in range(0, len(cells), 2)]
        wmax = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1],
                                   cv2.BORDER_CONSTANT) for r in rows]
        cv2.imwrite(str(out / "CONTACT_SHEET.png"), np.vstack(rows))
        print(f"contact sheet: {out / 'CONTACT_SHEET.png'}", flush=True)

    for si, s in enumerate(sessions[:args.video_sessions]):
        cam = s / "camera_chest" / "left"
        masks = sorted((cam / "segmentation_masks").glob("*.png"))
        if not masks:
            continue
        vw = None
        for mp in masks[:args.video_frames]:
            rgb_p = cam / "video_frames" / mp.name
            if not rgb_p.exists():
                continue
            rgb = cv2.imread(str(rgb_p))
            ids = mask_to_ids(read_mask(mp))
            if ids.dtype == np.uint16 and ids.max() >= 1000:
                ids = ids // 1000
            col = pal[np.clip(ids, 0, 4095).astype(int) % 4096]
            h = 336
            rgb_s = cv2.resize(rgb, (int(rgb.shape[1] * h / rgb.shape[0]), h))
            col_s = cv2.resize(col, (rgb_s.shape[1], h),
                               interpolation=cv2.INTER_NEAREST)
            fr = np.hstack([rgb_s, col_s])
            if vw is None:
                vw = cv2.VideoWriter(
                    str(out / f"sample_{si}_{s.name[:12]}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                    (fr.shape[1], fr.shape[0]))
            vw.write(fr)
        if vw is not None:
            vw.release()
            print(f"sample video {si}: {s.name}", flush=True)
    print("==> inspect done", flush=True)


if __name__ == "__main__":
    main()
