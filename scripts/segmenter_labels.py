"""Dense pseudo-GT from a Cityscapes-trained SegFormer, in the v14 taxonomy.

Why (v18, 2026-08-24): RUGD's dense GT contains almost no person / sidewalk /
road pixels, so v10 never had a real supervisor for campus classes (person
IoU ~0). SAM3-with-hints as a target teaches hint-copying (v15's failure).
A supervised Cityscapes segmenter is near-human on exactly those classes —
run it ONCE over the campus clips and use its output as target_labels
("distill the supervisor into the world model"; inference-time semantics
remain free, from the same diffusion call as RGB).

Output: <out_dir>/<stem>.npz {"labels": int8 [T, H, W]} — same format as
rugd_gt_labels_v14, so the training dataloader treats it as dense GT
(semantic_ce_gt_only counts these clips as GT clips).

Usage:
  python scripts/segmenter_labels.py \
      --videos /scratch/.../data/scand_clips/*.mp4 /scratch/.../data/gnd_clips/gnd_*.mp4 \
      --out_dir outputs/segformer_gt_labels_v14 [--width 560 --height 336]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Cityscapes trainIds (SegFormer head order) -> v14 class ids.
CITYSCAPES_TO_V14 = {
    0: 7,    # road
    1: 6,    # sidewalk
    2: 10,   # building -> obstacle
    3: 10,   # wall -> obstacle
    4: 10,   # fence -> obstacle
    5: 10,   # pole -> obstacle
    6: 10,   # traffic light -> obstacle
    7: 10,   # traffic sign -> obstacle
    8: 11,   # vegetation
    9: 3,    # terrain -> grass
    10: 1,   # sky
    11: 12,  # person
    12: 12,  # rider -> person
    13: 13,  # car -> vehicle
    14: 13,  # truck -> vehicle
    15: 13,  # bus -> vehicle
    16: 13,  # train -> vehicle
    17: 13,  # motorcycle -> vehicle
    18: 13,  # bicycle -> vehicle
}


def read_frames(path: Path, num_frames: int):
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(total - 1, 0), num_frames).astype(int)
    want = set(idx.tolist())
    frames, t = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if t in want:
            for _ in range(int((idx == t).sum())):   # duplicated indices on short clips
                frames.append(bgr[:, :, ::-1].copy())
        t += 1
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--model", default="nvidia/segformer-b4-finetuned-cityscapes-1024-1024")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = SegformerImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(device).eval()

    lut = np.zeros(19, dtype=np.int8)
    for k, v in CITYSCAPES_TO_V14.items():
        lut[k] = v

    for video in args.videos:
        out_path = args.out_dir / f"{video.stem}.npz"
        if out_path.exists():
            print(f"skip {video.stem} (exists)", flush=True)
            continue
        frames = read_frames(video, args.num_frames)
        if not frames:
            print(f"SKIP {video.stem}: unreadable", flush=True)
            continue
        labels = []
        with torch.no_grad():
            for s in range(0, len(frames), args.batch):
                chunk = frames[s:s + args.batch]
                inputs = proc(images=chunk, return_tensors="pt").to(device)
                logits = model(**inputs).logits          # [B, 19, h/4, w/4]
                up = torch.nn.functional.interpolate(
                    logits, size=(args.height, args.width),
                    mode="bilinear", align_corners=False)
                labels.append(up.argmax(1).cpu().numpy())
        lab = lut[np.concatenate(labels).astype(np.int64)]
        np.savez_compressed(out_path, labels=lab.astype(np.int8))
        u, c = np.unique(lab, return_counts=True)
        top = sorted(zip(c, u), reverse=True)[:4]
        print(f"{video.stem}: wrote {lab.shape}, top classes "
              f"{[(int(cls), round(100*n/lab.size, 1)) for n, cls in top]}", flush=True)


if __name__ == "__main__":
    main()
