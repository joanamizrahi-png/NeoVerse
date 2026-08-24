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

def v14_of_label_name(name: str) -> int:
    """Map any segmenter's class NAME to a v14 id (works for Cityscapes's 19
    classes and Mapillary Vistas's 65 alike — the bake-off needs both)."""
    n = name.lower()
    def has(*words):
        return any(w in n for w in words)
    if has("person", "rider", "pedestrian", "cyclist", "motorcyclist"):
        return 12
    if has("car", "truck", "bus", "train", "vehicle", "motorcycle", "bicycle",
           "caravan", "trailer", "boat"):
        return 13
    if has("sidewalk", "curb", "crosswalk", "pedestrian area"):
        return 6
    if has("bike lane", "road", "lane marking", "parking", "rail track",
           "service lane"):
        return 7
    if has("vegetation", "tree"):
        return 11
    if has("terrain", "grass", "sand", "mountain"):
        return 3
    if has("sky"):
        return 1
    if has("water"):
        return 5
    if has("snow", "gravel"):
        return 4
    return 10   # building, wall, fence, pole, sign, ... -> obstacle


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
    ap.add_argument("--model", default="facebook/mask2former-swin-large-cityscapes-semantic",
                    help="any HF semantic-segmentation model with safetensors; "
                         "bake-off rival: facebook/mask2former-swin-large-mapillary-vistas-semantic")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model, use_safetensors=True).to(device).eval()

    id2label = model.config.id2label
    n_cls = max(int(k) for k in id2label) + 1
    lut = np.full(n_cls, 10, dtype=np.int8)
    for k, name in id2label.items():
        lut[int(k)] = v14_of_label_name(str(name))

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
                out = model(**inputs)
                seg = proc.post_process_semantic_segmentation(
                    out, target_sizes=[(args.height, args.width)] * len(chunk))
                labels.append(np.stack([m.cpu().numpy() for m in seg]))
        lab = lut[np.concatenate(labels).astype(np.int64)]
        np.savez_compressed(out_path, labels=lab.astype(np.int8))
        u, c = np.unique(lab, return_counts=True)
        top = sorted(zip(c, u), reverse=True)[:4]
        print(f"{video.stem}: wrote {lab.shape}, top classes "
              f"{[(int(cls), round(100*n/lab.size, 1)) for n, cls in top]}", flush=True)


if __name__ == "__main__":
    main()
