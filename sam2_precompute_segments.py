"""Precompute class-AGNOSTIC SAM2 segments per clip (v8 Change 3 preprocessing).

Purpose: the segment-homogeneity loss needs to know which pixels belong
together — not what class they are. SAM2's automatic mask generation gives
exactly that. During training, predicted class probabilities are penalized for
DISAGREEING within a segment; speckle is by definition such a disagreement, so
this loss pays directly for its removal without needing correct-class info.

Alignment guarantee: same load_video() as sam3_precompute_labels.py /
inference.py, so segment maps align 1:1 with the frames the reconstructor and
dataloader consume.

Usage:
  python sam2_precompute_segments.py --input_path <clip.mp4> --num_frames 81
Output:
  outputs/sam2_segments/<clip-stem>.npz
    segments    [N, H, W] int16  (0 = unassigned, 1..K segment ids PER FRAME —
                                  ids are NOT tracked across frames; the
                                  homogeneity loss is per-frame)
    num_frames, height, width
  outputs/sam2_segments/<clip-stem>/seg_overlay_*.png  (a few, to eyeball)
"""
import os, sys, time, argparse
import numpy as np
from PIL import Image
import torch

# Reuse the canonical frame loader (identical sampling to the dataloader).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam3_precompute_labels import load_video


def masks_to_id_map(masks, h, w):
    """Paint a list of boolean masks into one int16 id map.

    Sorted by area DESCENDING before painting, so smaller (more specific)
    segments overwrite larger ones — same override principle as the SAM3
    priority scheme. 0 stays "unassigned"."""
    id_map = np.zeros((h, w), dtype=np.int16)
    order = sorted(range(len(masks)), key=lambda i: masks[i].sum(), reverse=True)
    seg_id = 0
    for i in order:
        seg_id += 1
        id_map[masks[i]] = seg_id
    return id_map


def overlay(img, id_map, alpha=0.55, seed=0):
    rng = np.random.default_rng(seed)
    colors = rng.integers(40, 255, size=(int(id_map.max()) + 1, 3), dtype=np.uint8)
    colors[0] = 0
    seg_rgb = colors[id_map]
    base = np.asarray(img).astype(np.float32)
    out = base * (1 - alpha) + seg_rgb.astype(np.float32) * alpha
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--resize_mode", choices=["center_crop", "resize"], default="center_crop")
    ap.add_argument("--static_scene", action="store_true")
    ap.add_argument("--model_id", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--points_per_side", type=int, default=16)
    ap.add_argument("--pred_iou_thresh", type=float, default=0.7)
    ap.add_argument("--stability_score_thresh", type=float, default=0.85)
    ap.add_argument("--points_per_batch", type=int, default=64)
    ap.add_argument("--overlay_every", type=int, default=16)
    ap.add_argument("--out_dir", default="outputs/sam2_segments")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.input_path))[0]
    out_npz = os.path.join(args.out_dir, f"{stem}.npz")
    if os.path.exists(out_npz):
        print(f"[skip] {out_npz} exists")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, stem), exist_ok=True)

    frames = load_video(args.input_path, args.num_frames,
                        resolution=(args.width, args.height),
                        resize_mode=args.resize_mode, static_scene=args.static_scene)
    print(f"{stem}: {len(frames)} frames @ {frames[0].size}", flush=True)

    from transformers import pipeline
    generator = pipeline(
        "mask-generation", model=args.model_id,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.bfloat16,
    )

    seg_maps = []
    t0 = time.time()
    for n, img in enumerate(frames):
        out = generator(
            img,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            points_per_batch=args.points_per_batch,
        )
        masks = [np.asarray(m, dtype=bool) for m in out["masks"]]
        id_map = masks_to_id_map(masks, args.height, args.width)
        seg_maps.append(id_map)
        if n % args.overlay_every == 0:
            overlay(img, id_map).save(
                os.path.join(args.out_dir, stem, f"seg_overlay_{n:03d}.png"))
            print(f"  frame {n:3d}: {len(masks)} masks, "
                  f"{(id_map > 0).mean():.0%} covered, "
                  f"{(time.time() - t0) / (n + 1):.1f} s/frame", flush=True)

    segments = np.stack(seg_maps)
    np.savez_compressed(out_npz, segments=segments,
                        num_frames=len(frames), height=args.height, width=args.width)
    print(f"wrote {out_npz} ({segments.max()} max segs/frame, "
          f"{(segments > 0).mean():.0%} mean coverage, "
          f"{time.time() - t0:.0f} s total)", flush=True)


if __name__ == "__main__":
    main()
