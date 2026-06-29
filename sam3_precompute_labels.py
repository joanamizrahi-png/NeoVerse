"""Edit #1 of semantic wiring: precompute SAM3 per-pixel labels for a clip,
ALIGNED to the exact frames NeoVerse's reconstructor will use, and save to disk.

Why a separate script: it loads SAM3 (3.3 GB) on its own, so SAM3 and the big
NeoVerse model never sit in memory at the same time (avoids OOM on the Jetson).
inference.py later just reads the cheap .npz this produces.

Alignment guarantee: we call the SAME load_video() that inference.py calls, with
the same args, so frame i here == frame i there (same sampling, same resolution).

Usage (match whatever flags you'll pass to inference.py):
  python sam3_precompute_labels.py --input_path examples/videos/driving.mp4
Output:
  outputs/sam3_labels/<clip-stem>.npz   (labels [N,H,W] int8 + class metadata)
  outputs/sam3_labels/<clip-stem>/sem_overlay_*.png   (a few, to eyeball)
"""
import os, sys, time, argparse, numpy as np
from PIL import Image
import torch
from diffsynth.utils.auxiliary import load_video
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- EDIT HERE: prompt, RGB color, traversable. Order = priority (later overwrites earlier). ---
CLASSES = [
    # Traversable
    ("road",       (128, 128, 128), True),
    ("sidewalk",   (210, 180, 140), True),
    ("grass",      (0,   180, 0),   True),
    ("path",       (139, 90,  43),  True),   # dirt path, gravel trail
    ("dirt",       (190, 150, 90),  True),   # dirt / packed earth trail
    ("gravel",     (150, 140, 120), True),   # gravel / small loose stones
    ("mulch",      (80,  50,  20),  True),   # mulch / leaf litter
    # Non-traversable, ground-level
    ("water",      (30,  80,  220), False),
    ("rock",       (190, 190, 190), False),  # rocks / boulders (creek bed)
    ("log",        (230, 170, 120), False),  # fallen logs
    ("stairs",     (180, 100, 30),  False),
    # Non-traversable, vertical
    ("building",   (140, 70,  20),  False),
    ("fence",      (100, 60,  100), False),
    ("vegetation", (34,  139, 34),  False),  # trees / tall foliage
    ("bush",       (80,  130, 50),  False),  # low bushes / shrubs
    # Dynamic obstacles
    ("car",        (0,   0,   255), False),
    ("bicycle",    (255, 165, 0),   False),
    ("person",     (255, 0,   0),   False),
    # Reference / background
    ("sky",        (135, 206, 235), False),
]
BPE = "/home/joana/joana/sam3_assets/bpe_simple_vocab_16e6.txt.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--resize_mode", choices=["center_crop", "resize"], default="center_crop")
    ap.add_argument("--static_scene", action="store_true")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--overlay_every", type=int, default=16, help="save an overlay every k frames")
    ap.add_argument("--prompts", default=None,
                    help="comma-separated SAM3 prompts to override the default CLASSES (auto colors)")
    args = ap.parse_args()

    global CLASSES
    if args.prompts:
        names = [p.strip() for p in args.prompts.split(",") if p.strip()]
        palette = [(128, 128, 128), (0, 0, 255), (255, 0, 0), (0, 180, 0),
                   (140, 70, 20), (210, 180, 140), (255, 165, 0), (160, 32, 240)]
        CLASSES = [(n, palette[i % len(palette)], False) for i, n in enumerate(names)]
        print(f"using custom prompts: {names}")

    # SAME frame loading as inference.py -> guarantees label/frame alignment
    frames = load_video(args.input_path, args.num_frames,
                        resolution=(args.width, args.height),
                        resize_mode=args.resize_mode,
                        static_scene=args.static_scene)
    N = len(frames)
    W, H = frames[0].size
    print(f"loaded {N} frames at {W}x{H} from {args.input_path}", flush=True)

    stem = os.path.splitext(os.path.basename(args.input_path))[0]
    out_dir = os.path.join("outputs/sam3_labels", stem)
    os.makedirs(out_dir, exist_ok=True)

    print("building SAM3...", flush=True)
    model = build_sam3_image_model(bpe_path=BPE, device="cuda", load_from_HF=True)
    proc = Sam3Processor(model, device="cuda", confidence_threshold=args.conf)

    labels = np.zeros((N, H, W), dtype=np.int8)   # 0 = unlabeled, 1..C = class
    colors = np.array([(0, 0, 0)] + [c for _, c, _ in CLASSES], dtype=np.uint8)
    t0 = time.time()
    for fi, img in enumerate(frames):
        img = img.convert("RGB")
        cmap = np.zeros((H, W), dtype=np.int8)
        state = proc.set_image(img)
        for ci, (name, _color, _trav) in enumerate(CLASSES, start=1):
            state = proc.set_text_prompt(name, state)
            m = state["masks"]
            if m.shape[0] > 0:
                cmap[m.any(dim=0).squeeze(0).cpu().numpy()] = ci   # priority overwrite
        labels[fi] = cmap
        if fi % args.overlay_every == 0:
            base = np.array(img).astype(np.float32)
            lab = cmap > 0
            ov = base.copy()
            ov[lab] = 0.45 * base[lab] + 0.55 * colors[cmap][lab]
            Image.fromarray(ov.astype(np.uint8)).save(os.path.join(out_dir, f"sem_overlay_{fi:03d}.png"))
        if fi % 10 == 0:
            print(f"  frame {fi}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    out_npz = os.path.join("outputs/sam3_labels", f"{stem}.npz")
    np.savez_compressed(
        out_npz,
        labels=labels,                                   # [N,H,W] int8
        class_names=np.array(["unlabeled"] + [n for n, _, _ in CLASSES]),
        class_colors=colors,                             # [C+1,3]
        traversable=np.array([False] + [t for _, _, t in CLASSES]),
        num_frames=N, height=H, width=W,
    )
    print(f"\nsaved {out_npz}  (labels {labels.shape}, {len(CLASSES)} classes + background)", flush=True)
    print(f"overlays in {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
