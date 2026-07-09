"""Precompute per-pixel SAM 3 labels for a clip using HuggingFace transformers.

Rewritten (2026-07-09) from the facebookresearch/sam3-based version, which hit
5+ layered dtype mismatches when trying to use SAM 3.1's checkpoint on Marlowe's
driver stack. The community-standard is HF's Sam3Model / Sam3Processor: dtype
handled cleanly as a first-class arg, tested against many envs, and it supports
the "efficient multi-prompt on one image" pattern by exposing get_vision_features.

For our 29-class taxonomy on 81 frames this cuts vision-encoder work ~29x
(vision features computed once per frame, then reused across all class prompts).

Alignment guarantee: same load_video() function as before -> the sampled frame
indices are identical to inference.py, so the labels stay aligned 1:1 with the
frames the NeoVerse reconstructor consumes.

Usage:
  python sam3_precompute_labels.py --input_path examples/videos/driving.mp4 --num_frames 81
Output (schema unchanged so the diffusion dataloader keeps working):
  outputs/sam3_labels/<clip-stem>.npz
    labels        [N, H, W] int8   (0=void, 1..29 our taxonomy)
    class_names   [30] str
    class_colors  [30, 3] uint8
    traversable   [30] bool
    num_frames, height, width
  outputs/sam3_labels/<clip-stem>/sem_overlay_*.png   (a few, to eyeball)
"""
import os, sys, time, argparse
import numpy as np
from PIL import Image
import torch
from decord import VideoReader
from transformers import Sam3Model, Sam3Processor

# --- EDIT HERE: prompt, RGB color, traversable. Order = priority (later overwrites earlier). ---
# 29-class outdoor Go2W taxonomy (class ids 1..29; void=0 handled as "unlabeled").
# Colors MUST stay identical to diffsynth/utils/semantics.py CLASS_COLORS[1:] and TRAVERSABLE
# flags MUST match nav-rl/src/env/reward.py TRAVERSABLE. Ordering encodes SAM3 priority (later
# wins) AND the class-id space -- all three files must be updated in lockstep.
#
# Priority layering (bottom -> top):
#   sky -> ground materials -> ground hazards -> pavement materials -> pavement functions ->
#   large statics -> vegetation -> small verticals + stairs -> dynamic objects.
# Dynamics come last so a person/vehicle on a road stays labeled person/vehicle.
CLASSES = [
    # ambient
    ("sky",           (200, 225, 245), False),
    # ground materials (default ground layer)
    ("dirt",          (139,  90,  43), True),   # FLAG: revisit -- may not be Go2W-traversable when loose
    ("sand",          (230, 200, 155), True),   # FLAG: revisit -- loose sand may not be Go2W-traversable
    ("grass",         ( 75, 190,  80), True),
    ("gravel",        (180, 155, 100), True),
    ("mulch",         (110,  55,  25), True),
    # ground hazards
    ("mud",           ( 55,  55,  30), False),
    ("water",         ( 50, 120, 200), False),
    ("rock",          (135, 145, 155), False),
    # pavement materials
    ("asphalt",       ( 55,  55,  65), True),
    ("concrete",      (225, 220, 190), True),
    # pavement functions (override materials for paved surfaces)
    ("road",          (110, 110, 115), True),
    ("sidewalk",      (180, 180, 180), True),
    ("crosswalk",     (255, 250, 235), True),
    # large vertical statics
    ("building",      (170,  75,  60), False),
    ("wall",          (175, 145, 175), False),
    ("fence",         ( 90,  60, 130), False),
    ("bridge",        ( 75, 155, 175), True),
    # vegetation
    ("tree",          ( 40, 105,  55), False),
    ("vegetation",    (170, 200,  55), False),  # was 'bush' in RUGD; broadened to shrubs/undergrowth
    ("log",           (135, 115,  90), False),
    # small vertical + climbable
    ("stairs",        (220, 140,  80), True),   # Go2W handles stairs
    ("pole",          ( 25,  65, 130), False),
    ("traffic sign",  (230, 195,  60), False),
    ("traffic light", (235,  85,  75), False),
    # dynamic (top priority -- override everything they occlude)
    ("vehicle",       (110, 130, 220), False),
    ("motorcycle",    (155,  60, 200), False),
    ("bicycle",       (100, 230, 200), False),
    ("person",        (205,  70, 145), False),
]


# ------------------------------------------------------------------
# Inlined from diffsynth/utils/auxiliary.py so this script runs from
# the lean `sam3` env (no diffsynth / modelscope needed for labeling).
# MUST stay byte-identical to the diffsynth version -- same frame sampling
# is the whole reason the SAM3 labels align 1:1 with inference.py's frames.
# ------------------------------------------------------------------
def center_crop(image, resolution):
    """Center crop a PIL Image to target resolution, scaling first to cover."""
    width, height = image.size
    target_width, target_height = resolution
    scale_final = max(target_width / width, target_height / height)
    output_width = int(width * scale_final)
    output_height = int(height * scale_final)
    scaled_image = image.resize((output_width, output_height), resample=Image.LANCZOS)
    left = (output_width - target_width) // 2
    top = (output_height - target_height) // 2
    return scaled_image.crop((left, top, left + target_width, top + target_height))


def load_video(data, num_frames, resolution=(560, 336), resize_mode="center_crop", static_scene=False):
    """Load N frames from a video (or image dir / list). Mirror of diffsynth's version."""
    def _process(image):
        if resize_mode == "resize":
            return image.resize(resolution, resample=Image.LANCZOS)
        return center_crop(image, resolution)

    assert isinstance(data, (str, list)), f"data must be str or list, got {type(data)}"
    if isinstance(data, str) and data.endswith((".jpg", ".jpeg", ".png")):
        data = [data]

    if isinstance(data, list):
        paths = sorted(data, key=lambda x: os.path.basename(x))
        idxs = np.arange(len(paths)) if static_scene else np.linspace(0, len(paths) - 1, num_frames, dtype=int)
        return [_process(Image.open(paths[i])) for i in idxs]
    if os.path.isdir(data):
        names = sorted(os.listdir(data))
        idxs = np.arange(len(names)) if static_scene else np.linspace(0, len(names) - 1, num_frames, dtype=int)
        return [_process(Image.open(os.path.join(data, names[i]))) for i in idxs]
    if os.path.isfile(data):
        vr = VideoReader(data)
        idxs = np.arange(len(vr)) if static_scene else np.linspace(0, len(vr) - 1, num_frames, dtype=int)
        raw = vr.get_batch(idxs).asnumpy()
        return [_process(Image.fromarray(f)) for f in raw]
    raise ValueError(f"Invalid data input: {data}")


def _dtype_from_str(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--resize_mode", choices=["center_crop", "resize"], default="center_crop")
    ap.add_argument("--static_scene", action="store_true")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="score threshold for keeping an instance mask")
    ap.add_argument("--mask_threshold", type=float, default=0.5,
                    help="per-pixel sigmoid threshold on the mask logits")
    ap.add_argument("--overlay_every", type=int, default=16,
                    help="save an overlay every k frames")
    ap.add_argument("--prompts", default=None,
                    help="comma-separated prompts to override the default CLASSES (auto colors)")
    ap.add_argument("--model_id", default="facebook/sam3",
                    help="HuggingFace model ID (facebook/sam3 is the community-tested default)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="model dtype -- bfloat16 is Meta's native, memory-efficient")
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

    dtype = _dtype_from_str(args.dtype)
    print(f"loading {args.model_id} in {args.dtype} ...", flush=True)
    # `dtype=` handles model creation + weight casting cleanly in one shot. No manual
    # cast-with-complex-preserve, no autocast wrestling -- HF's integration takes care
    # of all the intermediates that were tripping us on the raw sam3 package path.
    model = Sam3Model.from_pretrained(args.model_id, dtype=dtype, device_map="auto")
    processor = Sam3Processor.from_pretrained(args.model_id)
    model.eval()

    labels = np.zeros((N, H, W), dtype=np.int8)   # 0 = unlabeled, 1..C = class
    colors = np.array([(0, 0, 0)] + [c for _, c, _ in CLASSES], dtype=np.uint8)
    t0 = time.time()
    for fi, img in enumerate(frames):
        img = img.convert("RGB")
        cmap = np.zeros((H, W), dtype=np.int8)

        # === per-frame: encode vision ONCE, reuse across all class prompts ===
        img_inputs = processor(images=img, return_tensors="pt").to(model.device)
        with torch.no_grad():
            vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)
        target_sizes = img_inputs.get("original_sizes").tolist()

        # per-class inference: text encoder + DETR decoder + mask head only (cheap)
        for ci, (name, _color, _trav) in enumerate(CLASSES, start=1):
            text_inputs = processor(text=name, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(vision_embeds=vision_embeds, **text_inputs)
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=args.conf,
                mask_threshold=args.mask_threshold,
                target_sizes=target_sizes,
            )[0]
            m = results.get("masks")
            if m is None or (hasattr(m, "__len__") and len(m) == 0):
                continue
            # `masks` may come back as a stacked tensor [k, H, W] or a list of [H, W].
            if isinstance(m, list):
                m = torch.stack([mm if isinstance(mm, torch.Tensor) else torch.as_tensor(mm) for mm in m])
            # bool-any collapses the instance dimension: pixel is "this class" if ANY
            # instance covers it. Then priority overwrite in the outer loop order.
            class_mask = m.any(dim=0).cpu().numpy().astype(bool)
            cmap[class_mask] = ci

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
