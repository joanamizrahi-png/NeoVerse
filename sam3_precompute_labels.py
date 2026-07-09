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
from decord import VideoReader
import sam3.model_builder as _sam3_mb
from sam3.model.necks import Sam3DualViTDetNeck
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ------------------------------------------------------------------
# SAM 3.1 architecture patch (facebook/sam3.1 config.json shows 3 feature
# levels; Meta's build_sam3_image_model hardcodes 4). Without this patch,
# loading sam3.1_multiplex.pt gives missing_keys=['...convs.3.*'] and a
# dtype mismatch at inference. Enabled when the checkpoint path looks
# SAM 3.1-ish (contains "3.1" or "multiplex"), otherwise no-op so SAM 3
# checkpoints keep working unchanged.
# ------------------------------------------------------------------
_ORIG_CREATE_VIT_NECK = _sam3_mb._create_vit_neck


def _create_vit_neck_sam31(position_encoding, vit_backbone, enable_inst_interactivity=False):
    """SAM 3.1 neck: 3 scales instead of 4 (drops the 0.5 downsample)."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _maybe_patch_for_sam31(checkpoint_path: str) -> bool:
    """Monkey-patch model_builder for SAM 3.1 if the checkpoint looks like one."""
    tag = os.path.basename(checkpoint_path).lower()
    if "3.1" in tag or "3p1" in tag or "multiplex" in tag:
        _sam3_mb._create_vit_neck = _create_vit_neck_sam31
        return True
    _sam3_mb._create_vit_neck = _ORIG_CREATE_VIT_NECK  # restore, in case
    return False


# ------------------------------------------------------------------
# Inlined from diffsynth/utils/auxiliary.py so this script runs from
# the lean `sam3` env (no diffsynth / modelscope needed for labeling).
# MUST stay byte-identical to the diffsynth version — same frame sampling
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

# --- EDIT HERE: prompt, RGB color, traversable. Order = priority (later overwrites earlier). ---
# 29-class outdoor Go2W taxonomy (class ids 1..29; void=0 handled as "unlabeled").
# Colors MUST stay identical to diffsynth/utils/semantics.py CLASS_COLORS[1:] and TRAVERSABLE
# flags MUST match nav-rl/src/env/reward.py TRAVERSABLE. Ordering encodes SAM3 priority (later
# wins) AND the class-id space — all three files must be updated in lockstep.
#
# Priority layering (bottom → top):
#   sky → ground materials → ground hazards → pavement materials → pavement functions →
#   large statics → vegetation → small verticals + stairs → dynamic objects.
# Dynamics come last so a person/vehicle on a road stays labeled person/vehicle.
CLASSES = [
    # ambient
    ("sky",           (200, 225, 245), False),
    # ground materials (default ground layer)
    ("dirt",          (139,  90,  43), True),   # FLAG: revisit — may not be Go2W-traversable when loose
    ("sand",          (230, 200, 155), True),   # FLAG: revisit — loose sand may not be Go2W-traversable
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
    # dynamic (top priority — override everything they occlude)
    ("vehicle",       (110, 130, 220), False),
    ("motorcycle",    (155,  60, 200), False),
    ("bicycle",       (100, 230, 200), False),
    ("person",        (205,  70, 145), False),
]
# --- SAM 3.1 checkpoint on Marlowe (change --checkpoint if you move it or want SAM 3). ---
# BPE=None lets build_sam3_image_model auto-resolve the tokenizer via pkg_resources
# (points at the copy inside the sam3 package). Pass --bpe explicitly to override.
BPE = None
CKPT_31 = "/scratch/m000204-pm06b/joana/sam3_ckpts/3.1/sam3.1_multiplex.pt"


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
    ap.add_argument("--checkpoint", default=CKPT_31,
                    help="path to SAM 3 or 3.1 checkpoint .pt (default: SAM 3.1 on Marlowe)")
    ap.add_argument("--bpe", default=BPE,
                    help="path to BPE tokenizer vocab (default: None -> auto-resolve via sam3 package)")
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

    is_sam31 = _maybe_patch_for_sam31(args.checkpoint)
    print(f"building SAM{'3.1' if is_sam31 else '3'} from {args.checkpoint} ...", flush=True)
    # NOTE: build_sam3_image_model defaults to SAM 3 (facebook/sam3, "sam3.pt") when
    # load_from_HF=True and checkpoint_path=None. To use SAM 3.1, pass the checkpoint
    # path explicitly AND set load_from_HF=False so the auto-download doesn't override.
    # DO NOT cast the model to bfloat16 — the ViT backbone uses RoPE (complex numbers)
    # which get silently discarded on bf16 conversion. Let sam3's own autocast handle
    # activation dtype during forward.
    model = build_sam3_image_model(
        bpe_path=args.bpe,
        device="cuda",
        checkpoint_path=args.checkpoint,
        load_from_HF=False,
    )
    proc = Sam3Processor(model, device="cuda", confidence_threshold=args.conf)

    labels = np.zeros((N, H, W), dtype=np.int8)   # 0 = unlabeled, 1..C = class
    colors = np.array([(0, 0, 0)] + [c for _, c, _ in CLASSES], dtype=np.uint8)
    t0 = time.time()
    # sam3's ViT MLPs use a fused addmm_act kernel that always emits bfloat16, but
    # the surrounding LayerNorm / fc2 keep float32 weights -> dtype mismatch on plain
    # forward. Wrap inference in autocast(bfloat16) so Linear/Conv weights are cast
    # transparently, while RoPE's complex64 buffers stay complex (they only appear
    # inside operators autocast leaves alone).
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    for fi, img in enumerate(frames):
        img = img.convert("RGB")
        cmap = np.zeros((H, W), dtype=np.int8)
        with autocast_ctx:
            state = proc.set_image(img)
            for ci, (name, _color, _trav) in enumerate(CLASSES, start=1):
                # Reset prior prompt so state["masks"] reflects ONLY the current class.
                # Matches the pattern in sam3/examples/sam3_image_predictor_example.ipynb.
                proc.reset_all_prompts(state)
                state = proc.set_text_prompt(state=state, prompt=name)
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
