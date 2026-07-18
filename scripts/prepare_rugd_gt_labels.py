"""Build dense ground-truth label npz files for RUGD training clips (v7, Option B).

Why: v3-v6 trained with SAM3 pseudo-labels as the CLEAN target (Option A). SAM3
targets are themselves speckled and full of void, so the diffusion faithfully
learned speckle — that's one of the three speckle sources in the v6 output.
RUGD ships dense human-annotated masks: flat regions, no pepper. Training on
them teaches the model that semantic maps are FLAT.

Pairing is deterministic because prepare_rugd_clips.py made each clip from a
contiguous frame chunk: clip `rugd_<scene>_<k>.mp4` = sorted-frames[k*stride :
k*stride + 81]. We re-discover the annotation frames with the SAME discovery +
sort logic, take the SAME chunk, then:

  1. color-coded RUGD mask -> RUGD class index (palette parsed from the official
     RUGD_annotation-colormap.txt — never hardcoded)
  2. RUGD class name -> our 30-class Go2W taxonomy id (mapping below)
  3. resize/center-crop with NEAREST interpolation through the same geometry as
     load_video's center_crop (so pixel (i,j) in the label = pixel (i,j) in the
     training frame)
  4. save outputs/<out_dir>/rugd_<scene>_<k>.npz  {"labels": [81, H, W] int8}

The npz format matches the SAM3 label npz exactly, so the dataloader indexes
both with the same sampled frame indices.

IMPORTANT: run with the same --stride / --max_per_scene / --frames_per_clip as
the original prepare_rugd_clips.py run (defaults match its defaults). If the
chunking parameters differ, labels will pair with the wrong frames.

Usage (Marlowe login node is fine — CPU only, ~minutes):
    python scripts/prepare_rugd_gt_labels.py \
        --annotations_root /scratch/m000204-pm06b/joana/data/rugd/RUGD_annotations \
        --colormap /scratch/m000204-pm06b/joana/data/rugd/RUGD_annotations/RUGD_annotation-colormap.txt \
        --clips_dir /scratch/m000204-pm06b/joana/data/rugd_clips \
        --out_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/rugd_gt_labels
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

FRAME_RE = re.compile(r"^(?P<scene>.+?)[_-](?P<idx>\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)

# RUGD class name -> our 30-class taxonomy id (diffsynth/utils/semantics.CLASS_COLORS
# order — LOCKSTEP with sam3_precompute_labels.CLASSES and nav-rl TRAVERSABLE).
# Names normalized to lowercase with spaces; RUGD's "rock-bed" etc. handled below.
RUGD_TO_OURS = {
    "void": 0, "dirt": 2, "sand": 3, "grass": 4, "tree": 19, "pole": 23,
    "water": 8, "sky": 1, "vehicle": 26, "asphalt": 10, "gravel": 5,
    "building": 15, "mulch": 6, "rock-bed": 9, "rock bed": 9, "log": 21,
    "bicycle": 28, "person": 29, "fence": 17, "bush": 20, "sign": 24,
    "rock": 9, "bridge": 18, "concrete": 11, "picnic-table": 0, "picnic table": 0,
    "container/generic-object": 0, "container": 0, "generic-object": 0,
}
# picnic-table / container have no slot in our taxonomy -> void (tiny pixel share;
# revisit if they matter for traversability).


def parse_colormap(path: Path):
    """RUGD_annotation-colormap.txt lines: '<idx> <name> <R> <G> <B>'.
    Returns (colors [K,3] uint8, names list[str])."""
    colors, names = [], []
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        name = " ".join(parts[1:-3]).lower()
        r, g, b = (int(v) for v in parts[-3:])
        names.append(name)
        colors.append((r, g, b))
    if not colors:
        sys.exit(f"could not parse colormap at {path}")
    return np.array(colors, dtype=np.int16), names


def color_mask_to_ours(mask_rgb: np.ndarray, colors: np.ndarray, remap: np.ndarray) -> np.ndarray:
    """[H,W,3] color mask -> [H,W] our-class ids via exact palette match
    (nearest color as fallback for jpeg-ish edge artifacts)."""
    h, w = mask_rgb.shape[:2]
    flat = mask_rgb.reshape(-1, 3).astype(np.int16)
    d = np.abs(flat[:, None, :] - colors[None, :, :]).sum(-1)   # [N, K] L1
    rugd_idx = d.argmin(1)
    return remap[rugd_idx].reshape(h, w)


def center_crop_nearest(label: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Bit-exact geometry match to diffsynth auxiliary.center_crop (which uses
    int() truncation, not round), but NEAREST interpolation for labels."""
    h, w = label.shape
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    im = Image.fromarray(label.astype(np.uint8)).resize((new_w, new_h), Image.NEAREST)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return np.array(im.crop((left, top, left + target_w, top + target_h)), dtype=np.int8)


def discover_scenes(root: Path):
    scenes = defaultdict(list)
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.iterdir()):
            m = FRAME_RE.match(f.name)
            if m:
                scenes[m["scene"]].append(f)
    if not scenes:
        for f in sorted(root.iterdir()):
            if f.is_file() and (m := FRAME_RE.match(f.name)):
                scenes[m["scene"]].append(f)
    for scene, files in scenes.items():
        files.sort(key=lambda p: int(FRAME_RE.match(p.name)["idx"]))
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations_root", required=True, type=Path)
    ap.add_argument("--colormap", required=True, type=Path)
    ap.add_argument("--clips_dir", required=True, type=Path,
                    help="dir with rugd_<scene>_<k>.mp4 — determines which chunks to label")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--frames_per_clip", type=int, default=81)
    ap.add_argument("--stride", type=int, default=250)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=336)
    args = ap.parse_args()

    colors, names = parse_colormap(args.colormap)
    print(f"[gt_labels] colormap: {len(names)} classes: {names}")
    remap = np.zeros(len(names), dtype=np.int8)
    unmapped = []
    for i, n in enumerate(names):
        if n in RUGD_TO_OURS:
            remap[i] = RUGD_TO_OURS[n]
        else:
            unmapped.append(n)
            remap[i] = 0
    if unmapped:
        print(f"[gt_labels] WARNING: unmapped RUGD classes -> void: {unmapped}")

    scenes = discover_scenes(args.annotations_root)
    print(f"[gt_labels] {len(scenes)} annotation scenes found")

    clip_re = re.compile(r"^rugd_(?P<scene>.+)_(?P<k>\d+)\.mp4$")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for clip in sorted(args.clips_dir.glob("rugd_*.mp4")):
        m = clip_re.match(clip.name)
        if not m:
            continue
        scene, k = m["scene"], int(m["k"])
        frames = scenes.get(scene)
        if not frames:
            print(f"[gt_labels] SKIP {clip.stem}: no annotation scene '{scene}'")
            skipped += 1
            continue
        start = k * args.stride
        chunk = frames[start:start + args.frames_per_clip]
        if len(chunk) < args.frames_per_clip:
            print(f"[gt_labels] SKIP {clip.stem}: chunk {start}.. exceeds "
                  f"{len(frames)} annotated frames")
            skipped += 1
            continue
        labels = np.stack([
            center_crop_nearest(
                color_mask_to_ours(np.array(Image.open(f).convert("RGB")), colors, remap),
                args.width, args.height)
            for f in chunk
        ], axis=0)   # [81, H, W] int8

        out = args.out_dir / f"{clip.stem}.npz"
        np.savez_compressed(out, labels=labels)
        frac_void = float((labels == 0).mean())
        print(f"[gt_labels] {clip.stem}: wrote {labels.shape}, void {frac_void:.1%}")
        done += 1

    print(f"[gt_labels] done: {done} written, {skipped} skipped")
    if done:
        print(f"[gt_labels] point train config target_labels_dir at: {args.out_dir}")


if __name__ == "__main__":
    main()
