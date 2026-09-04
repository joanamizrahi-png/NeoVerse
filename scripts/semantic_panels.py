"""Side-by-side semantic panels for one clip: RGB | human GT | model A | model B ...

Every panel is colorized from CLASS IDS with the same palette (v14, version 4),
GT included, so colours mean the same thing in every panel and mp4 compression
never enters the comparison. Each model panel's title carries its pixel
accuracy against the GT (void excluded), so the eye and the number sit together.

    python scripts/semantic_panels.py \
        --gt /scratch/.../data/sanpo_val/labels/sanpoval_TTj3piLyS3_000656.npz \
        --panels v26_e10=/scratch/.../inference_train_semantic_v26_campus_sanpoval_..._p4 \
                 v26b_e10=/scratch/.../inference_train_semantic_v26b_campus_seg_sanpoval_..._p4 \
        --out /scratch/.../outputs/sem_panels_TTj3.mp4

Each panel dir must hold semantic_labels.npz (ids [T,H,W]); the first one
also supplies rgb.mp4 for the leftmost panel. Runs on a login node (CPU).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT labels npz (ids [T,H,W])")
    ap.add_argument("--panels", nargs="+", required=True, help="name=dir, in display order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.5, help="per-panel scale of 560x336")
    ap.add_argument("--palette", type=int, default=4)
    args = ap.parse_args()

    import cv2
    # Load the palette module by file path: importing `diffsynth` pulls in the
    # whole package (transformers included), which fails on a login node with
    # polluted user site-packages. class_taxonomy.py only needs numpy.
    import importlib.util
    tax_path = Path(__file__).resolve().parents[1] / "diffsynth" / "utils" / "class_taxonomy.py"
    spec = importlib.util.spec_from_file_location("class_taxonomy", tax_path)
    tax = importlib.util.module_from_spec(spec); spec.loader.exec_module(tax)
    # the raw colour tables, no torch: v14_palette() would import it
    if int(args.palette) == 4:
        cols = tax.V14_V4
    elif int(args.palette) == 3:
        cols = tax.V14_V3
    else:
        cols = [c for _, c, _ in tax.V14]
        if int(args.palette) == 2:
            cols = list(cols)
            for i, c in tax.V14_V2_OVERRIDES.items():
                cols[i] = c
    pal = np.asarray(cols, dtype=np.float32) / 255.0      # [14, 3] in [0, 1]

    gt = np.load(args.gt)["labels"].astype(np.int64)
    names, dirs, preds = [], [], []
    for spec in args.panels:
        name, d = spec.split("=", 1)
        lab = np.load(Path(d) / "semantic_labels.npz")["labels"].astype(np.int64)
        assert lab.shape == gt.shape, f"{name}: {lab.shape} vs GT {gt.shape}"
        names.append(name); dirs.append(Path(d)); preds.append(lab)

    def colour(ids):
        return (pal[np.clip(ids, 0, len(pal) - 1)] * 255).astype(np.uint8)

    valid = gt != 0
    accs = [float((p[valid] == gt[valid]).mean()) for p in preds]

    T, H, W = gt.shape
    h, w = int(H * args.scale) // 2 * 2, int(W * args.scale) // 2 * 2
    rgb = cv2.VideoCapture(str(dirs[0] / "rgb.mp4"))
    titles = ["RGB (" + names[0] + ")", "human GT"] + [f"{n}  acc {100 * a:.1f}%" for n, a in zip(names, accs)]
    cols = len(titles)
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w * cols, h + 22))
    for t in range(T):
        ok, fr = rgb.read()
        if not ok:
            fr = np.zeros((H, W, 3), np.uint8)
        tiles = [cv2.resize(fr, (w, h)), cv2.resize(colour(gt[t])[:, :, ::-1], (w, h), interpolation=cv2.INTER_NEAREST)]
        tiles += [cv2.resize(colour(p[t])[:, :, ::-1], (w, h), interpolation=cv2.INTER_NEAREST) for p in preds]
        row = np.concatenate(tiles, axis=1)
        bar = np.full((22, w * cols, 3), 30, np.uint8)
        for i, ttl in enumerate(titles):
            cv2.putText(bar, ttl, (i * w + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        out.write(np.concatenate([bar, row], axis=0))
    out.release()
    # OpenCV's mp4v is not QuickTime-playable (green frame). Re-encode as
    # H.264 yuv420p when ffmpeg is on the path; otherwise keep the mp4v file.
    import shutil, subprocess, os
    if shutil.which("ffmpeg"):
        tmp = args.out + ".mp4v.mp4"
        os.replace(args.out, tmp)
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "18", args.out])
        if r.returncode == 0:
            os.remove(tmp)
        else:
            os.replace(tmp, args.out)
            print("    (ffmpeg re-encode failed; kept the mp4v file)")
    print("==> " + args.out + "   " + "  ".join(f"{n} {100 * a:.1f}%" for n, a in zip(names, accs)))


if __name__ == "__main__":
    main()
