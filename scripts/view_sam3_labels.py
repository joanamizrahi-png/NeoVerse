"""Colorize a SAM3-labels .npz into an MP4 (or grid of PNGs) for eyeballing.

The .npz is what the diffusion sees as the CLEAN semantic training target under
Option A -- the exact tensor that gets colorized, VAE-encoded, and channel-
concatenated with the RGB latent. Watching it as a video lets you sanity-check:
  * palette colors look right (no unexpected classes)
  * label boundaries follow object contours (SAM3 didn't hallucinate)
  * temporal consistency isn't wildly jittery (SAM3 handled the same object
    across frames)

Usage:
  # colorize + write side-by-side RGB / semantic MP4 to outputs/sam3_labels/<stem>_view.mp4
  python scripts/view_sam3_labels.py --npz outputs/sam3_labels/driving.npz \
      --video examples/videos/driving.mp4

  # colorize only (no source video needed) -> just the semantic mp4
  python scripts/view_sam3_labels.py --npz outputs/sam3_labels/driving.npz

  # dump every Nth frame as a PNG grid (for scp + eyeballing on Mac)
  python scripts/view_sam3_labels.py --npz outputs/sam3_labels/driving.npz --pngs 8
"""
import argparse, os
import numpy as np
from PIL import Image


def colorize(labels: np.ndarray, class_colors: np.ndarray) -> np.ndarray:
    """[N,H,W] int -> [N,H,W,3] uint8 using the class_colors LUT."""
    N, H, W = labels.shape
    K = class_colors.shape[0]
    idx = labels.astype(np.int64).clip(0, K - 1)
    return class_colors[idx]                                 # [N,H,W,3] uint8


def load_source_frames(video_path: str, num_frames: int, hw: tuple) -> np.ndarray:
    """Load N frames from a video, resized to (H, W). Returns [N, H, W, 3] uint8."""
    from decord import VideoReader
    vr = VideoReader(video_path)
    idxs = np.linspace(0, len(vr) - 1, num_frames, dtype=int) if len(vr) != num_frames else np.arange(num_frames)
    raw = vr.get_batch(idxs).asnumpy()                       # [N,H0,W0,3] uint8
    if raw.shape[1:3] == hw:
        return raw
    # simple resize with PIL if source resolution differs from labels
    out = np.zeros((raw.shape[0], hw[0], hw[1], 3), dtype=np.uint8)
    for i, f in enumerate(raw):
        out[i] = np.array(Image.fromarray(f).resize((hw[1], hw[0]), Image.BILINEAR))
    return out


def write_mp4(frames: np.ndarray, path: str, fps: int = 8) -> None:
    """[N,H,W,3] uint8 -> MP4 via imageio-ffmpeg (already in the sam3 env)."""
    try:
        import imageio.v3 as iio
        iio.imwrite(path, frames, fps=fps, codec="libx264",
                    macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    except Exception:
        # fallback: plain imageio v2
        import imageio
        writer = imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1)
        for f in frames:
            writer.append_data(f)
        writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="path to sam3_labels/<stem>.npz")
    ap.add_argument("--video", default=None,
                    help="optional: source MP4 to render side-by-side with the semantics")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--pngs", type=int, default=0,
                    help="if > 0, dump every Nth colorized frame as a PNG to <stem>_pngs/")
    ap.add_argument("--overlay_alpha", type=float, default=0.55,
                    help="if --video is set, blend semantic on top of RGB with this alpha "
                         "(0 = only RGB, 1 = only semantic). Ignored otherwise.")
    ap.add_argument("--layout", choices=["overlay", "sidebyside"], default="sidebyside",
                    help="how to compose semantic with video when --video is set")
    args = ap.parse_args()

    data = np.load(args.npz)
    labels = data["labels"]                # [N,H,W] int8
    class_colors = data["class_colors"]    # [K,3] uint8 (includes void as row 0)
    class_names = data["class_names"]      # [K] str
    N, H, W = labels.shape
    print(f"loaded {args.npz}: {N} frames, {H}x{W}, {class_colors.shape[0]} classes")

    # class-frequency spot check -- helps flag if a class never fires
    for cid in range(class_colors.shape[0]):
        pct = (labels == cid).mean() * 100
        if pct > 0.5:
            print(f"  class {cid:2d} {class_names[cid]:15s}: {pct:5.1f}%")

    sem_rgb = colorize(labels, class_colors)   # [N,H,W,3] uint8
    stem = os.path.splitext(os.path.basename(args.npz))[0]
    out_dir = os.path.dirname(args.npz) or "."

    if args.pngs > 0:
        png_dir = os.path.join(out_dir, f"{stem}_pngs")
        os.makedirs(png_dir, exist_ok=True)
        for i in range(0, N, args.pngs):
            Image.fromarray(sem_rgb[i]).save(os.path.join(png_dir, f"sem_{i:03d}.png"))
        print(f"wrote {N // args.pngs + 1} pngs to {png_dir}/")

    if args.video:
        src = load_source_frames(args.video, N, (H, W))
        if args.layout == "sidebyside":
            composed = np.concatenate([src, sem_rgb], axis=2)     # [N, H, 2W, 3]
        else:
            a = float(args.overlay_alpha)
            composed = ((1 - a) * src.astype(np.float32) + a * sem_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(out_dir, f"{stem}_view.mp4")
        write_mp4(composed, out_path, args.fps)
        print(f"wrote {out_path}  (RGB | semantic {args.layout}, {args.fps} fps)")
    else:
        out_path = os.path.join(out_dir, f"{stem}_semantic.mp4")
        write_mp4(sem_rgb, out_path, args.fps)
        print(f"wrote {out_path}  (semantic only, {args.fps} fps)")


if __name__ == "__main__":
    main()
