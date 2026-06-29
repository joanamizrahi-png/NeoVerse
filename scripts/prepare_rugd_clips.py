"""Convert RUGD image sequences into MP4 clips for the SAM3-labeling pipeline.

RUGD ships as per-scene PNG frame sequences (creek_00001.png, creek_00002.png, ...).
NeoVerse's load_video() wants MP4. This script:
  1. Discovers RUGD scenes (either each scene in its own subfolder, or all frames
     flat with a "<scene>_<frame>.png" naming convention).
  2. For each scene, sorts frames numerically and groups them into 81-frame chunks
     spaced by --stride frames (to get visually diverse clips, not overlapping).
  3. ffmpegs each chunk into an MP4 written to examples/videos/training_pilot/.

Usage on Jetson:
    python scripts/prepare_rugd_clips.py --rugd_root ~/joana/rugd
    # optional flags:
    #   --stride 250         (frames between consecutive chunk starts; bigger = more diversity, fewer clips)
    #   --max_per_scene 6    (cap to avoid an over-represented scene)
    #   --fps 8              (output framerate)
    #   --out_dir examples/videos/training_pilot
"""
import argparse, os, re, subprocess, sys, shutil
from collections import defaultdict
from pathlib import Path


FRAME_RE = re.compile(r"^(?P<scene>.+?)[_-](?P<idx>\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


def discover_scenes(rugd_root: Path):
    """Return a dict: scene_name -> sorted list of frame paths.

    Handles two layouts:
      a) rugd_root/<scene>/<scene>_<idx>.png   (one subdir per scene)
      b) rugd_root/<scene>_<idx>.png            (all frames flat)
    """
    scenes = defaultdict(list)

    # Layout (a): scene subdirs
    for sub in sorted(rugd_root.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.iterdir()):
            m = FRAME_RE.match(f.name)
            if m:
                scenes[m["scene"]].append(f)

    # Layout (b): flat files in rugd_root
    if not scenes:
        for f in sorted(rugd_root.iterdir()):
            if not f.is_file():
                continue
            m = FRAME_RE.match(f.name)
            if m:
                scenes[m["scene"]].append(f)

    # Sort frames within each scene by numeric index
    for scene, files in scenes.items():
        files.sort(key=lambda p: int(FRAME_RE.match(p.name)["idx"]))

    return scenes


def chunk_to_mp4(frames, out_path: Path, fps: int):
    """ffmpeg a list of frame paths into one MP4."""
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found in PATH. Install with: sudo apt install ffmpeg")
        sys.exit(1)
    # Build a concat list (ffmpeg's "image2 pipe" expects a glob or sequential numbering;
    # using -safe 0 + concat protocol with absolute paths is the most flexible).
    listfile = out_path.with_suffix(".txt")
    with listfile.open("w") as f:
        # ffmpeg concat with file paths
        for fp in frames:
            f.write(f"file '{fp.resolve()}'\n")
            f.write(f"duration {1.0 / fps}\n")
        # repeat the last frame (concat demuxer quirk)
        f.write(f"file '{frames[-1].resolve()}'\n")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(listfile),
        "-vsync", "vfr", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "23",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    listfile.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rugd_root", required=True, help="Path to RUGD root (with scenes or flat frames)")
    ap.add_argument("--out_dir", default="examples/videos/training_pilot")
    ap.add_argument("--frames_per_clip", type=int, default=81)
    ap.add_argument("--stride", type=int, default=250,
                    help="Frames between consecutive clip starts within a scene")
    ap.add_argument("--max_per_scene", type=int, default=6,
                    help="Cap number of clips taken from any single scene")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    rugd_root = Path(args.rugd_root).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rugd_root.exists():
        print(f"RUGD root not found: {rugd_root}")
        sys.exit(1)

    scenes = discover_scenes(rugd_root)
    if not scenes:
        print(f"No frame files found under {rugd_root}.")
        print("Expected PNG/JPG files named like 'creek_00001.png' (either in subfolders per scene or flat).")
        sys.exit(1)

    print(f"Found {len(scenes)} scenes:")
    for s, fr in scenes.items():
        print(f"  {s}: {len(fr)} frames")

    total_clips = 0
    for scene, frames in scenes.items():
        n = len(frames)
        if n < args.frames_per_clip:
            print(f"SKIP {scene}: only {n} frames (< {args.frames_per_clip})")
            continue
        clip_idx = 0
        start = 0
        while start + args.frames_per_clip <= n and clip_idx < args.max_per_scene:
            chunk = frames[start:start + args.frames_per_clip]
            out_path = out_dir / f"rugd_{scene}_{clip_idx:02d}.mp4"
            if out_path.exists():
                print(f"SKIP {out_path.name} (exists)")
            else:
                print(f"WRITE {out_path.name}  (frames {start}..{start + args.frames_per_clip - 1})")
                chunk_to_mp4(chunk, out_path, args.fps)
                total_clips += 1
            clip_idx += 1
            start += args.stride

    print(f"\nDone. {total_clips} new clips in {out_dir}/")


if __name__ == "__main__":
    main()
