"""Colorize a label npz (SAM3 or RUGD-GT, [T,H,W] class ids) into an MP4 using
the canonical CLASS_COLORS palette. Runs anywhere with torch+numpy (Mac OK).

Usage: python scripts/labels_npz_to_mp4.py in.npz out.mp4
"""
import sys
from pathlib import Path
import numpy as np
import imageio.v3 as iio

# Load semantics.py directly by file path — importing the diffsynth package
# pulls heavy deps (huggingface_hub etc.) that dev machines may lack.
import importlib.util
_sem_path = Path(__file__).resolve().parents[1] / "diffsynth/utils/semantics.py"
_spec = importlib.util.spec_from_file_location("semantics_standalone", _sem_path)
_sem = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sem)
CLASS_COLORS, NUM_CLASSES = _sem.CLASS_COLORS, _sem.NUM_CLASSES

def main():
    in_npz, out_mp4 = sys.argv[1], sys.argv[2]
    labels = np.load(in_npz)["labels"]
    palette = (CLASS_COLORS.numpy() * 255).clip(0, 255).astype(np.uint8)
    frames = palette[np.clip(labels, 0, NUM_CLASSES - 1)]      # [T,H,W,3]
    iio.imwrite(out_mp4, frames, fps=16, codec="libx264",
                macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"wrote {out_mp4} ({frames.shape})")

if __name__ == "__main__":
    main()
