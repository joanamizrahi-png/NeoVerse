"""Regenerate all label npz files in the 14-class taxonomy (v14).

Reads every npz in the given dirs, remaps ids BY NAME using each file's own
embedded class_names (immune to the legacy id-ordering mismatch — each file
declares what its ids meant), writes *_v14 siblings with v14 ids + metadata.
CPU-only, safe on the login node (~seconds per file).

Usage (Marlowe):
  python scripts/remap_labels_to_v14.py \
      --dirs outputs/sam3_labels outputs/rugd_gt_labels
Output: outputs/sam3_labels_v14/, outputs/rugd_gt_labels_v14/
"""
from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import numpy as np

# Load the taxonomy module directly by path: importing via the diffsynth
# package would pull heavy deps (torch/modelscope) that plain login-node
# python doesn't have — this script is numpy-only on purpose.
import importlib.util
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "class_taxonomy", os.path.join(_root, "diffsynth/utils/class_taxonomy.py"))
_tax = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tax)
V14_NAMES, V14 = _tax.V14_NAMES, _tax.V14
NUM_CLASSES_V14, remap_array_from_names = _tax.NUM_CLASSES_V14, _tax.remap_array_from_names

# GT files (prepare_rugd_gt_labels.py) may lack class_names metadata; they are
# known to be in the CLASS_COLORS ordering — declared here once, by name.
CLASS_COLORS_ORDER = [
    "void", "sky", "dirt", "sand", "grass", "gravel", "mulch", "mud", "water",
    "rock", "asphalt", "concrete", "road", "sidewalk", "crosswalk", "building",
    "wall", "fence", "bridge", "tree", "vegetation", "log", "stairs", "pole",
    "traffic sign", "traffic light", "vehicle", "motorcycle", "bicycle", "person",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    args = ap.parse_args()

    colors = np.array([c for _, c, _ in V14], dtype=np.uint8)
    for d in args.dirs:
        out_dir = d.rstrip("/") + "_v14"
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(glob(os.path.join(d, "*.npz")))
        print(f"== {d}: {len(files)} files -> {out_dir}")
        for f in files:
            data = np.load(f, allow_pickle=True)
            labels = data["labels"]
            if "class_names" in data.files:
                lut = remap_array_from_names(list(data["class_names"]))
                src = "embedded names"
            else:
                lut = remap_array_from_names(CLASS_COLORS_ORDER)
                src = "CLASS_COLORS order (no metadata)"
            new = lut[labels.astype(np.int32)].astype(np.int8)
            out = os.path.join(out_dir, os.path.basename(f))
            np.savez_compressed(
                out, labels=new,
                class_names=np.array(V14_NAMES),
                class_colors=colors,
                num_classes=NUM_CLASSES_V14,
            )
            hist = np.bincount(new.flatten(), minlength=NUM_CLASSES_V14)
            top = np.argsort(hist)[::-1][:3]
            print(f"  {os.path.basename(f):34s} [{src}] top: "
                  + ", ".join(f"{V14_NAMES[t]} {hist[t]/new.size:.0%}" for t in top))
    print("done.")


if __name__ == "__main__":
    main()
