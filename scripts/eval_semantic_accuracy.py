"""Pixel accuracy + per-class IoU of a rendered semantic_labels.npz vs GT labels.

Usage:
  python scripts/eval_semantic_accuracy.py <pred_labels.npz> <gt_labels.npz> [--label name]

Both npz carry `labels` [T, H, W] int class ids in the v14 taxonomy.
GT void pixels (class 0) are excluded from scoring, matching the held-out
evaluation convention (v9: trail-6 78.8% / park-1 69.9%).
"""
import argparse

import numpy as np

V14_NAMES = ["void", "sky", "trail", "grass", "rough", "water", "sidewalk",
             "road", "pavement-unknown", "stairs", "obstacle", "vegetation",
             "person", "vehicle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_npz")
    ap.add_argument("gt_npz")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    pred = np.load(args.pred_npz)["labels"]
    gt = np.load(args.gt_npz)["labels"]
    t = min(len(pred), len(gt))
    pred, gt = pred[:t], gt[:t]
    assert pred.shape == gt.shape, f"shape mismatch: pred {pred.shape} vs gt {gt.shape}"

    valid = gt != 0
    acc = float((pred[valid] == gt[valid]).mean())
    tag = args.label or args.pred_npz
    print(f"{tag}: {t} frames, pixel accuracy (non-void GT): {acc * 100:.1f}%")

    for c in range(1, len(V14_NAMES)):
        gt_c, pr_c = gt == c, pred == c
        union = (gt_c | pr_c).sum()
        if gt_c.sum() == 0:
            continue
        iou = (gt_c & pr_c).sum() / union * 100
        share = gt_c.sum() / valid.sum() * 100
        print(f"  {V14_NAMES[c]:>17}: IoU {iou:5.1f}  (GT share {share:4.1f}%)")


if __name__ == "__main__":
    main()
