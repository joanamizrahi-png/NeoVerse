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
        # Where this GT class's pixels actually went (top-3 predicted classes):
        # an IoU of 0.0 alone can't distinguish "predicted the near-synonym"
        # (sidewalk->pavement) from "predicted garbage".
        u, cnt = np.unique(pred[gt_c], return_counts=True)
        top = np.argsort(-cnt)[:3]
        went = ", ".join(
            f"{V14_NAMES[u[i]] if u[i] < len(V14_NAMES) else u[i]} "
            f"{100 * cnt[i] / cnt.sum():.0f}%" for i in top)
        print(f"  {V14_NAMES[c]:>17}: IoU {iou:5.1f}  (GT share {share:4.1f}%)"
              f"  -> painted as: {went}")

    # One-line reward-relevance summary: accuracy when traversability-
    # equivalent pairs count as matches (sidewalk<->pavement, trail<->grass,
    # obstacle<->vegetation). The 14-class table above stays the main result.
    lut = np.arange(32, dtype=np.int8)
    for keep, other in ((6, 8), (2, 3), (10, 11)):
        lut[other] = keep
    macc = float((lut[pred][valid] == lut[gt][valid]).mean())
    print(f"  traversability-merged accuracy: {macc * 100:.1f}%")


if __name__ == "__main__":
    main()
