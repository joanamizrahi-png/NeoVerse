"""How good is SAM3, per class, against real human ground truth?

SAM3 is the hint for the whole world model AND the label source for the scene
clouds the proximity cost reads — but its per-class accuracy has never been
measured, because RUGD's GT has almost no sidewalk/road pixels and the SANPO
converter feeds SANPO's own GT in as the hint. The held-out SANPO val set
(convert_sanpo_val.py) has dense human labels and is disjoint from training,
so running SAM3 over it and comparing is the missing measurement.

Two questions it answers, both live decisions (2026-09-01):
  * P(SAM3 = road | GT = sidewalk) — Joana: "the problem with sam3 is that it
    might label sidewalk as road", which is why road sits at 0.5 in the
    walkway table instead of lower. This sets that number instead of guessing.
  * P(SAM3 = grass | GT = sidewalk) and the reverse — whether the terrain
    boundary in the fused cloud is crisp enough to hang a ground-level
    proximity cost on.

GT void (class 0) means UNANNOTATED in SANPO, so those pixels are excluded
from every statistic rather than counted as a class.

Usage (login node — pure numpy, no GPU):
    python scripts/sam3_vs_gt.py \
        --pred_dir outputs/sam3_labels \
        --gt_dir /scratch/m000204-pm06b/joana/data/sanpo_val/labels
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

V14 = ["void", "sky", "trail", "grass", "rough", "water", "sidewalk", "road",
       "pavement", "stairs", "obstacle", "vegetation", "person", "vehicle"]
K = len(V14)


def load(p: Path) -> np.ndarray:
    d = np.load(p)
    key = "labels" if "labels" in d else list(d.keys())[0]
    return np.asarray(d[key]).astype(np.int16)


def assert_v14(p: Path) -> None:
    """Refuse to score raw SAM3 prompt indices against v14 ground truth.

    sam3_precompute_labels.py writes RAW indices to outputs/sam3_labels;
    remap_labels_to_v14.py converts them and stamps class_names/num_classes.
    Comparing the raw directory produces a confusion matrix that looks like a
    catastrophically bad segmenter — sidewalk -> vehicle 86%, person -> void
    97%, and sky correct because it happens to share an index. It cost us a
    real scare on 2026-09-01, so this is now a hard stop rather than a number
    somebody has to be suspicious of.
    """
    d = np.load(p, allow_pickle=True)
    names = [str(x) for x in d["class_names"]] if "class_names" in d.files else None
    if names != V14:
        raise SystemExit(
            f"{p} is NOT in the v14 label space "
            f"(class_names={'absent' if names is None else names[:4]}...).\n"
            f"These are raw SAM3 prompt indices. Remap first:\n"
            f"    python scripts/remap_labels_to_v14.py --dirs <that dir>\n"
            f"and point --pred_dir at <that dir>_v14.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    gts = sorted(Path(args.gt_dir).glob("*.npz"))
    conf = np.zeros((K, K), dtype=np.int64)     # rows = GT, cols = SAM3
    used = 0
    checked = False
    for g in gts:
        p = Path(args.pred_dir) / g.name
        if not p.exists():
            continue
        if not checked:
            assert_v14(p)
            checked = True
        gt, pr = load(g), load(p)
        if gt.shape != pr.shape:
            n = min(len(gt), len(pr))
            gt, pr = gt[:n], pr[:n]
            if gt.shape != pr.shape:
                print(f"  skip {g.stem}: {gt.shape} vs {pr.shape}", flush=True)
                continue
        m = (gt > 0) & (gt < K) & (pr >= 0) & (pr < K)   # GT void = unannotated
        conf += np.bincount((gt[m].astype(np.int64) * K + pr[m]),
                            minlength=K * K).reshape(K, K)
        used += 1
        if args.limit and used >= args.limit:
            break

    if used == 0:
        raise SystemExit(f"no overlapping stems between {args.pred_dir} and "
                         f"{args.gt_dir} — run sam3_precompute_labels.py on the "
                         f"val clips first")
    print(f"=== SAM3 vs SANPO human GT — {used} clips, "
          f"{conf.sum() / 1e6:.1f}M annotated pixels ===\n")

    tot = conf.sum(1)                                   # GT pixels per class
    print(f"{'GT class':<11}{'share':>7}{'recall':>8}{'precis':>8}{'IoU':>8}"
          f"   most common SAM3 answers")
    for i in range(K):
        if tot[i] == 0:
            continue
        tp = conf[i, i]
        rec = tp / tot[i]
        prec = tp / max(conf[:, i].sum(), 1)
        iou = tp / max(tot[i] + conf[:, i].sum() - tp, 1)
        top = np.argsort(-conf[i])[:3]
        answers = ", ".join(f"{V14[j]} {100 * conf[i, j] / tot[i]:.0f}%"
                            for j in top if conf[i, j] > 0)
        print(f"{V14[i]:<11}{tot[i] / conf.sum():>6.1%}{rec:>8.3f}{prec:>8.3f}"
              f"{iou:>8.3f}   {answers}")

    def pr(gt_name, pred_name):
        i, j = V14.index(gt_name), V14.index(pred_name)
        return conf[i, j] / tot[i] if tot[i] else float("nan")

    print("\n=== the two decisions this was run for ===")
    print(f"  P(SAM3=road     | GT=sidewalk) = {pr('sidewalk', 'road'):.3f}"
          f"   <- sets road's traversability score")
    print(f"  P(SAM3=sidewalk | GT=road)     = {pr('road', 'sidewalk'):.3f}")
    print(f"  P(SAM3=grass    | GT=sidewalk) = {pr('sidewalk', 'grass'):.3f}"
          f"   <- ground-level proximity term")
    print(f"  P(SAM3=sidewalk | GT=grass)    = {pr('grass', 'sidewalk'):.3f}"
          f"   <- the DANGEROUS one: grass called walkable")
    walk = [V14.index(c) for c in ("sidewalk", "pavement", "road", "trail")]
    nonw = [V14.index(c) for c in ("grass", "obstacle", "vegetation", "water")]
    wn = conf[np.ix_(walk, nonw)].sum() / max(conf[walk].sum(), 1)
    nw = conf[np.ix_(nonw, walk)].sum() / max(conf[nonw].sum(), 1)
    print(f"\n  walkable  -> non-walkable confusion: {wn:.3f}  (robot needlessly stops)")
    print(f"  non-walk. -> walkable confusion:     {nw:.3f}  (robot walks somewhere it shouldn't)")

    if args.csv:
        np.savetxt(args.csv, conf, fmt="%d", delimiter=",",
                   header=",".join(V14), comments="")
        print(f"\n==> confusion matrix: {args.csv}")


if __name__ == "__main__":
    main()
