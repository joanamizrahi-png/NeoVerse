"""Download the LABELED slice of SANPO-Real (Google's pedestrian-viewpoint
urban dataset, CC BY 4.0) for the v21 semantic fine-tune.

Why SANPO: it is the urban RUGD we were missing — real video at chest height
with dense temporally-consistent panoptic masks and an explicit walkability
taxonomy (road / sidewalk / crosswalk / curb / other-walkable). The v15-v20
regression traced to training urban clips on SAM3 pseudo-GT; SANPO gives
urban scenes HUMAN ground truth, making the recipe symmetric with RUGD.

Only ~237 of 701 sessions carry segmentation masks (~112K labeled frames,
6TB total dataset). This script:
  1. enumerates sessions under sanpo-real/ via the public GCS JSON API,
  2. keeps only sessions that have segmentation_masks/,
  3. downloads masks + matching LEFT camera frames (+ session metadata),
skipping files that already exist (resumable; re-run freely).

No gcloud/gsutil needed — plain HTTPS. Run as a CPU slurm job or on a login
node with modest --workers.

Usage:
    python scripts/download_sanpo.py \
        --out /scratch/m000204-pm06b/joana/data/sanpo \
        --max_sessions 60 --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "gresearch"
ROOT = "sanpo_dataset/v0"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://storage.googleapis.com/{BUCKET}"


def _get(url: str, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def list_prefixes(prefix: str) -> list:
    """Immediate 'subdirectories' under prefix (delimiter listing)."""
    out, token = [], None
    while True:
        q = {"prefix": prefix, "delimiter": "/", "maxResults": "1000"}
        if token:
            q["pageToken"] = token
        data = json.loads(_get(f"{API}?{urllib.parse.urlencode(q)}"))
        out += data.get("prefixes", [])
        token = data.get("nextPageToken")
        if not token:
            return out


def list_objects(prefix: str) -> list:
    """All object names under prefix (recursive)."""
    out, token = [], None
    while True:
        q = {"prefix": prefix, "maxResults": "1000"}
        if token:
            q["pageToken"] = token
        data = json.loads(_get(f"{API}?{urllib.parse.urlencode(q)}"))
        out += [item["name"] for item in data.get("items", [])]
        token = data.get("nextPageToken")
        if not token:
            return out


def download_one(name: str, out_root: Path) -> str:
    rel = name[len(ROOT) + 1:]
    dest = out_root / rel
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MEDIA}/{urllib.parse.quote(name)}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(_get(url))
    tmp.rename(dest)
    return "dl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_sessions", type=int, default=60,
                    help="labeled sessions to fetch (60 =~ tens of thousands "
                         "of labeled frames; raise later if v21 wants more)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list_only", action="store_true",
                    help="just print labeled sessions and estimated counts")
    args = ap.parse_args()
    out_root = Path(args.out)

    # Layout (verified 2026-08-28):
    #   <session>/camera_chest/left/segmentation_masks/NNNNNN.png
    #   <session>/camera_chest/left/video_frames/NNNNNN.png
    #   <session>/camera_chest/left/frame_segmentation_annotation_type.json
    # We take the CHEST camera only — the closest viewpoint to a small robot.
    sessions = list_prefixes(f"{ROOT}/sanpo-real/")
    print(f"{len(sessions)} sessions under sanpo-real/", flush=True)

    labeled = []
    for i, s in enumerate(sessions):
        cam = s + "camera_chest/left/"
        masks = list_objects(cam + "segmentation_masks/")
        if masks:
            labeled.append((s, cam, masks))
            print(f"  [{len(labeled)}] {s.split('/')[-2]} -> "
                  f"{len(masks)} masks", flush=True)
        if len(labeled) >= args.max_sessions:
            break
        if i % 50 == 0 and i > 0:
            print(f"  ...scanned {i} sessions, {len(labeled)} labeled so far",
                  flush=True)

    print(f"==> {len(labeled)} labeled sessions selected", flush=True)
    if args.list_only:
        return

    for si, (s, cam, masks) in enumerate(labeled):
        mask_ids = {m.rsplit("/", 1)[-1] for m in masks}
        wanted = list(masks)
        # only frames that HAVE a mask (labeled sub-video slice)
        wanted += [f for f in list_objects(cam + "video_frames/")
                   if f.rsplit("/", 1)[-1] in mask_ids]
        wanted += [cam + "frame_segmentation_annotation_type.json",
                   s + "description.json", cam.rsplit("left/", 1)[0]
                   + "camera_poses.csv"]
        done = {"dl": 0, "skip": 0, "err": 0}

        def grab(name):
            try:
                return download_one(name, out_root)
            except Exception:
                return "err"

        with cf.ThreadPoolExecutor(args.workers) as ex:
            for res in ex.map(grab, wanted):
                done[res] += 1
        print(f"[{si + 1}/{len(labeled)}] {s.split('/')[-2]}  "
              f"downloaded={done['dl']} skipped={done['skip']} "
              f"errors={done['err']}", flush=True)

    print("==> sanpo download done", flush=True)


if __name__ == "__main__":
    main()
