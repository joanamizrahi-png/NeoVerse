"""Survey a ROS 2 bag's odometry and propose clip windows for prepare_rosbag_clips.py.

Reads ONLY the odometry topic (a few MB even in a 70 GB bag), so it is fast.
It writes, next to --out:

  <name>_odom_full.csv     t_sec, x, y, yaw_deg, dist_m (cumulative), v_mps
  <name>_topdown.png       the whole drive from above, coloured by time, with the
                           proposed clip windows drawn on it
  <name>_windows.csv       one row per proposed clip: name, start_sec, duration_sec,
                           length_m, moving_frac
  <name>_commands.sh       the exact prepare_rosbag_clips.py line for every window

Windows are cut by DISTANCE, not time: a clip every --step_m metres of travel,
each spanning --clip_m metres, so the 81 sampled frames land ~clip_m/80 apart
whatever the walking speed. Stretches where the robot stood still (< --still_mps
for > --still_sec) are skipped, so no clip is 81 frames of one spot.

    python scripts/bag_survey.py --bag /path/to/bag_dir --name campusA \
        --out /path/to/clips [--odom_topic auto] [--clip_m 30] [--step_m 15]

Then run the lines in <name>_commands.sh. The bag directory must contain
metadata.yaml (run `ros2 bag reindex <dir> mcap` once if it only has the .mcap).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

ODOM_TYPES = {"nav_msgs/msg/Odometry"}


def yaw_from_quat(x, y, z, w) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def read_odometry(bag: Path, odom_topic: str):
    from rosbags.highlevel import AnyReader
    with AnyReader([bag]) as reader:
        odos = {c.topic: c for c in reader.connections if c.msgtype in ODOM_TYPES}
        if not odos:
            raise SystemExit("no nav_msgs/msg/Odometry topic in this bag; run "
                             "prepare_rosbag_clips.py --inspect to list topics")
        if odom_topic == "auto":
            if len(odos) > 1:
                print(f"[survey] several odometry topics, taking the first: {sorted(odos)}")
            odom_topic = sorted(odos)[0]
        conn = odos[odom_topic]
        t_ns, xyz, yaw = [], [], []
        for c, t, raw in reader.messages(connections=[conn]):
            m = reader.deserialize(raw, c.msgtype)
            p, q = m.pose.pose.position, m.pose.pose.orientation
            t_ns.append(t)
            xyz.append((p.x, p.y, p.z))
            yaw.append(yaw_from_quat(q.x, q.y, q.z, q.w))
        start_ns = reader.start_time
    t = (np.asarray(t_ns, dtype=np.float64) - start_ns) / 1e9
    return odom_topic, t, np.asarray(xyz, dtype=np.float64), np.asarray(yaw, dtype=np.float64)


def moving_mask(t, xy, still_mps, still_sec):
    """True where the robot is moving. Speed from a 1 s window; stillness only
    counts when it lasts longer than still_sec."""
    n = len(t)
    v = np.zeros(n)
    for i in range(n):
        j = np.searchsorted(t, t[i] + 1.0)
        j = min(max(j, i + 1), n - 1)
        dt = max(t[j] - t[i], 1e-6)
        v[i] = np.linalg.norm(xy[j] - xy[i]) / dt
    still = v < still_mps
    # dilate short still gaps back to moving so a brief pause does not cut a clip
    mask = ~still
    i = 0
    while i < n:
        if still[i]:
            j = i
            while j < n and still[j]:
                j += 1
            if t[j - 1] - t[i] < still_sec:
                mask[i:j] = True
            i = j
        else:
            i += 1
    return mask, v


def propose_windows(t, dist, moving, clip_m, step_m, min_moving_frac):
    """Windows by distance travelled. Returns list of (start_sec, duration_sec,
    length_m, moving_frac)."""
    out = []
    total = float(dist[-1])
    d0 = 0.0
    while d0 + clip_m <= total + 1e-6:
        i0 = int(np.searchsorted(dist, d0))
        i1 = int(np.searchsorted(dist, d0 + clip_m))
        i1 = min(i1, len(t) - 1)
        mf = float(moving[i0:i1 + 1].mean()) if i1 > i0 else 0.0
        if mf >= min_moving_frac:
            out.append((float(t[i0]), float(t[i1] - t[i0]), float(dist[i1] - dist[i0]), mf))
        d0 += step_m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--odom_topic", default="auto")
    ap.add_argument("--image_topic", default="auto",
                    help="written into the commands file only")
    ap.add_argument("--clip_m", type=float, default=30.0, help="metres per clip")
    ap.add_argument("--step_m", type=float, default=15.0, help="metres between clip starts")
    ap.add_argument("--still_mps", type=float, default=0.05)
    ap.add_argument("--still_sec", type=float, default=2.0)
    ap.add_argument("--min_moving_frac", type=float, default=0.8,
                    help="drop windows where the robot stood still more than this share")
    ap.add_argument("--pano_topic", default=None, help="forwarded to the commands file")
    args = ap.parse_args()

    topic, t, xyz, yaw = read_odometry(args.bag, args.odom_topic)
    if len(t) < 10:
        raise SystemExit(f"only {len(t)} odometry samples on {topic}")
    xy = xyz[:, :2]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(step)])
    moving, v = moving_mask(t, xy, args.still_mps, args.still_sec)
    print(f"[survey] {topic}: {len(t)} samples, {t[-1]:.0f} s, {dist[-1]:.0f} m travelled, "
          f"moving {moving.mean() * 100:.0f}% of the time, median speed while moving "
          f"{np.median(v[moving]) if moving.any() else 0:.2f} m/s")

    wins = propose_windows(t, dist, moving, args.clip_m, args.step_m, args.min_moving_frac)
    print(f"[survey] {len(wins)} windows of {args.clip_m:.0f} m every {args.step_m:.0f} m")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / args.name
    with open(f"{stem}_odom_full.csv", "w") as fh:
        fh.write("t_sec,x,y,yaw_deg,dist_m,v_mps\n")
        for i in range(len(t)):
            fh.write(f"{t[i]:.3f},{xy[i, 0]:.3f},{xy[i, 1]:.3f},{math.degrees(yaw[i]):.1f},"
                     f"{dist[i]:.2f},{v[i]:.3f}\n")
    with open(f"{stem}_windows.csv", "w") as fh:
        fh.write("clip,start_sec,duration_sec,length_m,moving_frac\n")
        for k, (s, d, L, mf) in enumerate(wins):
            fh.write(f"{args.name}_{k:02d},{s:.1f},{d:.1f},{L:.1f},{mf:.2f}\n")
    with open(f"{stem}_commands.sh", "w") as fh:
        fh.write("#!/usr/bin/env bash\nset -e\n")
        pano = f" --pano_topic {args.pano_topic}" if args.pano_topic else ""
        for k, (s, d, L, mf) in enumerate(wins):
            fh.write(f"python scripts/prepare_rosbag_clips.py --bag {args.bag} "
                     f"--out {args.out} --name {args.name}_{k:02d} "
                     f"--image_topic {args.image_topic} --odom_topic {topic} "
                     f"--start_sec {s:.1f} --duration {d:.1f}{pano}   # {L:.0f} m\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(xy[:, 0], xy[:, 1], c=t, s=2, cmap="viridis")
        ax.plot(xy[~moving, 0], xy[~moving, 1], ".", color="red", ms=3, label="standing still")
        for k, (s, d, L, mf) in enumerate(wins):
            i0 = int(np.searchsorted(t, s)); i1 = int(np.searchsorted(t, s + d))
            ax.plot(xy[i0:i1, 0], xy[i0:i1, 1], "-", lw=1, alpha=0.6)
            ax.annotate(f"{k:02d}", xy[i0], fontsize=7)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="best")
        ax.set_title(f"{args.name}: {dist[-1]:.0f} m, {len(wins)} clips of {args.clip_m:.0f} m")
        fig.savefig(f"{stem}_topdown.png", dpi=150, bbox_inches="tight")
        print(f"[survey] wrote {stem}_topdown.png")
    except Exception as e:  # matplotlib is optional on the robot
        print(f"[survey] no plot ({e})")
    print(f"[survey] wrote {stem}_windows.csv and {stem}_commands.sh")


if __name__ == "__main__":
    main()
