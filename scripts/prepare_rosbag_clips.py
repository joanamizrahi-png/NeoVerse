"""Generic rosbag -> NeoVerse clip converter (GND / SCAND / MuSoHu / lab bags).

Extracts an 81-frame RGB clip (560x336 mp4, our pipeline format) plus the
odometry over the same window (the GT demonstration trajectory) from any bag
with an image topic and an odometry topic. No ROS install needed — uses the
pure-python `rosbags` package (pip install rosbags).

Outputs:
  <out>/<name>.mp4        81 frames, 560x336 — feed to the standard pipeline
                          (reconstruct -> SAM3 -> extract_poses -> report card)
  <out>/<name>_odom.npz   t [N], xyz [N,3], quat [N,4] over the clip window —
                          the GT trajectory, time-aligned to the frames

Usage:
  python scripts/prepare_rosbag_clips.py --bag path/to/run.bag \
      --out /scratch/.../data/gnd_clips --name gnd_campusA_00 \
      [--image_topic auto] [--odom_topic auto] [--start_sec 30] [--duration 20]

List a bag's topics first with --inspect.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def open_reader(bag_path: Path):
    from rosbags.highlevel import AnyReader
    return AnyReader([bag_path])


IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
ODOM_TYPES = {"nav_msgs/msg/Odometry"}


def pick_topics(reader, image_topic, odom_topic):
    imgs = {c.topic: c.msgtype for c in reader.connections if c.msgtype in IMAGE_TYPES}
    odos = {c.topic: c.msgtype for c in reader.connections if c.msgtype in ODOM_TYPES}
    if image_topic == "auto":
        assert imgs, f"no image topics; run --inspect. Topics: {[c.topic for c in reader.connections]}"
        # prefer compressed color, avoid depth topics
        ranked = sorted(imgs, key=lambda t: ("depth" in t, "compressed" not in t, t))
        image_topic = ranked[0]
    if odom_topic == "auto":
        odom_topic = sorted(odos)[0] if odos else None
    return image_topic, odom_topic


def decode_image(msg, msgtype):
    import cv2
    if "Compressed" in msgtype:
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    data = np.frombuffer(msg.data, np.uint8)
    enc = msg.encoding.lower()
    h, w = msg.height, msg.width
    if enc in ("bgr8", "rgb8"):
        img = data.reshape(h, w, 3)
        return img[:, :, ::-1] if enc == "bgr8" else img
    if enc in ("bgra8", "rgba8"):
        img = data.reshape(h, w, 4)[:, :, :3]
        return img[:, :, ::-1] if enc == "bgra8" else img
    raise ValueError(f"unhandled encoding {msg.encoding}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, type=Path)
    ap.add_argument("--inspect", action="store_true", help="list topics and exit")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--image_topic", default="auto")
    ap.add_argument("--odom_topic", default="auto")
    ap.add_argument("--start_sec", type=float, default=0.0,
                    help="offset into the bag before the clip window starts")
    ap.add_argument("--duration", type=float, default=20.0,
                    help="seconds of bag time the 81 frames span")
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=336)
    args = ap.parse_args()

    with open_reader(args.bag) as reader:
        if args.inspect:
            for c in sorted(reader.connections, key=lambda c: c.topic):
                print(f"  {c.topic:50s} {c.msgtype} ({c.msgcount} msgs)")
            dur = (reader.end_time - reader.start_time) / 1e9
            print(f"bag duration: {dur:.1f}s")
            return
        assert args.out and args.name, "--out and --name required (or use --inspect)"

        image_topic, odom_topic = pick_topics(reader, args.image_topic, args.odom_topic)
        print(f"image topic: {image_topic}\nodom topic:  {odom_topic}")

        t0 = reader.start_time + int(args.start_sec * 1e9)
        t1 = t0 + int(args.duration * 1e9)
        conns_img = [c for c in reader.connections if c.topic == image_topic]

        # collect frame timestamps first (light pass), pick num_frames evenly
        stamps = [t for c, t, _ in reader.messages(connections=conns_img)
                  if t0 <= t <= t1]
        assert len(stamps) >= args.num_frames, (
            f"only {len(stamps)} frames in window; shrink --start_sec/--duration")
        picks = set(np.linspace(0, len(stamps) - 1, args.num_frames).astype(int).tolist())
        picked_stamps = [s for i, s in enumerate(stamps) if i in picks]

        import cv2
        frames, k = [], 0
        for c, t, raw in reader.messages(connections=conns_img):
            if not (t0 <= t <= t1):
                continue
            if k in picks:
                msg = reader.deserialize(raw, c.msgtype)
                img = decode_image(msg, c.msgtype)
                # center-crop to target aspect then resize (matches load_video)
                H, W = img.shape[:2]
                ar = args.width / args.height
                ch = int(W / ar)
                if ch <= H:
                    y0 = (H - ch) // 2
                    img = img[y0:y0 + ch]
                else:
                    cw = int(H * ar)
                    x0 = (W - cw) // 2
                    img = img[:, x0:x0 + cw]
                frames.append(cv2.resize(img, (args.width, args.height),
                                         interpolation=cv2.INTER_AREA))
            k += 1
        print(f"extracted {len(frames)} frames")

        odom = {"t": [], "xyz": [], "quat": []}
        if odom_topic:
            conns_od = [c for c in reader.connections if c.topic == odom_topic]
            for c, t, raw in reader.messages(connections=conns_od):
                if not (t0 <= t <= t1):
                    continue
                msg = reader.deserialize(raw, c.msgtype)
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                odom["t"].append((t - t0) / 1e9)
                odom["xyz"].append([p.x, p.y, p.z])
                odom["quat"].append([q.x, q.y, q.z, q.w])
            print(f"extracted {len(odom['t'])} odometry samples")

    args.out.mkdir(parents=True, exist_ok=True)
    import cv2 as _cv
    vw = _cv.VideoWriter(str(args.out / f"{args.name}.mp4"),
                         _cv.VideoWriter_fourcc(*"mp4v"), 16,
                         (args.width, args.height))
    for f in frames:
        vw.write(f[:, :, ::-1])
    vw.release()
    np.savez_compressed(
        args.out / f"{args.name}_odom.npz",
        t=np.array(odom["t"]), xyz=np.array(odom["xyz"]),
        quat=np.array(odom["quat"]),
        frame_stamps=(np.array(picked_stamps) - t0) / 1e9)
    print(f"wrote {args.out}/{args.name}.mp4 + _odom.npz")


if __name__ == "__main__":
    main()
