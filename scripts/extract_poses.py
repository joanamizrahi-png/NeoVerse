"""Extract real per-frame camera poses + intrinsics for a video clip.

Runs NeoVerse's WorldMirror reconstructor on a clip, saves the estimated
camera trajectory + intrinsics to a .npz that nav-rl's reward validation
consumes via `--pose_source npz`.

Output npz schema:
    positions   [T, 3]        camera position in world coords per frame
    headings    [T, 3]        camera forward unit vector per frame
    w2c         [T, 4, 4]     world-to-camera transform per frame
    K           [3, 3]        camera intrinsics (assumed constant across frames)
    num_frames  scalar        T (for sanity checking)

Usage:
    python scripts/extract_poses.py \\
        --input_path examples/videos/driving.mp4 \\
        --model_path /scratch/.../NeoVerse/models \\
        --reconstructor_path /scratch/.../NeoVerse/models/NeoVerse/reconstructor.ckpt
Output: outputs/poses/<clip_stem>.npz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as F

from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth.utils.auxiliary import load_video, homo_matrix_inverse


def _extract_headings(extrinsics_c2w: np.ndarray) -> np.ndarray:
    """From camera-to-world matrices, extract per-frame forward direction.

    Camera-to-world's rotation columns are (right, down, forward_world_dir).
    Our reward function expects "heading" = the direction the robot is moving,
    which for a forward-mounted camera on a walking robot = the camera's forward.
    In camera frame, "forward" is +z. In world frame that's the 3rd column of R.
    """
    R = extrinsics_c2w[:, :3, :3]           # (T, 3, 3)
    forward_cam = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # For each frame: world_forward = R @ forward_cam = R[:, :, 2]
    headings = R[:, :, 2].astype(np.float32)
    # Normalize (should already be unit-length; belt-and-suspenders)
    norms = np.linalg.norm(headings, axis=1, keepdims=True)
    return headings / np.maximum(norms, 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, type=Path)
    ap.add_argument("--output_dir", default="outputs/poses", type=Path)
    ap.add_argument("--model_path", required=True, type=Path,
                    help="Base NeoVerse model directory (has NeoVerse/reconstructor.ckpt)")
    ap.add_argument("--reconstructor_path", required=True, type=Path)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--resize_mode", choices=["center_crop", "resize"], default="center_crop")
    ap.add_argument("--static_scene", action="store_true")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input_path.stem
    out_path = args.output_dir / f"{stem}.npz"

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # --- Load pipeline (we only need the reconstructor, not the diffusion) ---
    print(f"[extract_poses] loading reconstructor from {args.reconstructor_path}", flush=True)
    pipe = WanVideoNeoVersePipeline.from_pretrained(
        local_model_path=str(args.model_path),
        reconstructor_path=str(args.reconstructor_path),
        device="cuda",
        torch_dtype=dtype,
    )

    # --- Load frames ---
    print(f"[extract_poses] loading video: {args.input_path}", flush=True)
    images = load_video(
        str(args.input_path), args.num_frames,
        resolution=(args.width, args.height),
        resize_mode=args.resize_mode,
        static_scene=args.static_scene,
    )
    T = len(images)
    device = pipe.device
    views = {
        "img": torch.stack([F.to_tensor(im)[None] for im in images], dim=1).to(device),
        "is_target": torch.zeros((1, T), dtype=torch.bool, device=device),
    }
    if args.static_scene:
        views["is_static"] = torch.ones((1, T), dtype=torch.bool, device=device)
        views["timestamp"] = torch.zeros((1, T), dtype=torch.int64, device=device)
    else:
        views["is_static"] = torch.zeros((1, T), dtype=torch.bool, device=device)
        views["timestamp"] = torch.arange(0, T, dtype=torch.int64, device=device).unsqueeze(0)

    # --- Run reconstructor ---
    print(f"[extract_poses] running reconstructor on {T} frames ...", flush=True)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
        predictions = pipe.reconstructor(views, is_inference=True, use_motion=False)

    # Predictions per-frame:
    # rendered_extrinsics [B, T, 4, 4]  (camera-to-world)
    # rendered_intrinsics [B, T, 3, 3]  (K per frame, usually the same K throughout)
    c2w = predictions["rendered_extrinsics"][0].detach().cpu().float().numpy()   # (T, 4, 4)
    K_per_frame = predictions["rendered_intrinsics"][0].detach().cpu().float().numpy()   # (T, 3, 3)
    K = K_per_frame[0]   # take first frame's K as the constant K

    # Positions = translation part of c2w
    positions = c2w[:, :3, 3].astype(np.float32)   # (T, 3)

    # Headings = camera's forward direction in world frame
    headings = _extract_headings(c2w)              # (T, 3)

    # world -> camera: invert c2w
    w2c = homo_matrix_inverse(torch.from_numpy(c2w)).numpy().astype(np.float32)   # (T, 4, 4)

    print(f"[extract_poses] positions range: x=[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}]  "
          f"y=[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]  "
          f"z=[{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]", flush=True)

    np.savez_compressed(
        out_path,
        positions=positions,
        headings=headings,
        w2c=w2c,
        K=K.astype(np.float32),
        num_frames=T,
    )
    print(f"[extract_poses] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
