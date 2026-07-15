"""Full-pipeline freeze camera: reconstruct + rasterize + diffuse at a FIXED pose.

Unlike freeze_camera_demo.py (raw rasterizer only, holey output), this uses
the full NeoVerse pipeline including Wan 2.1 diffusion, so we get a clean
inpainted RGB. Same idea otherwise:

  * Reconstruct source video once (WorldMirror)
  * Set target_cam2world = frame 0's pose, REPEATED across all timestamps
  * Rasterize at that fixed pose per timestamp
  * Feed through the diffusion pipeline (VAE + DiT + control_branch)
  * Save clean RGB MP4

This is essentially inference.py with a one-line override on the trajectory.

Usage:
    python freeze_camera_diffuse.py \\
        --input_path examples/videos/driving.mp4 \\
        --checkpoint models/NeoVerse/reconstructor.ckpt \\
        --output_dir outputs/freeze_camera/diffused/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as F

from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth import save_video
from diffsynth.utils.auxiliary import load_video, homo_matrix_inverse


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", default="examples/videos/driving.mp4")
    ap.add_argument("--output_dir", default="outputs/freeze_camera/diffused")
    ap.add_argument("--model_path", default="/scratch/m000204-pm06b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path", default="/scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--freeze_frame", type=int, default=0,
                    help="which source-video frame's pose to hold the camera at")
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--use_lora", action="store_true", default=True,
                    help="use the 4-step distilled LoRA (fast)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default="A smooth video with complete scene content. "
                    "Inpaint any missing regions or margins naturally to match the surrounding scene.")
    ap.add_argument("--negative_prompt", default="")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load pipeline ----
    lora_path = None
    if args.use_lora:
        lora_path = os.path.join(
            args.model_path,
            "NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
        )
    print(f"[freeze-diffuse] loading pipeline from {args.model_path}", flush=True)
    pipe = WanVideoNeoVersePipeline.from_pretrained(
        local_model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        lora_path=lora_path, lora_alpha=1.0,
        device="cuda", torch_dtype=torch.bfloat16,
    )

    # ---- Load source video ----
    print(f"[freeze-diffuse] loading source video: {args.input_path}", flush=True)
    images = load_video(args.input_path, args.num_frames,
                        resolution=(args.width, args.height),
                        resize_mode="center_crop", static_scene=False)
    N = len(images)
    print(f"[freeze-diffuse] loaded {N} frames at {args.width}x{args.height}", flush=True)

    device = pipe.device
    views = {
        "img": torch.stack([F.to_tensor(im)[None] for im in images], dim=1).to(device),
        "is_target": torch.zeros((1, N), dtype=torch.bool, device=device),
        "is_static": torch.zeros((1, N), dtype=torch.bool, device=device),
        "timestamp": torch.arange(0, N, dtype=torch.int64, device=device).unsqueeze(0),
    }

    # ---- Reconstruct ----
    print(f"[freeze-diffuse] running reconstructor...", flush=True)
    with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
        predictions = pipe.reconstructor(views, is_inference=True, use_motion=False)

    gaussians = predictions["splats"]
    K = predictions["rendered_intrinsics"][0]                # [N, 3, 3]
    cam2world = predictions["rendered_extrinsics"][0]        # [N, 4, 4]
    timestamps = predictions["rendered_timestamps"][0]       # [N]

    # ---- FIX camera at frame `args.freeze_frame`. Rasterize at that fixed pose
    #      for every timestamp. Same idea as freeze_camera_demo but full pipeline.
    print(f"[freeze-diffuse] freezing camera at frame {args.freeze_frame} for all {N} timestamps", flush=True)
    fixed_c2w = cam2world[args.freeze_frame:args.freeze_frame + 1].repeat(N, 1, 1)
    fixed_w2c = homo_matrix_inverse(fixed_c2w)
    K_rep = K[args.freeze_frame:args.freeze_frame + 1].repeat(N, 1, 1)

    target_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
        gaussians, render_viewmats=[fixed_w2c], render_Ks=[K_rep],
        render_timestamps=[timestamps],
        sh_degree=0, width=args.width, height=args.height,
    )
    target_mask = (target_alpha > 1.0).float()

    wrapped_data = {
        "source_views": views,
        "target_rgb": target_rgb,
        "target_depth": target_depth,
        "target_mask": target_mask,
        "target_poses": fixed_c2w.unsqueeze(0),
        "target_intrs": K_rep.unsqueeze(0),
    }

    # ---- Diffuse ----
    print(f"[freeze-diffuse] running 4-step diffusion...", flush=True)
    generated = pipe(
        prompt=args.prompt, negative_prompt=args.negative_prompt,
        seed=args.seed, rand_device=pipe.device,
        height=args.height, width=args.width, num_frames=N,
        cfg_scale=1.0 if args.use_lora else 5.0,
        num_inference_steps=4 if args.use_lora else 50,
        tiled=False,
        **wrapped_data,
    )

    out_path = os.path.join(args.output_dir, f"freeze_frame{args.freeze_frame}_diffused.mp4")
    save_video(generated, out_path, fps=16)
    print(f"[freeze-diffuse] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
