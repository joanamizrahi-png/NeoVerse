"""Proof: freeze the camera at ONE world pose, advance only the timestamp.

This is the RL-relevant render path: reconstruct the 4D scene ONCE, then call the
rasterizer directly with (pose, timestamp) — no trajectory system, no diffusion.
We hold the camera pose fixed across all frames and let only time tick forward,
so the camera stands still while the scene's own dynamics play out.

Outputs (raw renders, no diffusion):
  outputs/freeze_camera/freeze_rgb.mp4       camera fixed, time advancing
  outputs/freeze_camera/freeze_semantic.mp4  same, semantic channel
  (for contrast) outputs/freeze_camera/orig_rgb.mp4  the ORIGINAL moving-camera render
"""
import os, sys, numpy as np, torch
from torchvision.transforms import functional as F
import imageio.v2 as imageio
from diffsynth.utils import ModelConfig
from diffsynth.models import ModelManager
from diffsynth.utils.auxiliary import load_video, homo_matrix_inverse

INPUT = "examples/videos/driving.mp4"
LABELS = "outputs/sam3_labels/driving.npz"
OUT = "outputs/freeze_camera"
FREEZE_FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # which pose to freeze at
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 24      # 81 OOMs the Jetson; fewer frames = less recon memory
USE_SEMANTICS = False                                         # RGB-only (labels are tied to the 81-frame sampling)
os.makedirs(OUT, exist_ok=True)
device = "cuda"

print("loading ONLY the reconstructor (no 14B diffusion -> low memory)...", flush=True)
torch_dtype = torch.bfloat16
mm = ModelManager()
cfg = ModelConfig(path="models/NeoVerse/reconstructor.ckpt", offload_device=device)
cfg.download_if_necessary()
mm.load_model(cfg.path, device=device, torch_dtype=torch_dtype)
reconstructor = mm.fetch_model("reconstructor")

images = load_video(INPUT, N_FRAMES, resolution=(560, 336), resize_mode="center_crop")
H, W = images[0].size[1], images[0].size[0]
print(f"loaded {len(images)} frames at {W}x{H}", flush=True)

views = {
    "img": torch.stack([F.to_tensor(im)[None] for im in images], dim=1).to(device),
    "is_target": torch.zeros((1, len(images)), dtype=torch.bool, device=device),
    "is_static": torch.zeros((1, len(images)), dtype=torch.bool, device=device),  # dynamic scene
    "timestamp": torch.arange(0, len(images), dtype=torch.int64, device=device).unsqueeze(0),
}
if USE_SEMANTICS:
    d = np.load(LABELS); colors = d["class_colors"].astype(np.uint8)
    views["labels"] = torch.as_tensor(d["labels"], dtype=torch.long, device=device).unsqueeze(0)

print("reconstructing 4D scene (once)...", flush=True)
with torch.amp.autocast("cuda", dtype=torch_dtype):
    predictions = reconstructor(views, is_inference=True, use_motion=False)

gaussians = predictions["splats"]
K = predictions["rendered_intrinsics"][0]            # [N,3,3]
cam2world = predictions["rendered_extrinsics"][0]    # [N,4,4]
timestamps = predictions["rendered_timestamps"][0]   # [N]
N = len(timestamps)
raster = reconstructor.gs_renderer.rasterizer


def render(viewmats, ks, ts, feature="rgb"):
    out = raster.forward(gaussians, render_viewmats=[viewmats], render_Ks=[ks],
                         render_timestamps=[ts], sh_degree=0, width=W, height=H, feature=feature)
    return out[0]                                     # [B,V,H,W,*] -> already batch-stacked


def save_rgb(t, path):
    arr = (t[0].clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)  # [N,H,W,3]
    imageio.mimsave(path, list(arr), fps=16)


def save_sem(t, path):
    idx = t[0].argmax(-1).cpu().numpy().astype(np.int32)   # [N,H,W]
    imageio.mimsave(path, list(colors[idx]), fps=16)


# (A) ORIGINAL: camera follows its real per-frame path, time advancing  -> baseline
print("rendering ORIGINAL moving-camera path...", flush=True)
w2c_orig = homo_matrix_inverse(cam2world)             # [N,4,4] real path
save_rgb(render(w2c_orig, K, timestamps), f"{OUT}/orig_rgb.mp4")

# (B) FREEZE: ONE pose repeated for every timestamp -> camera still, time advancing
print(f"rendering FROZEN camera at frame {FREEZE_FRAME}, advancing time...", flush=True)
fixed_pose = cam2world[FREEZE_FRAME:FREEZE_FRAME + 1].repeat(N, 1, 1)   # [N,4,4] identical
fixed_w2c = homo_matrix_inverse(fixed_pose)
K_rep = K[FREEZE_FRAME:FREEZE_FRAME + 1].repeat(N, 1, 1)
save_rgb(render(fixed_w2c, K_rep, timestamps), f"{OUT}/freeze_rgb.mp4")
if USE_SEMANTICS:
    save_sem(render(fixed_w2c, K_rep, timestamps, feature="labels"), f"{OUT}/freeze_semantic.mp4")
print(f"DONE. compare {OUT}/orig_rgb.mp4 (camera moves) vs {OUT}/freeze_rgb.mp4 (camera fixed, time only)", flush=True)
