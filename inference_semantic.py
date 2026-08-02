"""Inference for the semantic-finetuned NeoVerse model.

The vanilla `inference.py` was written for the pretrained 16-channel RGB path -- it
loads pipe, runs the diffusion, and decodes ONE video via the VAE. Our finetune
outputs a 32-channel latent (RGB half + semantic half), so we need extra machinery:

  1. Load base pipeline as usual
  2. Apply the same semantic expansions we used during training
     (expand_dit_for_semantics + expand_control_branch_for_semantics) so state-
     dict shapes match
  3. Load the finetune checkpoint (LoRA + expanded head weights) on top
  4. Monkey-patch `pipe.vae.decode` so that when it sees a 32-channel latent it
     decodes BOTH halves, stashes the semantic decoded video, and returns just
     the RGB half (so the pipeline's own return path stays intact)
  5. After pipe(...) returns, we have RGB video (in the return value) plus the
     semantic decoded video (in a closure dict)
  6. Semantic half comes out as RGB pixels in the same palette we colorized to
     during training. `rgb_to_labels` maps that back to class IDs, then we
     colorize with the canonical palette and save an MP4.

Usage:
  python inference_semantic.py \
      --input_path examples/videos/driving.mp4 \
      --trajectory static \
      --checkpoint /scratch/.../train_semantic_v1/checkpoint-epoch-5.safetensors \
      --output_dir outputs/inference_v1_driving/

Produces in --output_dir:
  rgb.mp4              -- RGB output from the diffusion
  semantic.mp4         -- Colorized semantic output (our palette)
  semantic_labels.npz  -- Raw class-id map [T, H, W] int8

NOTE: THIS SCRIPT HAS NOT YET BEEN RUNTIME-TESTED. The state-dict loading path and
the VAE-decode monkey-patch are the two most likely places to hit issues. See the
inline comments at those spots for what to check if it errors.
"""
import argparse, os
import numpy as np
import torch
from safetensors.torch import load_file
from torchvision.transforms import functional as F

from peft import LoraConfig, inject_adapter_in_model

from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth import save_video
from diffsynth.utils.auxiliary import CameraTrajectory, load_video, homo_matrix_inverse
from diffsynth.utils.semantics import (
    expand_dit_for_semantics,
    expand_control_branch_for_semantics,
    CLASS_COLORS,
    NUM_CLASSES,
    rgb_to_labels,
)


def _inject_lora_for_finetune(pipe, rank: int = 32, target_modules=None):
    """Inject peft LoRA slots into DiT to match training-time injection.

    IMPORTANT: this list MUST match `lora_target_modules` in the training yaml.
    v5 used q,k,v,o,ffn.0,ffn.2,patch_embedding,head.head. If a target that was
    trained is missing here, its LoRA tensors are silently discarded at
    _load_finetune_checkpoint (strict=False) -- and for patch_embedding /
    head.head specifically, whose semantic-slot base weights are zero-init and
    frozen, that means the ENTIRE semantic pathway is amputated at inference.
    That's exactly why v5 outputs looked like palette noise. If we ever add
    more target modules to training, update this list. The assertion below
    turns silent failure into a hard failure.
    """
    if target_modules is None:
        target_modules = ["q", "k", "v", "o", "ffn.0", "ffn.2",
                          "patch_embedding", "head.head"]
    lora_config = LoraConfig(r=rank, lora_alpha=rank, target_modules=target_modules)
    pipe.dit = inject_adapter_in_model(lora_config, pipe.dit)


def _load_finetune_checkpoint(pipe: WanVideoNeoVersePipeline, ckpt_path: str) -> None:
    """Load a train_semantic checkpoint (safetensors) into the pipeline in-place.

    The training config uses `remove_prefix_in_ckpt: pipe.` so keys on disk look
    like `dit.blocks.0.attn.q.weight`, `control_branch.control_patch_embedding.weight`, etc.
    (no `pipe.` prefix). We route each key to the correct submodule.
    """
    print(f"Loading finetune weights from {ckpt_path}", flush=True)
    state = load_file(ckpt_path)
    print(f"  {len(state)} tensors in checkpoint", flush=True)

    # Group by top-level module name (before the first dot).
    by_module = {}
    for key, val in state.items():
        if "." not in key:
            print(f"  skipping top-level tensor: {key}")
            continue
        top, rest = key.split(".", 1)
        by_module.setdefault(top, {})[rest] = val

    for name, submodule_state in by_module.items():
        target = getattr(pipe, name, None)
        if target is None or not hasattr(target, "load_state_dict"):
            print(f"  skipping {name}: not a loadable submodule of pipe")
            continue
        # strict=False so we tolerate LoRA-only checkpoints (base weights unchanged),
        # and so the reconstructor (loaded separately) doesn't error on missing keys.
        result = target.load_state_dict(submodule_state, strict=False)
        missing = getattr(result, "missing_keys", []) or []
        unexpected = getattr(result, "unexpected_keys", []) or []
        print(
            f"  loaded {name}: {len(submodule_state)} tensors, "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
        if unexpected:
            # Hard fail: silent unexpected keys are how we lost v5's semantic
            # pathway. If this ever fires, add the training-time module to
            # _inject_lora_for_finetune's target list.
            raise RuntimeError(
                f"{name}: {len(unexpected)} unexpected keys — first: "
                f"{unexpected[0]}. Update _inject_lora_for_finetune's "
                f"target_modules to match training config."
            )


def _make_dual_decode(orig_decode, sink: dict):
    """Return a monkey-patch for pipe.vae.decode that decodes both halves.

    When the incoming latent is 32-channel, split into rgb (first 16) and
    semantic (last 16), decode both through the same VAE, stash the semantic
    decoded video in `sink['sem_video']`, and return the RGB decoded video so
    the pipeline's own return value stays unchanged.
    """
    def decode(latents, **kwargs):
        if latents.shape[1] == 32:
            rgb_latents = latents[:, :16]
            sem_latents = latents[:, 16:]
            rgb_video = orig_decode(rgb_latents, **kwargs)
            sem_video = orig_decode(sem_latents, **kwargs)
            sink["sem_video"] = sem_video
            sink["rgb_video"] = rgb_video
            return rgb_video
        # Baseline 16-channel path (e.g. if the finetune expansion didn't apply)
        return orig_decode(latents, **kwargs)
    return decode


@torch.no_grad()
def _decoded_video_to_uint8(video_tensor: torch.Tensor) -> np.ndarray:
    """VAE decode returns bf16 in some layout close to [-1, 1] pixel space.
    Normalize to uint8 RGB in [T, H, W, 3].

    Handles the common Wan return shapes:
      * [B, C=3, T, H, W]      -- most common
      * [B, T, C=3, H, W]      -- some VAE variants
      * [T, C=3, H, W]         -- if pipeline already dropped the batch
    """
    x = video_tensor.detach().to(torch.float32).cpu()
    # Squeeze batch dim
    if x.ndim == 5 and x.shape[0] == 1:
        x = x[0]
    # Now expected [C, T, H, W] or [T, C, H, W]
    if x.ndim == 4:
        if x.shape[0] == 3:
            x = x.permute(1, 2, 3, 0)                 # [C, T, H, W] -> [T, H, W, C]
        elif x.shape[1] == 3:
            x = x.permute(0, 2, 3, 1)                 # [T, C, H, W] -> [T, H, W, C]
    x = (x + 1.0) * 0.5                                # from [-1,1] to [0,1]
    x = x.clamp_(0.0, 1.0) * 255.0
    return x.to(torch.uint8).numpy()


@torch.no_grad()
def _sem_video_to_labels_and_colorized(sem_video: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """From the VAE-decoded semantic video (RGB in [-1,1]) produce:
      * labels  : [T, H, W] int class ids (0..NUM_CLASSES-1)
      * sem_rgb : [T, H, W, 3] uint8 colorized with the canonical palette
                  (may look sharper than the raw VAE output; that's the point)
    """
    x = sem_video.detach().to(torch.float32).cpu()
    if x.ndim == 5 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 4:
        if x.shape[0] == 3:
            x = x.permute(1, 2, 3, 0)
        elif x.shape[1] == 3:
            x = x.permute(0, 2, 3, 1)
    x = ((x + 1.0) * 0.5).clamp_(0.0, 1.0)             # [T, H, W, 3] float in [0,1]

    # Nearest-palette-color snap -> class ids
    labels = rgb_to_labels(x)                          # [T, H, W] int
    labels_np = labels.cpu().numpy().astype(np.int8)

    # Colorize with the canonical palette (crisper than the VAE-decoded blobs)
    palette = (CLASS_COLORS.detach().cpu().float() * 255).clamp_(0, 255).to(torch.uint8).numpy()
    sem_rgb = palette[labels_np]                       # [T, H, W, 3] uint8
    return labels_np, sem_rgb


@torch.no_grad()
def semantic_inference(
    input_path: str,
    checkpoint: str,
    output_dir: str,
    model_path: str,
    reconstructor_path: str,
    trajectory: str = "static",
    prompt: str = "A smooth video with complete scene content. Inpaint any missing regions or margins naturally to match the surrounding scene.",
    negative_prompt: str = "",
    num_frames: int = 81,
    width: int = 560,
    height: int = 336,
    resize_mode: str = "center_crop",
    seed: int = 42,
    use_lora: bool = True,
    disable_semantic_channels: bool = False,
    alpha_threshold: float = 1.0,
    static_scene: bool = False,
    semantic_channels: int = 16,
    semantic_expansion_version: int = 1,   # match training config; 2 = v6 _sem split
    semantic_x0_prediction: bool = False,  # MUST match training: v8 ckpts True, v6/v7 False
    lora_rank: int = 32,                   # match training config's lora_rank
    lora_target_modules: "list[str] | None" = None,  # match training's list; None -> default
    zero_trunk_lora: bool = False,         # diagnostic: zero attention/FFN LoRA after load
    semantic_labels: "np.ndarray | None" = None,
):
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Base pipeline ----
    lora_path = None
    if use_lora:
        lora_path = os.path.join(
            model_path, "NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
        )
    print(f"Loading base pipeline from {model_path} ...", flush=True)
    pipe = WanVideoNeoVersePipeline.from_pretrained(
        local_model_path=model_path,
        reconstructor_path=reconstructor_path,
        lora_path=lora_path,
        lora_alpha=1.0,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    # ---- 2. Semantic expansion (MUST match training-time expansion) ----
    if not disable_semantic_channels:
        pipe.semantic_channels = semantic_channels
        pipe.semantic_x0_prediction = bool(semantic_x0_prediction)
        if semantic_expansion_version == 1:
            expand_dit_for_semantics(pipe.dit, extra=semantic_channels)
            if pipe.control_branch is not None:
                expand_control_branch_for_semantics(pipe.control_branch, extra=semantic_channels)
        elif semantic_expansion_version == 2:
            from diffsynth.utils.semantics import (
                expand_dit_for_semantics_v2,
                expand_control_branch_for_semantics_v2,
            )
            expand_dit_for_semantics_v2(pipe.dit, extra=semantic_channels)
            if pipe.control_branch is not None:
                expand_control_branch_for_semantics_v2(pipe.control_branch, extra=semantic_channels)
        else:
            raise ValueError(f"unknown semantic_expansion_version={semantic_expansion_version}")
        print(f"Applied semantic expansion v{semantic_expansion_version} (+{semantic_channels} latent channels)", flush=True)

    # ---- 2b. Inject LoRA on DiT to match training. Must happen BEFORE checkpoint load
    # or the LoRA weights get discarded as unexpected keys.
    _inject_lora_for_finetune(pipe, rank=lora_rank, target_modules=lora_target_modules)
    print(f"Injected LoRA slots on DiT (rank {lora_rank}, "
          f"targets={lora_target_modules or 'default'})", flush=True)

    # ---- 3. Load finetune weights ----
    _load_finetune_checkpoint(pipe, checkpoint)

    # ---- 3b. Optional diagnostic: zero the trunk LoRA (--zero_trunk_lora) ----
    # Hypothesis test for the v6 mottle: the shared attention/FFN LoRA is the ONLY
    # trained component that touches the RGB path (I/O layers + control branch
    # base are frozen). Zeroing lora_B reverts the trunk to the pristine merged
    # base while KEEPING the full-rank _sem I/O modules and control sem conv.
    #   -> RGB clean + semantic still structured  = drop/tame trunk LoRA (v8)
    #   -> RGB clean + semantic collapses         = trunk routing is needed; tune
    #      its LR/rank instead of removing it.
    if zero_trunk_lora:
        n_zeroed = 0
        with torch.no_grad():
            for name, param in pipe.dit.named_parameters():
                if "lora_B" in name:
                    param.zero_()
                    n_zeroed += 1
        print(f"[zero_trunk_lora] zeroed {n_zeroed} lora_B tensors — trunk = pristine base", flush=True)

    # ---- 4. Load input video ----
    print(f"Loading input video: {input_path}", flush=True)
    images = load_video(
        input_path, num_frames,
        resolution=(width, height),
        resize_mode=resize_mode,
        static_scene=static_scene,
    )
    print(f"  {len(images)} frames at {images[0].size}", flush=True)

    # ---- 5. Build camera trajectory ----
    cam_traj = CameraTrajectory.from_predefined(
        trajectory, num_frames=len(images), mode="relative",
    )

    # ---- 6. Reconstruct + render (same as inference.py) ----
    device = pipe.device
    views = {
        "img": torch.stack([F.to_tensor(image)[None] for image in images], dim=1).to(device),
        "is_target": torch.zeros((1, len(images)), dtype=torch.bool, device=device),
    }
    if static_scene:
        views["is_static"] = torch.ones((1, len(images)), dtype=torch.bool, device=device)
        views["timestamp"] = torch.zeros((1, len(images)), dtype=torch.int64, device=device)
    else:
        views["is_static"] = torch.zeros((1, len(images)), dtype=torch.bool, device=device)
        views["timestamp"] = torch.arange(0, len(images), dtype=torch.int64, device=device).unsqueeze(0)

    # ---- 6b. Semantic hint at inference ----
    # Real SAM3-of-input-frames labels: matches the training-time hint distribution
    # (holey semantic rasterization from labeled Gaussians). Fall back to zeros only
    # if the caller didn't pass any -- that gives palette-noise output because the
    # model wasn't trained to inpaint from a blank hint.
    if not disable_semantic_channels:
        if semantic_labels is not None:
            # load_video may crop to num_frames < labels.shape[0] (SAM3 was run at
            # the video's native length); slice labels to match if longer, error if shorter.
            if semantic_labels.shape[0] > len(images):
                semantic_labels = semantic_labels[:len(images)]
            assert semantic_labels.shape[0] == len(images), (
                f"label frames {semantic_labels.shape[0]} < video frames {len(images)}; "
                f"re-run sam3_precompute_labels.py with --num_frames {len(images)}"
            )
            views["labels"] = torch.as_tensor(
                semantic_labels, dtype=torch.long, device=device
            ).unsqueeze(0)  # [1, N, H, W]
            print(f"Using SAM3 semantic labels: {views['labels'].shape}", flush=True)
        else:
            views["labels"] = torch.zeros(
                (1, len(images), height, width), dtype=torch.uint8, device=device
            )
            print("WARNING: no --semantic_labels provided; using zero hint. "
                  "Semantic output will likely be palette-noise.", flush=True)

    with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
        predictions = pipe.reconstructor(views, is_inference=True, use_motion=False)

    gaussians = predictions["splats"]
    K = predictions["rendered_intrinsics"][0]
    input_cam2world = predictions["rendered_extrinsics"][0]
    timestamps = predictions["rendered_timestamps"][0]

    ratio = torch.linspace(1, cam_traj.zoom_ratio, K.shape[0], device=device)
    K_zoomed = K.clone()
    K_zoomed[:, 0, 0] *= ratio
    K_zoomed[:, 1, 1] *= ratio

    target_cam2world = cam_traj.c2w.to(device)
    if cam_traj.mode == "relative" and not static_scene:
        target_cam2world = input_cam2world @ target_cam2world
    target_world2cam = homo_matrix_inverse(target_cam2world)

    target_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
        gaussians,
        render_viewmats=[target_world2cam],
        render_Ks=[K_zoomed],
        render_timestamps=[timestamps],
        sh_degree=0, width=width, height=height,
    )
    target_mask = (target_alpha > alpha_threshold).float()

    if cam_traj.use_first_frame:
        target_rgb[0, 0] = views["img"][0, 0].permute(1, 2, 0)
        target_mask[0, 0] = 1.0

    # Second rasterizer pass over the SAME gaussians with feature="labels" produces
    # the rough HOLEY semantic hint the DiT expects (control_branch's expanded 112-ch
    # patch_embed = 32 latent + 16 semantic + 64 mask_cam). Without this, 4DPreprocesser's
    # fast-path returns target_semantic=None and 4DEmbedder produces target_semantic_latents=None,
    # which the control_branch then can't shape-match. Mirrors the training-time
    # rasterization call at wan_video_neoverse.py:506.
    target_semantic = None
    if not disable_semantic_channels and "labels" in views:
        sem_probs, _, _ = pipe.reconstructor.gs_renderer.rasterizer.forward(
            gaussians,
            render_viewmats=[target_world2cam],
            render_Ks=[K_zoomed],
            render_timestamps=[timestamps],
            sh_degree=0, width=width, height=height,
            feature="labels",
        )
        target_semantic = sem_probs.argmax(dim=-1).to(torch.long)

        # Save the RAW rasterized holey semantic hint so we can compare it
        # side-by-side with the diffusion-inpainted output. The point of the
        # semantic finetune is that the diffusion fills holes in this hint;
        # this MP4 is the "before", semantic.mp4 is the "after".
        os.makedirs(output_dir, exist_ok=True)
        holey_labels = target_semantic[0].detach().cpu().numpy().astype(np.int8)
        palette = (CLASS_COLORS.detach().cpu().float() * 255).clamp_(0, 255).to(torch.uint8).numpy()
        holey_rgb = palette[np.clip(holey_labels, 0, NUM_CLASSES - 1)]  # [T, H, W, 3]
        import imageio.v3 as iio
        holey_out = os.path.join(output_dir, "holey_semantic.mp4")
        iio.imwrite(holey_out, holey_rgb, fps=16,
                    codec="libx264", macro_block_size=1,
                    ffmpeg_params=["-pix_fmt", "yuv420p"])
        print(f"Saved holey semantic hint: {holey_out}", flush=True)

    wrapped_data = {
        "source_views": views,
        "target_rgb": target_rgb,
        "target_depth": target_depth,
        "target_mask": target_mask,
        "target_poses": target_cam2world.unsqueeze(0),
        "target_intrs": K_zoomed.unsqueeze(0),
        "target_semantic": target_semantic,
    }

    # ---- 7. Monkey-patch VAE decode to split the 32-ch output ----
    sink: dict = {}
    orig_decode = pipe.vae.decode
    pipe.vae.decode = _make_dual_decode(orig_decode, sink)

    # ---- 8. Run diffusion ----
    num_inference_steps = 4 if use_lora else 50
    cfg_scale = 1.0 if use_lora else 5.0
    print(f"Running diffusion ({num_inference_steps} steps, cfg={cfg_scale}) ...", flush=True)
    generated_frames = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed, rand_device=pipe.device,
        height=height, width=width, num_frames=len(target_cam2world),
        cfg_scale=cfg_scale, num_inference_steps=num_inference_steps, tiled=False,
        **wrapped_data,
    )

    # Restore vae.decode (in case caller reuses the pipe)
    pipe.vae.decode = orig_decode

    # ---- 9. Save RGB output ----
    rgb_out = os.path.join(output_dir, "rgb.mp4")
    save_video(generated_frames, rgb_out, fps=16)
    print(f"Saved RGB output: {rgb_out}", flush=True)

    # ---- 10. Post-process semantic output ----
    if "sem_video" not in sink:
        print("WARNING: no semantic output captured -- was the model actually finetuned "
              "with semantic_channels? Skipping semantic save.", flush=True)
        return

    labels, sem_rgb = _sem_video_to_labels_and_colorized(sink["sem_video"])
    import imageio.v3 as iio

    # Save the RAW VAE-decoded semantic video BEFORE palette snapping. semantic.mp4
    # snaps every pixel to the nearest palette color, which turns any blur into
    # per-pixel class confetti — it cannot distinguish "smooth but blurry prediction"
    # (fixable: sharpen with more training / better sampling) from "true noise"
    # (architecture problem). This video can.
    raw_rgb = _decoded_video_to_uint8(sink["sem_video"])
    raw_out = os.path.join(output_dir, "semantic_raw.mp4")
    iio.imwrite(raw_out, raw_rgb, fps=16,
                codec="libx264", macro_block_size=1,
                ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"Saved RAW decoded semantic MP4: {raw_out}", flush=True)

    # Save colorized (palette-snapped) semantic mp4
    sem_out = os.path.join(output_dir, "semantic.mp4")
    iio.imwrite(sem_out, sem_rgb, fps=16,
                codec="libx264", macro_block_size=1,
                ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"Saved semantic MP4: {sem_out}  (shape {sem_rgb.shape})", flush=True)

    # Save raw class-id map alongside
    labels_out = os.path.join(output_dir, "semantic_labels.npz")
    np.savez_compressed(labels_out, labels=labels, num_classes=NUM_CLASSES)
    print(f"Saved raw class ids: {labels_out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Semantic-finetuned NeoVerse inference")
    p.add_argument("--input_path", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="Path to a train_semantic checkpoint .safetensors")
    p.add_argument("--output_dir", default="outputs/inference_semantic")
    p.add_argument("--model_path", default="models",
                   help="Base NeoVerse model directory (has NeoVerse/*.safetensors + reconstructor.ckpt)")
    p.add_argument("--reconstructor_path", default="models/NeoVerse/reconstructor.ckpt")
    p.add_argument("--trajectory", default="static",
                   choices=["pan_left", "pan_right", "tilt_up", "tilt_down",
                            "move_left", "move_right", "push_in", "pull_out",
                            "boom_up", "boom_down", "orbit_left", "orbit_right",
                            "static"])
    p.add_argument("--prompt",
                   default="A smooth video with complete scene content. Inpaint any missing regions or margins naturally to match the surrounding scene.")
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--width", type=int, default=560)
    p.add_argument("--height", type=int, default=336)
    p.add_argument("--resize_mode", choices=["center_crop", "resize"], default="center_crop")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disable_lora", action="store_true",
                   help="Skip Wan's 4-step distilled LoRA (slower but sometimes cleaner)")
    p.add_argument("--static_scene", action="store_true")
    p.add_argument("--semantic_channels", type=int, default=16,
                   help="Must match training-time value")
    p.add_argument("--semantic_expansion_version", type=int, default=1, choices=[1, 2],
                   help="Must match training-time value. 1 = v3/4/5 in-place grow. "
                        "2 = v6 parallel _sem submodules.")
    p.add_argument("--lora_rank", type=int, default=32,
                   help="Must match training-time lora_rank")
    p.add_argument("--zero_trunk_lora", action="store_true",
                   help="Diagnostic: zero the attention/FFN LoRA after checkpoint load. "
                        "Isolates trunk-LoRA drift as the mottle source (see 3b comment).")
    p.add_argument("--lora_target_modules", default=None,
                   help="Comma-separated. Must match training-time list. "
                        "Default (None) = q,k,v,o,ffn.0,ffn.2,patch_embedding,head.head "
                        "(v5 default). For v6 pass 'q,k,v,o,ffn.0,ffn.2' since patch_embedding "
                        "and head.head are full-trainable sub-modules there, not LoRA'd.")
    p.add_argument("--semantic_labels", default=None,
                   help="Path to SAM3 label .npz (from sam3_precompute_labels.py). "
                        "If omitted, auto-looks for outputs/sam3_labels/<input-stem>.npz")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load SAM3 labels: explicit --semantic_labels, or auto-derive from input filename.
    # Mirrors vanilla inference.py's convention: outputs/sam3_labels/<stem>.npz
    semantic_labels = None
    labels_path = args.semantic_labels
    if labels_path is None:
        stem = os.path.splitext(os.path.basename(args.input_path))[0]
        auto = os.path.join("outputs/sam3_labels", f"{stem}.npz")
        labels_path = auto if os.path.exists(auto) else None
    if labels_path is not None:
        semantic_labels = np.load(labels_path)["labels"]
        print(f"Loaded semantic labels from {labels_path}: {semantic_labels.shape}", flush=True)

    semantic_inference(
        input_path=args.input_path,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        trajectory=args.trajectory,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        resize_mode=args.resize_mode,
        seed=args.seed,
        use_lora=not args.disable_lora,
        static_scene=args.static_scene,
        semantic_channels=args.semantic_channels,
        semantic_expansion_version=args.semantic_expansion_version,
        lora_rank=args.lora_rank,
        lora_target_modules=(args.lora_target_modules.split(",")
                             if args.lora_target_modules else None),
        zero_trunk_lora=args.zero_trunk_lora,
        semantic_labels=semantic_labels,
    )


if __name__ == "__main__":
    main()
