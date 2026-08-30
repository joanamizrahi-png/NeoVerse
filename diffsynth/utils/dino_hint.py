"""DINOv2 conditioning hint for the semantic diffusion model (v25, 2026-08-30).

Motivation (Joana): the semantic half of the DiT must infer "this blurry
region is road / sidewalk / grass" from splat-render pixels plus SAM3's noisy
hint. DINOv2 separates those materials without any training and is robust to
blur, so we hand its patch features to the control branch as a fourth hint —
input-only, so a one-way encoder is fine here (it could never replace the VAE,
which must decode the generated semantic latent back to pixels).

Wiring (all inert unless `dino_hint_channels > 0` in the training config, or
`control_patch_embedding.dino_proj.*` keys exist in an inference checkpoint):
  - compute_dino_hint():  pixel-space splat render -> [B, 384, T_lat, H/8, W/8]
                          (called in WanVideoUnit_4DEmbedder, both regimes)
  - attach_dino_hint():   adds a ZERO-INIT Conv3d projection onto the existing
                          SplitControlPatchEmbedding (same trick as the .sem
                          conv) -- at init the run is bit-identical to non-DINO
  - model_fn stashes the features on the embedding; its forward consumes them

The DINO backbone itself is frozen, lives in a module-level cache (NOT on the
pipe -- it must never enter the checkpoint), and is loaded from the torch.hub
cache (~/.cache/torch/hub). Warm the cache on a login node before the first
compute-node run:  python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

_DINO_CACHE = {"model": None}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_dino(device, dtype):
    if _DINO_CACHE["model"] is None:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _DINO_CACHE["model"] = model
    return _DINO_CACHE["model"].to(device=device, dtype=dtype)


@torch.no_grad()
def compute_dino_hint(pipe, target_rgb: torch.Tensor) -> torch.Tensor:
    """Frozen-DINOv2 features of the splat render, on the latent grid.

    target_rgb: [B, T, C, H, W] float in [0, 1] -- the PIXEL-space render,
    i.e. before preprocess_video() shifts it to [-1, 1].

    Returns [B, 384, T_lat, H/8, W/8] in pipe.torch_dtype, where the T_lat
    frames are indices 0, 4, 8, ... -- the last frame of each VAE temporal
    group, matching the stride-4 conv the mask/cam hint already uses.
    """
    B, T, C, H, W = target_rgb.shape
    assert H % 14 == 0 and W % 14 == 0, (
        f"DINO hint needs 14-divisible resolution, got {H}x{W} "
        "(all 112-multiple resolutions in this stack qualify)")
    idx = torch.arange(0, T, 4, device=target_rgb.device)
    frames = target_rgb.index_select(1, idx)                    # [B, T', C, H, W]
    t_lat = frames.shape[1]
    x = rearrange(frames, "b t c h w -> (b t) c h w").to(
        device=pipe.device, dtype=pipe.torch_dtype)
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    dino = _get_dino(pipe.device, pipe.torch_dtype)
    tokens = dino.forward_features(x)["x_norm_patchtokens"]     # [(B T'), ph*pw, 384]
    ph, pw = H // 14, W // 14
    grid = tokens.transpose(1, 2).reshape(B * t_lat, -1, ph, pw)
    grid = F.interpolate(grid, size=(H // 8, W // 8),
                         mode="bilinear", align_corners=False)
    return rearrange(grid, "(b t) c h w -> b c t h w", b=B)


def attach_dino_hint(control_branch, dino_dim: int = 384):
    """Add a zero-init dino_proj Conv3d onto the (already v2-expanded)
    control_patch_embedding. Idempotent. Call AFTER
    expand_control_branch_for_semantics_v2 and BEFORE freeze/checkpoint-load.

    Trainable path: control_branch.control_patch_embedding.dino_proj
    (add it to `trainable_models` in the training yaml).
    """
    from .semantics import SplitControlPatchEmbedding
    cpe = control_branch.control_patch_embedding
    assert isinstance(cpe, SplitControlPatchEmbedding), (
        "DINO hint requires semantic_expansion_version: 2 "
        "(SplitControlPatchEmbedding); got " + type(cpe).__name__)
    if getattr(cpe, "dino_proj", None) is not None:
        return control_branch
    proj = torch.nn.Conv3d(
        dino_dim, cpe.out_channels,
        kernel_size=cpe.kernel_size, stride=cpe.stride,
    ).to(cpe.base.weight.device, cpe.base.weight.dtype)
    with torch.no_grad():
        proj.weight.zero_()
        if proj.bias is not None:
            proj.bias.zero_()
    cpe.dino_proj = proj
    cpe._dino_feats = None
    return control_branch
