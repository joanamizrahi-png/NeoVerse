"""Semantic-finetune utilities: colorize SAM3 labels for the VAE, and grow the DiT
to jointly generate a semantic latent alongside RGB.

Design (see docs/FINETUNE_IMPLEMENTATION.md):
- Discrete class masks are NOT image-like, so they don't encode cleanly into the
  RGB-pretrained VAE. We COLORIZE them (class -> fixed RGB color) first, then encode
  through the SAME VAE — exactly how NeoVerse already handles depth.
- Semantics is a GENERATION target: its latent is channel-concatenated with the RGB
  latent (16 -> 32), so the DiT learns to output [RGB ; semantic] jointly. We expand
  the DiT's input patch-embedding and output head, ZERO-INITIALIZING the new channels
  so the pretrained RGB behavior is unchanged at step 0; only the new channels learn.

UNTESTED — pending a cluster smoke-test (training can't run on the Jetson).
"""
import math
import torch

# Class index -> RGB color = RUGD's OFFICIAL 24-class colormap (index 0 = void/unlabeled),
# extracted+verified from the RUGD annotation masks. Single source of truth: SAM3 prompts
# (sam3_precompute_labels.CLASSES), RUGD ground-truth masks, and this decode table all share
# this taxonomy -> hint (SAM3) and clean target (RUGD) live in the SAME class space, so the
# diffusion only DENOISES/INPAINTS, never translates taxonomies.
CLASS_COLORS = torch.tensor([
    [0,   0,   0],     # 0  void / unlabeled
    [108, 64,  20],    # 1  dirt
    [255, 229, 204],   # 2  sand
    [0,   102, 0],     # 3  grass
    [0,   255, 0],     # 4  tree
    [0,   153, 153],   # 5  pole
    [0,   128, 255],   # 6  water
    [0,   0,   255],   # 7  sky
    [255, 255, 0],     # 8  vehicle
    [255, 0,   127],   # 9  container / generic-object
    [64,  64,  64],    # 10 asphalt
    [255, 128, 0],     # 11 gravel
    [255, 0,   0],     # 12 building
    [153, 76,  0],     # 13 mulch
    [102, 102, 0],     # 14 rock-bed
    [102, 0,   0],     # 15 log
    [0,   255, 128],   # 16 bicycle
    [204, 153, 255],   # 17 person
    [102, 0,   204],   # 18 fence
    [255, 153, 204],   # 19 bush
    [0,   102, 102],   # 20 sign
    [153, 204, 255],   # 21 rock
    [102, 255, 255],   # 22 bridge
    [101, 101, 11],    # 23 concrete
    [114, 85,  47],    # 24 picnic-table
], dtype=torch.float32) / 255.0          # [K, 3], values in [0, 1]
NUM_CLASSES = CLASS_COLORS.shape[0]


def labels_to_rgb(labels: torch.Tensor) -> torch.Tensor:
    """[*, H, W] int class ids  ->  [*, H, W, 3] float in [0,1] (colorized image)."""
    idx = labels.long().clamp(0, NUM_CLASSES - 1)
    return CLASS_COLORS.to(labels.device)[idx]


def rgb_to_labels(rgb: torch.Tensor) -> torch.Tensor:
    """[*, H, W, 3] in [0,1]  ->  [*, H, W] int class ids (nearest fixed color).

    Use this to decode the diffusion's generated/decoded semantic image back to classes.
    """
    d = (rgb.unsqueeze(-2) - CLASS_COLORS.to(rgb.device)).pow(2).sum(-1)   # [*, H, W, K]
    return d.argmin(-1)


@torch.no_grad()
def expand_dit_for_semantics(dit, extra: int = 16):
    """In-place: grow the DiT to ingest + predict `extra` extra latent channels (semantics).

    - patch_embedding (input Conv3d): in_dim -> in_dim+extra, new input channels ZERO
      (so the semantic input is ignored at init -> RGB path identical to pretrained).
    - head (output Linear): out_dim -> out_dim+extra, new output channels ZERO
      (semantic prediction starts at ~0, then learns). Respects the head's (x y z c)
      patch layout where channel `c` is innermost.

    Call ONCE after loading the pretrained DiT, before training. Idempotency is the
    caller's responsibility (don't call twice).
    """
    dev, dt = dit.patch_embedding.weight.device, dit.patch_embedding.weight.dtype

    # ---- input: patch_embedding Conv3d(in_dim, dim) -> Conv3d(in_dim+extra, dim) ----
    old = dit.patch_embedding
    new = torch.nn.Conv3d(old.in_channels + extra, old.out_channels,
                          kernel_size=old.kernel_size, stride=old.stride).to(dev, dt)
    new.weight.data.zero_()
    new.weight.data[:, :old.in_channels] = old.weight.data
    if old.bias is not None:
        new.bias.data = old.bias.data.clone()
    dit.patch_embedding = new

    # ---- output: head Linear(dim, out_dim*p) -> Linear(dim, (out_dim+extra)*p) ----
    # unpatchify uses 'b (f h w) (x y z c) -> b c (f x)(h y)(w z)' with c innermost,
    # so expand the per-patch channel sub-dim, not a naive row-append.
    p = int(math.prod(dit.patch_size))
    lin = dit.head.head                                   # nn.Linear(dim, out_dim*p)
    dim = lin.in_features
    old_outc = lin.out_features // p
    new_outc = old_outc + extra
    W = lin.weight.data.view(p, old_outc, dim)            # [p, out_dim, dim]
    newW = W.new_zeros(p, new_outc, dim); newW[:, :old_outc] = W
    nlin = torch.nn.Linear(dim, new_outc * p, bias=lin.bias is not None).to(dev, dt)
    nlin.weight.data = newW.reshape(new_outc * p, dim)
    if lin.bias is not None:
        b = lin.bias.data.view(p, old_outc)
        newb = b.new_zeros(p, new_outc); newb[:, :old_outc] = b
        nlin.bias.data = newb.reshape(new_outc * p)
    dit.head.head = nlin
    return dit


@torch.no_grad()
def expand_control_branch_for_semantics(control_branch, extra: int = 16):
    """In-place: grow control_branch.control_patch_embedding to accept `extra`
    extra latent input channels (semantics), inserted BETWEEN the latent channels
    (RGB + depth) and the mask/cam channels.

    Current input channel layout of control_patch_embedding:
        [RGB (16), depth (16), mask_cam (64)]  = 96
    After expansion:
        [RGB (16), depth (16), SEMANTIC (extra, zero-init), mask_cam (64)]  = 96 + extra

    New semantic channels are ZERO-INIT, so at step 0 the control branch produces
    the same hints as the pretrained model — only the new channels learn.

    Call ONCE after loading the pretrained control branch, before training.
    Idempotency is the caller's responsibility (don't call twice).
    """
    old = control_branch.control_patch_embedding
    n_latent = 32                       # RGB (16) + depth (16), pretrained order
    new_in = old.in_channels + extra    # e.g., 96 + 16 = 112

    new = torch.nn.Conv3d(
        new_in, old.out_channels,
        kernel_size=old.kernel_size, stride=old.stride,
    ).to(old.weight.device, old.weight.dtype)

    new.weight.data.zero_()
    # Copy pretrained RGB + depth weights into the first n_latent input channels
    new.weight.data[:, :n_latent] = old.weight.data[:, :n_latent]
    # Channels [n_latent : n_latent+extra] remain zero — semantic slot, learns from scratch
    # Copy pretrained mask_cam weights, shifted right by `extra` channels
    new.weight.data[:, n_latent + extra:] = old.weight.data[:, n_latent:]

    if old.bias is not None:
        new.bias.data = old.bias.data.clone()

    control_branch.control_patch_embedding = new
    return control_branch
