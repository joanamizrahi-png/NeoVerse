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

# Class index -> RGB color. Index 0 = unlabeled/background. MUST match the class order
# in sam3_precompute_labels.CLASSES (13 classes + background).
CLASS_COLORS = torch.tensor([
    [0,   0,   0],     # 0 unlabeled
    [128, 128, 128],   # 1 road
    [210, 180, 140],   # 2 sidewalk
    [0,   180, 0],     # 3 grass
    [139, 90,  43],    # 4 path
    [30,  80,  220],   # 5 water
    [180, 100, 30],    # 6 stairs
    [140, 70,  20],    # 7 building
    [100, 60,  100],   # 8 fence
    [34,  139, 34],    # 9 vegetation
    [0,   0,   255],   # 10 car
    [255, 165, 0],     # 11 bicycle
    [255, 0,   0],     # 12 person
    [135, 206, 235],   # 13 sky
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
