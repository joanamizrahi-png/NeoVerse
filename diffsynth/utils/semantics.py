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

# Class index -> RGB color for the 29-class outdoor Go2W taxonomy (index 0 = void).
# Palette designed for VAE roundtrip robustness (min pairwise distance ~35 units; UDPDiff-style).
# Aux CE loss during finetuning is expected to enforce class boundaries for the closer pairs
# (see docs/FINETUNE_IMPLEMENTATION.md).
# LOCKSTEP: this order MUST match sam3_precompute_labels.CLASSES and nav-rl reward.TRAVERSABLE.
CLASS_COLORS = torch.tensor([
    [  0,   0,   0],   # 0  void / unlabeled
    [200, 225, 245],   # 1  sky
    [139,  90,  43],   # 2  dirt
    [230, 200, 155],   # 3  sand
    [ 75, 190,  80],   # 4  grass
    [180, 155, 100],   # 5  gravel
    [110,  55,  25],   # 6  mulch
    [ 55,  55,  30],   # 7  mud
    [ 50, 120, 200],   # 8  water
    [135, 145, 155],   # 9  rock
    [ 55,  55,  65],   # 10 asphalt
    [225, 220, 190],   # 11 concrete
    [110, 110, 115],   # 12 road
    [180, 180, 180],   # 13 sidewalk
    [255, 250, 235],   # 14 crosswalk
    [170,  75,  60],   # 15 building
    [175, 145, 175],   # 16 wall
    [ 90,  60, 130],   # 17 fence
    [ 75, 155, 175],   # 18 bridge
    [ 40, 105,  55],   # 19 tree
    [170, 200,  55],   # 20 vegetation
    [135, 115,  90],   # 21 log
    [220, 140,  80],   # 22 stairs
    [ 25,  65, 130],   # 23 pole
    [230, 195,  60],   # 24 traffic sign
    [235,  85,  75],   # 25 traffic light
    [110, 130, 220],   # 26 vehicle
    [155,  60, 200],   # 27 motorcycle
    [100, 230, 200],   # 28 bicycle
    [205,  70, 145],   # 29 person
], dtype=torch.float32) / 255.0          # [K, 3], values in [0, 1]
NUM_CLASSES = CLASS_COLORS.shape[0]


# v14: the active palette is selectable once at startup (train.py /
# inference_semantic.py call set_active_palette with their num_semantic_classes)
# so every colorize/decode site follows without threading a parameter through.
_ACTIVE_PALETTE = CLASS_COLORS


def set_active_palette(num_classes: int, version: int = 1):
    """Switch colorize/decode to the palette for `num_classes`. 30 = legacy
    CLASS_COLORS (default); 14 = the v14 navigation taxonomy. `version`
    selects the v14 color set (2026-08-29): checkpoints must be trained AND
    decoded with the same version — v21 and earlier are version 1."""
    global _ACTIVE_PALETTE
    if num_classes == NUM_CLASSES:
        _ACTIVE_PALETTE = CLASS_COLORS
    else:
        from .class_taxonomy import v14_palette, NUM_CLASSES_V14
        assert num_classes == NUM_CLASSES_V14, \
            f"no palette defined for num_classes={num_classes}"
        _ACTIVE_PALETTE = v14_palette(version=version)
    return _ACTIVE_PALETTE


def get_active_palette() -> torch.Tensor:
    """The palette currently selected by set_active_palette (colorize AND
    any artifact-saving code must use this, never CLASS_COLORS directly)."""
    return _ACTIVE_PALETTE


def labels_to_rgb(labels: torch.Tensor) -> torch.Tensor:
    """[*, H, W] int class ids  ->  [*, H, W, 3] float in [0,1] (colorized image)."""
    pal = _ACTIVE_PALETTE
    idx = labels.long().clamp(0, pal.shape[0] - 1)
    return pal.to(labels.device)[idx]


def rgb_to_labels(rgb: torch.Tensor) -> torch.Tensor:
    """[*, H, W, 3] in [0,1]  ->  [*, H, W] int class ids (nearest fixed color).

    Use this to decode the diffusion's generated/decoded semantic image back to classes.
    """
    d = (rgb.unsqueeze(-2) - _ACTIVE_PALETTE.to(rgb.device)).pow(2).sum(-1)   # [*, H, W, K]
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


# ---------------------------------------------------------------------------
# v2 expansion: parallel `_sem` submodules for true RGB/semantic decoupling.
#
# Motivation: v5's approach (grow the module + rank-32 LoRA on top) meant the
# semantic pathway was rank-limited AND the RGB slice of head/patch_embed could
# drift via the LoRA delta. Semantic output ended up as coarse blobs, not sharp
# class regions.
#
# v2 wraps the pretrained patch_embedding / head / control_patch_embedding in
# a Module that runs the pretrained ("base") layer on the RGB slice of the
# input and a NEW zero-init ("sem") layer of the same shape on the semantic
# slice. Their outputs are summed for the input side, or interleaved into
# channel-innermost layout for the head side. Semantic channels get FULL-RANK
# 16 -> dim (and dim -> 16*p) capacity, and the RGB path stays bit-identical
# to the pretrained weights unless someone unfreezes it.
#
# What to freeze in the v6 config:
#   trainable: dit.patch_embedding.sem, dit.head.head.sem,
#              control_branch.control_patch_embedding.sem
#   frozen (plus a small LoRA on q/k/v/o/ffn for semantic routing capacity):
#              everything else
# ---------------------------------------------------------------------------


class SplitPatchEmbedding(torch.nn.Module):
    """Wraps the pretrained input Conv3d + a zero-init parallel Conv3d for semantics.

    Forward takes a (B, base_ch + sem_ch, T, H, W) latent, feeds the first
    `base_ch` channels through the base layer, feeds the next `sem_ch` channels
    through the sem layer, and returns the sum. At init the sem branch is zero,
    so hidden-state == base_layer(rgb_slice) — bit-identical to the pretrained
    forward on the pretrained-only input.
    """
    def __init__(self, base: torch.nn.Conv3d, sem: torch.nn.Conv3d):
        super().__init__()
        self.base = base
        self.sem = sem
        self.base_ch = base.in_channels
        self.sem_ch = sem.in_channels

    def forward(self, x):
        base_out = self.base(x[:, :self.base_ch])
        if getattr(self, "vanilla_mode", False):
            return base_out
        sem_out = self.sem(x[:, self.base_ch:self.base_ch + self.sem_ch])
        return base_out + sem_out

    # Expose Conv3d-like attrs that other code inspects.
    @property
    def weight(self):
        return self.base.weight

    @property
    def in_channels(self):
        return self.base_ch + self.sem_ch

    @property
    def out_channels(self):
        return self.base.out_channels

    @property
    def kernel_size(self):
        return self.base.kernel_size

    @property
    def stride(self):
        return self.base.stride


class SplitHead(torch.nn.Module):
    """Wraps the pretrained output Linear + a zero-init parallel Linear for semantics.

    The DiT head's flat output layout is (patch_pos, channel) with channel
    INNERMOST (unpatchify reshapes as `b (f h w) (x y z c)` -> `b c ...`).
    We reshape each branch's output to [B, N, p, ch], concat on the channel
    axis so base fills c=0..base_ch-1 and sem fills c=base_ch..base_ch+sem_ch-1,
    then flatten back. At init sem is zero => RGB rows bit-identical to base
    output.
    """
    def __init__(self, base: torch.nn.Linear, sem: torch.nn.Linear, patch_size_prod: int):
        super().__init__()
        self.base = base
        self.sem = sem
        self.p = patch_size_prod
        self.base_ch = base.out_features // patch_size_prod
        self.sem_ch = sem.out_features // patch_size_prod
        assert self.base_ch * patch_size_prod == base.out_features, \
            f"base head out_features {base.out_features} not divisible by p={patch_size_prod}"
        assert self.sem_ch * patch_size_prod == sem.out_features, \
            f"sem head out_features {sem.out_features} not divisible by p={patch_size_prod}"

    def forward(self, x):
        # x: [B, N, dim]
        base_flat = self.base(x)                                     # [B, N, base_ch * p]
        sem_flat = self.sem(x)                                       # [B, N, sem_ch * p]
        # v27 (2026-09-01): optional DINO hint, added to the SEMANTIC branch
        # ONLY (attach_dino_sem_head + model_fn stash it here). The v25 wiring
        # put it in the shared control embedding and destroyed RGB; base_flat
        # above never sees it, so the RGB rows are untouched by construction.
        # Consume-and-clear, like the control-branch variant.
        feats = getattr(self, "_dino_feats", None)
        if feats is not None:
            self._dino_feats = None
            d = self.dino_proj(feats.to(sem_flat.dtype))             # [B,C,f,h,w]
            d = d.flatten(2).transpose(1, 2)                         # [B, N, C]
            if d.shape[1] == sem_flat.shape[1]:
                sem_flat = sem_flat + d
            else:
                print(f"[dino_sem_head] token mismatch {d.shape[1]} vs "
                      f"{sem_flat.shape[1]}; hint skipped", flush=True)
        B, N, _ = base_flat.shape
        base_r = base_flat.view(B, N, self.p, self.base_ch)          # [B, N, p, base_ch]
        sem_r = sem_flat.view(B, N, self.p, self.sem_ch)             # [B, N, p, sem_ch]
        combined = torch.cat([base_r, sem_r], dim=-1)                # [B, N, p, base_ch+sem_ch]
        return combined.view(B, N, self.p * (self.base_ch + self.sem_ch))

    # Expose Linear-like attrs.
    @property
    def weight(self):
        return self.base.weight

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return (self.base_ch + self.sem_ch) * self.p


class SplitControlPatchEmbedding(torch.nn.Module):
    """Wraps control_branch.control_patch_embedding for v6.

    Input channel layout (same as v5's expanded form):
        [RGB (16), depth (16), SEMANTIC (sem_ch), mask_cam (64)] = 96 + sem_ch
    Base layer processes the 96 non-semantic channels: RGB, depth, mask_cam.
    Sem layer processes just the SEMANTIC slice (sem_ch channels).
    Outputs summed.

    Base is the pretrained Conv3d(96, dim); sem is a new zero-init Conv3d(sem_ch, dim).
    RGB hint at init == pretrained control_branch output.
    """
    def __init__(self, base: torch.nn.Conv3d, sem: torch.nn.Conv3d,
                 sem_start: int = 32, sem_ch: int = 16):
        super().__init__()
        self.base = base
        self.sem = sem
        self.sem_start = sem_start
        self.sem_ch = sem_ch

    def forward(self, x):
        # x: [B, 96+sem_ch, T, H, W]. Slice out semantic; concat the rest for base.
        sem_x = x[:, self.sem_start:self.sem_start + self.sem_ch]
        base_x = torch.cat(
            [x[:, :self.sem_start], x[:, self.sem_start + self.sem_ch:]],
            dim=1,
        )
        if getattr(self, "vanilla_mode", False):
            return self.base(base_x)
        out = self.base(base_x) + self.sem(sem_x)
        # v25 DINO hint: inert unless attach_dino_hint() installed dino_proj
        # AND model_fn stashed features for this forward. Consumed once so a
        # forward without a fresh stash can never reuse stale features.
        dino_proj = getattr(self, "dino_proj", None)
        dino_feats = getattr(self, "_dino_feats", None)
        if dino_proj is not None and dino_feats is not None:
            out = out + dino_proj(dino_feats.to(
                device=dino_proj.weight.device, dtype=dino_proj.weight.dtype))
            self._dino_feats = None
        return out

    @property
    def weight(self):
        return self.base.weight

    @property
    def in_channels(self):
        return self.base.in_channels + self.sem_ch

    @property
    def out_channels(self):
        return self.base.out_channels

    @property
    def kernel_size(self):
        return self.base.kernel_size

    @property
    def stride(self):
        return self.base.stride


@torch.no_grad()
def expand_dit_for_semantics_v2(dit, extra: int = 16):
    """v6 expansion. Adds parallel zero-init `_sem` submodules on patch_embedding
    and head.head. Pretrained modules are left as-is; RGB path is bit-identical
    to pretrained until someone unfreezes it.

    After this call:
        dit.patch_embedding      -- SplitPatchEmbedding(base=Conv3d(16,dim), sem=Conv3d(extra,dim))
                                    trainable path: dit.patch_embedding.sem
        dit.head.head            -- SplitHead(base=Linear(dim,16*p),
                                              sem=Linear(dim,extra*p))
                                    trainable path: dit.head.head.sem

    Call ONCE after loading pretrained DiT, before freeze/LoRA setup.
    """
    dev, dt = dit.patch_embedding.weight.device, dit.patch_embedding.weight.dtype

    base_pe = dit.patch_embedding
    sem_pe = torch.nn.Conv3d(
        extra, base_pe.out_channels,
        kernel_size=base_pe.kernel_size, stride=base_pe.stride,
    ).to(dev, dt)
    sem_pe.weight.data.zero_()
    if sem_pe.bias is not None:
        sem_pe.bias.data.zero_()
    dit.patch_embedding = SplitPatchEmbedding(base=base_pe, sem=sem_pe)

    p = int(math.prod(dit.patch_size))
    base_head = dit.head.head
    dim = base_head.in_features
    sem_head = torch.nn.Linear(dim, extra * p, bias=base_head.bias is not None).to(dev, dt)
    sem_head.weight.data.zero_()
    if sem_head.bias is not None:
        sem_head.bias.data.zero_()
    dit.head.head = SplitHead(base=base_head, sem=sem_head, patch_size_prod=p)
    return dit


@torch.no_grad()
def expand_control_branch_for_semantics_v2(control_branch, extra: int = 16):
    """v6 expansion for control_branch. Adds a parallel zero-init Conv3d that
    consumes just the semantic slice of the input tensor. Base Conv3d(96, dim)
    is unchanged and continues to process RGB+depth+mask_cam.

    Input tensor layout is unchanged from v5:
        [RGB (16), depth (16), SEMANTIC (extra), mask_cam (64)] = 96 + extra

    trainable path: control_branch.control_patch_embedding.sem
    """
    base_cpe = control_branch.control_patch_embedding
    sem_cpe = torch.nn.Conv3d(
        extra, base_cpe.out_channels,
        kernel_size=base_cpe.kernel_size, stride=base_cpe.stride,
    ).to(base_cpe.weight.device, base_cpe.weight.dtype)
    sem_cpe.weight.data.zero_()
    if sem_cpe.bias is not None:
        sem_cpe.bias.data.zero_()
    control_branch.control_patch_embedding = SplitControlPatchEmbedding(
        base=base_cpe, sem=sem_cpe, sem_start=32, sem_ch=extra,
    )
    return control_branch


class SemanticClassHead(torch.nn.Module):
    """v8 Change 2: reads per-pixel class logits off the VAE-DECODED semantic
    output (the colorized map, full resolution). Used two ways:
      - training: decoded-space cross-entropy — the loss that finally asks
        "is this pixel the right class?" at the pixel scale where speckle lives
      - inference (optional): replaces nearest-palette decoding

    Class-count agnostic: num_classes is a constructor arg; checkpoints carry
    it in the final conv's shape (see inference auto-instantiation).
    """

    def __init__(self, num_classes: int = 30, hidden: int = 64, depth: int = 2):
        super().__init__()
        # depth = number of 3x3 convs before the final 1x1. depth<=2 keeps the
        # historical all-dilation-1 stack (v8..v10 checkpoints load exactly);
        # depth>2 grows dilation 1,2,4,... so the receptive field covers a
        # region rather than a ~5px neighborhood — context to disambiguate
        # soft color blends (the v10 failure mode of palette-snap).
        layers, in_ch = [], 3
        for i in range(depth):
            dil = 1 if depth <= 2 else 2 ** min(i, 4)
            layers += [torch.nn.Conv2d(in_ch, hidden, 3, padding=dil, dilation=dil),
                       torch.nn.ReLU()]
            in_ch = hidden
        layers += [torch.nn.Conv2d(hidden, num_classes, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        # x: [N, 3, H, W] decoded colorized frames in [-1, 1]; cast to our own
        # param dtype (the head may be fp32 while the pipe runs bf16).
        return self.net(x.to(next(self.parameters()).dtype))


# ---------------------------------------------------------------------------
# Vanilla-RGB reference mode (for the RGB-preservation loss).
#
# With the v2 expansion the RGB output can drift from the pretrained function
# through exactly two doors: the trunk LoRA, and the `_sem` branches of the two
# input-side Split wrappers (their output is SUMMED into the shared token
# stream). SplitHead's RGB rows are a concat, structurally untouched by its sem
# branch. So disabling those two doors reproduces the pretrained RGB function
# bit-exactly, on the same conditioning, with no second model in memory.
# ---------------------------------------------------------------------------
from contextlib import contextmanager


@contextmanager
def vanilla_rgb_reference(*module_roots):
    """Temporarily compute the exact pretrained (vanilla) RGB function.

    Disables peft LoRA adapters and the `_sem` input branches on every module
    under the given roots (pass the dit and, if used, the control_branch).
    Restores everything on exit. Use under torch.no_grad().
    """
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:
        BaseTunerLayer = None
    toggled_lora, toggled_split = [], []
    for root in module_roots:
        if root is None:
            continue
        for m in root.modules():
            if BaseTunerLayer is not None and isinstance(m, BaseTunerLayer):
                if hasattr(m, "enable_adapters"):
                    m.enable_adapters(False)
                    toggled_lora.append(m)
            elif isinstance(m, (SplitPatchEmbedding, SplitControlPatchEmbedding)):
                m.vanilla_mode = True
                toggled_split.append(m)
    try:
        yield
    finally:
        for m in toggled_lora:
            m.enable_adapters(True)
        for m in toggled_split:
            m.vanilla_mode = False


# ---------------------------------------------------------------------------
# Track B (2026-08-15): ANALOG-BITS semantic encoding.
#
# The colorize->VAE path represents classes as blendable colors — a soft
# prediction can land between two classes and decode as a THIRD (the v10
# palette-snap failure). Analog bits kill that at the root: each class id is
# 4 binary digits carried in 4 dedicated latent channels as +/-1 values,
# produced DIRECTLY at latent resolution (no VAE on the semantic slot). A
# soft bit still rounds to 0 or 1; codes cannot blend into other codes.
#
# Resolution contract (matches Wan's video VAE latent grid): spatial /8,
# temporal 1 + (T-1)/4 (causal: frame 0, then stride-4 groups). Spatial
# downsampling is MAJORITY-VOTE per 8x8 block (mode pooling), not nearest —
# thin structures lose either way, but mode is stable and unbiased.
# ---------------------------------------------------------------------------
SEMANTIC_BITS = 4          # 2^4 = 16 >= 14 classes


def _mode_pool2d(labels: torch.Tensor, stride: int, num_classes: int) -> torch.Tensor:
    """[T, H, W] int -> [T, H/s, W/s] int by per-block majority vote."""
    onehot = torch.nn.functional.one_hot(labels.long(), num_classes)  # [T,H,W,K]
    onehot = onehot.permute(0, 3, 1, 2).float()                       # [T,K,H,W]
    pooled = torch.nn.functional.avg_pool2d(onehot, stride)           # [T,K,h,w]
    return pooled.argmax(dim=1)                                       # [T,h,w]


def labels_to_analog_bits(labels: torch.Tensor, num_classes: int,
                          t_causal: bool = True, s_stride: int = 8) -> torch.Tensor:
    """[T, H, W] int class ids -> [SEMANTIC_BITS, T', H/8, W/8] float in {-1,+1}.

    Channel-first to concat with the RGB latent [16, T', h, w] along dim 0.
    """
    if t_causal:
        idx = [0] + list(range(1, labels.shape[0], 4))                # causal-VAE frame map
        labels = labels[idx]
    small = _mode_pool2d(labels, s_stride, num_classes)               # [T',h,w]
    bits = ((small.unsqueeze(0) >> torch.arange(
        SEMANTIC_BITS, device=labels.device).view(-1, 1, 1, 1)) & 1)  # [B,T',h,w]
    return bits.float() * 2.0 - 1.0


def analog_bits_to_labels(bits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """[SEMANTIC_BITS, T', h, w] float -> [T', h, w] int ids (threshold at 0)."""
    hard = (bits > 0).long()
    ids = torch.zeros(bits.shape[1:], dtype=torch.long, device=bits.device)
    for b in range(SEMANTIC_BITS):
        ids += hard[b] << b
    return ids.clamp(0, num_classes - 1)
