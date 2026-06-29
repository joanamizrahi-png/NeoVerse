# Diffusion-semantics finetune — implementation guide

Goal: finetune NeoVerse's Wan diffusion to **generate a clean 2D semantic map jointly with
RGB** (UDPDiff-style channel-concat). Freeze RGB; learn only the new semantic channels.

**Status:** core code written + committed; **untested — needs a cluster smoke-test**
(training can't run on the Jetson — reconstruction OOMs at ~115 GB; see CLAUDE.md).

## What's DONE (committed, guarded so RGB path is unchanged by default)
- `diffsynth/utils/semantics.py` — `labels_to_rgb` / `rgb_to_labels` (colorization) +
  `expand_dit_for_semantics(dit)` (grows patch-embed in 16→32 and head out 16→32, zero-init).
- `wan_video_neoverse.py`:
  - `WanVideoUnit_NoiseInitializer` — noise channels = `z_dim + pipe.semantic_channels`.
  - `WanVideoUnit_InputVideoEmbedder` — if `pipe.semantic_channels>0` and `semantic_labels`
    given: colorize → VAE-encode → **concat into `input_latents`** (the generation target).
- All guarded by `getattr(pipe, "semantic_channels", 0)` (default 0 → inert).

## TODO (do on the cluster, with testing)

**1. Dataloader** — `training/data/datasets/spatialvid.py`: load the SAM3 `.npz` for the clip
and return `semantic_labels` [T,H,W] int aligned to the frames. (Run `sam3_precompute_labels.py`
on each training clip first; key the `.npz` by clip name.) Make sure it flows into the
`input_video`/data dict the pipeline units consume.

**2. `train.py`** — turn it on + freeze:
```python
from diffsynth.utils.semantics import expand_dit_for_semantics
SEM_CH = 16
self.pipe.semantic_channels = SEM_CH          # activates the guarded units
expand_dit_for_semantics(self.pipe.dit, extra=SEM_CH)   # ONCE, after load, before training
# freeze everything, then unfreeze only the new channels (+ a small LoRA on dit)
for p in self.pipe.dit.parameters(): p.requires_grad_(False)
self.pipe.dit.patch_embedding.requires_grad_(True)   # new input channels
self.pipe.dit.head.head.requires_grad_(True)         # new output channels
# + add_lora_to_model(..., lora_target_modules=q,k,v,o,ffn...) as the existing path does
```
Note: zero-init means at step 0 the model == pretrained RGB model; loss on the RGB channels
should match baseline, and only the semantic channels move.

**3. Loss balance (optional refinement)** — `training_loss` currently MSEs all 32 channels
equally. If semantics dominates/lags, split: `loss = mse(pred[:,:16], tgt[:,:16]) + λ*mse(pred[:,16:], tgt[:,16:])`.

**4. Inference decode** — in `inference.py`, when `semantic_channels>0`: after generation,
split the 32-ch output, VAE-decode the semantic half → RGB image → `rgb_to_labels()` → class map.
(The diffusion-generated semantics REPLACE the rasterizer's `feature="labels"` path; that holey
3D render can optionally become the conditioning, see #6.)

**5. Smoke-test (do this FIRST on the cluster)** — overfit ONE clip:
- set `semantic_channels=16`, point the dataloader at a single labeled clip, train ~200 steps.
- expect: total loss drops; decode the semantic half of the output → recognizable classes.
- This validates the wiring end-to-end before any real training run.

**6. (v2) Condition on the rough 3D semantic** — feed our existing `feature="labels"` render
into the control branch alongside depth: `wan_video_neoverse_controller.py` line ~91
`torch.cat((target_rgb_latents, target_depth_latents, target_semantic_latents), dim=1)`,
`control_in_dim 96→112`, `mask_cam_out_dim = control_in_dim - 48`, control_patch_embedding
`Conv3d(112, dim)` zero-init new channels. Makes it an inpainting task (rough→clean), reusing 1a.

## Quick sanity checks before training
- `import` works: `from diffsynth.utils.semantics import expand_dit_for_semantics, labels_to_rgb`
- CLASS_COLORS order in `semantics.py` matches `sam3_precompute_labels.CLASSES` (13 + bg).
- After `expand_dit_for_semantics`: `dit.patch_embedding.in_channels == 32`,
  `dit.head.head.out_features == 32 * prod(patch_size)`.
- A normal RGB run (no `semantic_channels` set) behaves exactly as before (guards default to 0).
