# Project: Semantic 4D World Model → RL Navigation (MA thesis, targeting ICRA)

This is a fork of NeoVerse, extended for a thesis project. This file is the project
context — read it first.

## The goal (one sentence)

Build a **semantic 4D world model** from real monocular video and use it as a
**simulator to train an outdoor legged-robot navigation policy** (RL), with a reward
based on **semantic traversability** (is the robot's next foot-placement on
walkable terrain?). Deploy on a real robot (Unitree Go2 / Gitamini). Real-world
transfer is the key novelty.

## ⭐ CURRENT STATUS (2026-07-01) — read this first

**Design (unchanged, INPAINTING, not co-denoising from scratch):**
SAM3/RUGD labels → 3D-fuse onto Gaussians → render at a view → **holey** semantic → the
diffusion **fills the holes** → clean semantic. SAM3 runs once per scene (at reconstruction),
not per RL step. Consistent with how NeoVerse already inpaints rough RGB → clean RGB.
- Training example = `(rough RGB + rough depth + rough HOLEY semantic + clean RGB target + clean semantic target)`.
- Label source (agreed): **SAM3-on-input-frames** = the input hint (attached to Gaussians →
  reconstructor renders the holey view). **RUGD ground-truth masks** = the CLEAN training target.
  This means during deployment we only need SAM3, but training uses RUGD's cleaner labels.

### DONE THIS SESSION (Mac side, all guarded — RGB-only runs unchanged)

The full diffusion-side wiring for semantic finetune is now in place. Everything is guarded
by `pipe.semantic_channels > 0` and defaults to inert.

**1. Model surgery** (`diffsynth/utils/semantics.py`):
- `labels_to_rgb` / `rgb_to_labels` — the colorize/decolorize trick (was already there).
- `expand_dit_for_semantics(dit, extra=16)` — grows DiT input patch-embed + output head from
  16→32 latent channels. Zero-init new channels; pretrained RGB behavior unchanged at step 0.
- `expand_control_branch_for_semantics(control_branch, extra=16)` — **NEW**. Grows
  `control_patch_embedding` from Conv3d(96, dim) → Conv3d(112, dim). Inserts new channels
  **between** the latent block (RGB+depth, positions 0-31) and the mask_cam block
  (shifted 32-95 → 48-111). Zero-init new channels.

**2. Control branch** (`diffsynth/models/wan_video_neoverse_controller.py`):
- `NeoVerseControlBranch.forward()` gained `target_semantic_latents=None` optional kwarg
  positioned between `freqs` and `use_gradient_checkpointing`. When provided, `target_latents`
  becomes a 3-way cat (RGB + depth + semantic).

**3. Pipeline plumbing** (`diffsynth/pipelines/wan_video_neoverse.py`):
- `WanVideoUnit_4DPreprocesser.process()` runs a second rasterizer pass with `feature="labels"`
  at the target trajectory when `pipe.semantic_channels > 0` AND `source_views["labels"]` is
  present. Argmaxes to `[B,T,H,W]` class-id tensor, timestamp-sorted. Emitted as `target_semantic`.
- `WanVideoUnit_4DEmbedder` now accepts `target_semantic` (class ids), colorizes → permutes →
  `preprocess_video` → VAE-encode → `target_semantic_latents`. Returns `None` on RGB-only.
- `model_fn_wan_video` gained `target_semantic=None`. **CRITICAL**: converted the
  `control_branch(...)` call to keyword args because the controller's new positional param
  would silently misalign old-style positional calls.
- `WanVideoUnit_InputVideoEmbedder` uses `semantic_labels` (**the clean target**) to build the
  32-ch `input_latents`. This was already in place from the earlier session.

**4. Dataloader** (`training/data/datasets/spatialvid.py`):
- Constructor takes `labels_dir=None` (path to per-clip SAM3 `.npz`).
- If set, loads `<scene_id>.npz` and indexes by the sampled frame indices → per-view `labels`
  attached to each view dict. `compose_batches_from_list` auto-batches new tensor fields.
- Refactored the context/target loop to a single dict (functional-identical to before).

**5. Training entry** (`train.py`):
- `WanTrainingModule.__init__(semantic_channels=0)`. If > 0, calls
  `expand_dit_for_semantics` + `expand_control_branch_for_semantics` AFTER checkpoint load,
  BEFORE freeze/LoRA setup.
- Config plumb: `semantic_channels=int(getattr(args, "semantic_channels", 0))` in `__main__`.

### End-to-end flow when `semantic_channels: 16` is set in the config

```
dataloader (SpatialVID with labels_dir)
    → view dicts contain "labels" (SAM3 int class ids per frame)
train.py forward_preprocess
    → inputs_shared["source_views"] = data
WanVideoUnit_4DPreprocesser
    → reconstructor(source_views with labels) → Gaussians with labels attached
    → rasterizer.forward(feature="rgb")    → target_rgb
    → rasterizer.forward(feature="depth")  → target_depth  (via existing depth path)
    → rasterizer.forward(feature="labels") → argmax → target_semantic  (holey)
WanVideoUnit_4DEmbedder
    → VAE-encodes target_rgb → target_rgb_latents
    → VAE-encodes target_depth → target_depth_latents
    → colorize + VAE-encodes target_semantic → target_semantic_latents
WanVideoUnit_InputVideoEmbedder
    → VAE-encodes input_video → 16-ch input_latents
    → colorize + VAE-encodes semantic_labels (CLEAN target) → concat → 32-ch input_latents
model_fn_wan_video → control_branch(target_semantic_latents=…) → hints
    → DiT (expanded 16→32 in/out) → 32-ch noise_pred
training_loss → MSE(noise_pred, training_target from 32-ch input_latents)
```

### STILL TODO (the last-mile leak)

**`semantic_labels` (the clean training target) is not populated by the dataloader yet.**
`WanVideoUnit_InputVideoEmbedder` reads it but it's currently always None. Two options for
the pilot smoke-test:

- **Option A (self-supervised sanity)**: set `semantic_labels = per_frame_labels` from the
  dataloader (same SAM3 labels used for the input hint). Trains the diffusion to reproduce
  its own SAM3 labels. Validates plumbing, doesn't validate the "clean-up" goal.
- **Option B (real training)**: add a `target_labels_dir` config that points to RUGD
  ground-truth masks. Wire a second load in the dataloader; `forward_preprocess` puts them
  into `inputs_shared["semantic_labels"]`.

Recommended: do Option A first (5-line change, unblocks the cluster smoke-test), then
Option B once RUGD annotations are unpacked and clip↔mask pairing is confirmed.

**Other TODOs still valid from the previous session's `docs/FINETUNE_IMPLEMENTATION.md`:**
- Freezing pattern in `train.py` (currently uses the existing `trainable_models` flag — needs
  a cluster config that unfreezes only `patch_embedding` + `head` + a small LoRA).
- Inference decode: split 32-ch DiT output, VAE-decode semantic half, `rgb_to_labels` →
  class map. Not needed for training, but needed for eval / RL rollouts.
- Smoke-test: overfit ONE clip on the cluster; verify total loss drops + decoded semantic
  half looks recognizable.

### Data status
- RUGD download to `~/joana/rugd_full/` may still be in flight. `ls ~/joana/rugd_full/` to check.
- Once complete: run `scripts/prepare_rugd_clips.py --rugd_root ~/joana/rugd_full` to chunk
  to 81-frame MP4s. Then `scripts/run_sam3_on_pilot.sh` batches SAM3 over all clips.
- After that: point the training config at `labels_dir=outputs/sam3_labels` and RUGD
  ground-truth mask dir when we add Option B.

See `docs/FINETUNE_IMPLEMENTATION.md` for the original design notes; this section supersedes
the older TODO list there.

## How NeoVerse works (3 layers)

1. **Reconstruct (3D):** feed-forward WorldMirror/VGGT-based reconstructor →
   4D Gaussian Splats from a short monocular video. Code:
   `diffsynth/auxiliary_models/worldmirror/models/models/rasterization.py`.
2. **Render (3D→2D):** rasterize Gaussians from a target viewpoint → rough RGB + depth
   (+ now semantics). Rasterizer.forward + GaussianSplatRenderer in the same file.
3. **Diffusion (2D):** Wan 2.1 14B video diffusion (+4-step distilled LoRA) polishes the
   rough 2D render into clean RGB and inpaints occlusion holes. Pipeline:
   `diffsynth/pipelines/wan_video_neoverse.py`. Inference entry: `inference.py`.

Key fact: RGB *color* lives in 3D (on the Gaussians), but the *polished* RGB is 2D-only
(diffusion output, not written back to 3D). The diffusion exists because making the
reconstructor produce hole-free photorealistic 3D is too hard — holes/artifacts at novel
views are inherent; the 2D diffusion cleans them.

## What's already built (semantic wiring — DONE, works end-to-end)

We attach SAM3 semantic labels to the 3D Gaussians and render a view-consistent 2D
semantic image alongside RGB+depth. Verified: rendered semantic class distribution
matches the SAM3 input (road/building/car/tree all correct, view-consistent).

- `sam3_precompute_labels.py` — runs SAM3 on a clip's frames (using the SAME `load_video`
  as inference, so labels align 1:1) → `outputs/sam3_labels/<clip>.npz`
  (labels [N,H,W] int8 + class metadata). `--prompts "road,car,..."` to set classes.
- `inference.py` — `--semantic_labels <npz>` (auto-loads `outputs/sam3_labels/<stem>.npz`);
  adds `views["labels"]`, runs a 2nd rasterizer pass `forward(..., feature="labels")`,
  argmaxes → saves `target_semantic.mp4`. All label code guarded by `if "labels"` so the
  no-label path is unchanged.
- `rasterization.py` — threads `labels` through: `Gaussians.__init__` + `transition_labels`,
  `apply_confidence_filter` mask_keys, `_create_constant/dynamic_gaussians`, `prepare_splats`
  (one-hot `splats["labels"]`), `rasterize_splats` (colors=labels, render_mode "RGB", argmax).
- `freeze_camera_demo.py` — proves camera/time are independent dials (fixed pose, advancing
  timestamp = static camera, moving cars). Loads reconstructor only.

This SAM3-fuse pipeline = the "preliminary" / Track-A path. It gives 3D-consistent but
**holey** semantics (no Gaussian in unseen regions → no label).

## The plan (decided with advisor Jing)

**Track A — Semantics (NEW direction): finetune the diffusion to output clean 2D
semantics jointly with RGB** (no text prompt; fixed class set; hole-free because the
diffusion inpaints). Chosen over 3D-fuse because the RL policy only ever consumes 2D
images, so 3D-complete semantics aren't needed.

- **Recipe (validated by UDPDiff / IDC-Net / Stable-Part-Diffusion-4D):**
  colorize the semantic mask → encode through the SHARED VAE → **channel-concat** with the
  RGB latent → finetune the DiT to **co-denoise** both → decode. FREEZE RGB, train mostly
  the new semantic channels.
- **CRITICAL gotcha:** discrete class masks are NOT image-like — you must **colorize**
  them (class→RGB color, "spatial color encoding") before VAE-encoding; naive latent concat
  fails. May need a small separate decoder for the semantic channel (GeoVideo does; IDC-Net
  shows shared decoder works with the colormap trick).
- **Data:** SAM3 pseudo-labels (validated for off-road traversability by **V-STRONG**), or
  pre-labeled RELLIS-3D / RUGD. Start with one, scale with the other.
- Existing training code to adapt: `train.py`, `training/data/datasets/spatialvid.py`
  (dataloader returns frames+caption → add colorized semantic maps), `training/configs/train.yaml`,
  ZeRO-2 config. Runs with Accelerate + DeepSpeed.

**Track B — Policy (RL):** world model = simulator. Input = a few key frames (1-2s) +
goal point. Stage 1 STATIC env (5-10 waypoints, 0.5-1m apart), reward = traversable-or-not
(use the SAM3-per-frame pipeline preliminarily), no collision-avoidance yet, output one-step
actions or future trajectories. Stage 2 DYNAMIC env (shorter horizon). Fine-tune a
pretrained walker with FEW rollouts (World4RL/World-Gymnast style — diffusion too slow for
from-scratch RL). See `docs/SEMANTIC_RL_RESEARCH.md` for the full cited research report.

## Hardware reality — IMPORTANT

- Dev was on a **Jetson AGX Thor** (aarch64, 122 GB unified memory). The semantic wiring
  was built and validated there.
- **The Jetson CANNOT reliably run NeoVerse reconstruction**: it peaks ~115-125 GB during
  reconstruction (fixed cost, does NOT scale with frame count) and hard-locks the board at
  the 122 GB ceiling. Reconstruction/diffusion/training MUST run on a **CLUSTER**.
- The Jetson is the **deployment** target (runs the trained policy + fast perception), not
  the training rig.
- Mac is for **writing code** (the finetuning script) — not running NeoVerse.

## Immediate next steps

1. SAM3-label a training video set (RELLIS/RUGD for clean labels + own clips) — recipe-independent.
2. Write the finetuning edits: colorized-semantic channel + loss into the Wan diffusion +
   `train.py` dataloader. (UDPDiff-style; IDC-Net = minimal reference.)
3. Train on cluster. Then wire the finetuned semantic output into the policy reward.

## Notes
- `jetson_patches/decord_shim.py` — a decord replacement (real decord has no aarch64 wheel /
  won't build vs ffmpeg 6); only needed on the Jetson env, not Mac/cluster.
- Env on Jetson: conda env `neoverse`, py3.12, torch 2.9.1 cu130 from jetson-ai-lab. For
  Mac/cluster, set up a fresh env per upstream README (CUDA 12.x).
