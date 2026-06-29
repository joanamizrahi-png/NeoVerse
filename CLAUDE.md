# Project: Semantic 4D World Model → RL Navigation (MA thesis, targeting ICRA)

This is a fork of NeoVerse, extended for a thesis project. This file is the project
context — read it first.

## The goal (one sentence)

Build a **semantic 4D world model** from real monocular video and use it as a
**simulator to train an outdoor legged-robot navigation policy** (RL), with a reward
based on **semantic traversability** (is the robot's next foot-placement on
walkable terrain?). Deploy on a real robot (Unitree Go2 / Gitamini). Real-world
transfer is the key novelty.

## ⭐ CURRENT STATUS (2026-06-29) — read this first

**Corrected design (the diffusion INPAINTS, it does NOT replace SAM3):**
SAM3/RUGD labels → 3D-fuse onto Gaussians → render at a view → **holey** semantic → the
diffusion **fills the holes** → clean semantic. SAM3 runs once per scene (at reconstruction),
not per RL step. This is consistent with how NeoVerse already inpaints rough RGB → clean RGB.
- Training example = `(holey rendered semantic + rough RGB)` → `(clean semantic)`.
- Label source = ONE consistent choice: **RUGD annotations** (clean, to de-risk first) or SAM3
  (for scaling to non-RUGD video later). **Do NOT mix** (SAM3 input + RUGD target = mismatch).

**Running (overnight, detached):** RUGD frames + annotations downloading to `~/joana/rugd_full/`
(`RUGD_frames-with-annotations/` + `RUGD_annotations/`). HF rate-limits → slow but resuming.
Logs: `rugd_download.log`, `rugd_annotations_download.log`.

**Code state:** the OUTPUT/generation-target half is scaffolded + committed
(`diffsynth/utils/semantics.py` + the `input_latents` concat, guarded by `pipe.semantic_channels`).
**Still TODO:** the CONDITIONING half — control branch takes the holey semantic like depth
(`wan_video_neoverse_controller.py` ~L91, `control_in_dim` 96→112, zero-init).

**NEXT (tomorrow):**
1. Verify both downloads finished (`ls ~/joana/rugd_full/RUGD_*`).
2. Chunk RUGD frames → clips (`scripts/prepare_rugd_clips.py`) + pair each frame w/ its RUGD mask.
3. Add the conditioning edit (control branch + holey semantic render).
4. Dataloader: feed RUGD masks as the clean target.
5. Cluster smoke-test: overfit ONE clip (training can't run on the Jetson).

See `docs/FINETUNE_IMPLEMENTATION.md` for the exact edits.

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
