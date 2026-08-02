# Semantic architecture decision — v8 design (2026-08-01)

The decision to make: **which upgrades to the semantic diffusion go into v8, in
what order.** Scope corrected after discussion: the encoder/clustering ideas
from the last advisor meeting are upgrades **to the diffusion training**, not a
switch to a different semantic representation; 3D semantic training stays a
"later" item, as filed.

References: `SEMANTIC_V8_RESEARCH.md` (18 adversarially-verified claims, cited
below as [R1]-[R6]), v6/v7 training configs, the world-model boundary results
(`nav-rl/WORLD_MODEL_LIMITS.md`).

---

## 1. Current architecture (v7) and its two diagnosed flaws

```
GT semantic mask (class IDs per pixel)
  -> colorize (class -> RGB color)                     FLAW 1
  -> shared photo-VAE encode -> 16 latent channels     FLAW 1
  -> concat with RGB latents -> 32 channels
DiT (patch_embed/head expanded 16->32; parallel _sem submodules since v6)
  conditioned on the rasterized hint (rough RGB + depth + holey semantics)
  co-denoises RGB + semantics jointly (distill LoRA merged at train time)
Loss: MSE on predicted noise/velocity, in latent space  FLAW 2
Inference: denoise -> VAE decode -> nearest-palette -> class map
```

- **FLAW 1 — the label encoding.** Class masks are compressed by a VAE built
  for photographs; it has no concept of classes. Colorize-then-VAE is
  measurably the weakest label encoding tested in the literature (IoU 0.432
  analog-bits vs 0.312 colorize) [R3].
- **FLAW 2 — the loss.** Noise-space MSE never asks "is this pixel the right
  class"; predicting noise for label maps is specifically identified as
  failing to guide segmentation [R1]. Nothing anywhere in the objective
  rewards sharp boundaries or homogeneous regions [R2].
- v7 status: correct classes, clean large regions, residual boundary
  speckle — exactly what this recipe produces when working as designed [R2].

## 2. The upgrade menu

| ID | Upgrade | Attacks | Effort | Evidence |
|----|---------|---------|--------|----------|
| U1 | **x0-prediction for the semantic half** — train the sem channels to predict the *clean* latent, not noise/velocity | FLAW 2 | ~1-2 d | [R1]: switching to clean-signal prediction removes noisy patches; two independent papers |
| U2 | **Decoded-space CE** — at low-noise timesteps, VAE-decode the sem half and apply cross-entropy against GT classes (small conv head decodes colorized output -> logits) | FLAW 2 | ~1-2 d | [R4]: decoded-space CE/Dice works; plain CE beats hybrid objectives; refinement buys ~5 pts boundary IoU |
| U3 | **SAM2 segment-homogeneity loss** (advisor suggestion) — precompute class-agnostic SAM2 segments per frame; penalize label variation *within* a segment (intra-segment entropy of predicted class probs) | speckle directly | ~1-2 d incl. preprocessing | segments encode "these pixels belong together" without needing to know the class; speckle is by definition a homogeneity violation. No literature citation — advisor-proposed, mechanism sound |
| U4 | **Encoder-feature label pathway** (advisor suggestion: SAM3 encoder / DINO) — replace colorize+VAE for the label channel with features from a pretrained semantic encoder; small learned head decodes rendered features to our class set | FLAW 1 | ~4-6 d | family-B literature distills SAM/CLIP/DINO features successfully [R6]; collaborator (S.T.) has working SAM3+SIGLIP2 fusion code to learn from |
| U5 | **Analog-bits encoding** — class ID -> binary bits per pixel, scaled to [-1,1]; bypass palette+VAE | FLAW 1 | ~3-5 d | [R3]: +12 IoU over colorize head-to-head; PDM recipe |
| U6 | **3D semantic training** (advisor: "later") — supervise semantics on the Gaussians in 3D | representation | out of scope for v8 | family-B research + collaborator code become directly relevant when this opens |

U4 and U5 are alternatives for FLAW 1 (do not stack). U1+U2+U3 stack with
each other and with either of U4/U5.

## 3. Recommended recipe

**v8-core = U1 + U2 + U3.** One training run, ~4-6 implementation days total,
every piece attacks a *diagnosed* mechanism, and none touches the label
encoding — so the v7 data pipeline, configs, and checkpoints all carry over.
Highest evidence-per-effort of any combination.

**v8-deep = U4**, staged after v8-core is read. Rationale for deferring: it is
the biggest surgery (new encoder in the loop, new decode head, retraining from
a different input distribution), and its value depends on how much residual
error v8-core leaves. If v8-core's boundary IoU is already acceptable for the
reward, U4 may be unnecessary; if not, U4 is the deeper fix — and by then the
collaborator's fusion code should be in hand. U5 is the fallback if U4 proves
too invasive.

Decision points the recipe leaves open (for discussion, not blocking):
- U2's decode head: nearest-palette soft-assignment (no new params) vs. a
  small trained conv head (cleaner gradients — recommended).
- U3 segment source: SAM2 per-frame (simplest) vs. SAM2 video mode
  (temporally-linked segments — also gives a free temporal-consistency signal;
  costlier preprocessing). Recommend per-frame for v8, video mode later.
- Loss weights: start CE and homogeneity at ~0.1x the diffusion loss and
  tune by validation mIoU, not by eye.

## 4. Class-set coupling (decide BEFORE the v8 training run)

The advisor wants fewer, navigation-relevant, higher-level classes. This is
**not** part of today's architecture decision, but it must be settled before
v8 trains, because the class set defines the training targets — deciding after
means training twice. Sketch for that discussion (from 30 classes to ~7):

| High-level class | Absorbs (examples) |
|---|---|
| preferred surface | trail, sidewalk, dirt, gravel, asphalt, concrete |
| walkable | grass, mulch |
| risky surface | sand, mud, rock, stairs, vegetation (low) |
| obstacle | tree, wall, building, fence, pole, log |
| dynamic | person, vehicle, bicycle |
| hazard | water |
| sky / void | sky; unlabeled |

Nav-relevant distinctions preserved (sidewalk-vs-road if kept as scored
subclasses); everything the policy treats identically collapses. Fewer classes
also *helps* v8: fewer targets, cleaner boundaries, more examples per class.
`config/traversability.yaml` scores collapse accordingly (score = per-class,
so a remap table is the only code change).

## 5. Evaluation plan (same for every variant — this is what decides)

- **mIoU + boundary IoU** on held-out RUGD frames with dense GT, vs three
  baselines: v7 (current), SAM3-per-frame on rendered RGB (the no-training
  alternative), and the rasterized hint itself (floor).
- **Speckle metric**: intra-SAM2-segment label entropy on outputs (the thing
  U3 optimizes, measured on held-out data — also quantifies v7's speckle for
  the paper).
- **Reward fidelity**: traversability score along expert trajectories using
  each semantic source vs. GT-label scores — the metric that matters for RL.

## 6. Risks / honest notes

- U1 touches the training target computation for half the channels of a
  flow-matching model — the one place where a subtle bug silently trains
  garbage. Budget a 1-clip overfit smoke test before any long run.
- U2/U3 require VAE decode in the training loop at selected timesteps:
  ~2x step cost at those steps; run decode-losses on a subset (e.g. 25% of
  steps) if throughput hurts.
- SAM2 preprocessing over all training clips is a one-time batch job (cluster,
  ~minutes/clip); segments cached as npz like the SAM3 labels.
- The two advisor suggestions are implemented here as *losses/pathways inside
  the existing co-denoising design* — if the intent was a larger architectural
  change (e.g., replace the diffusion with an encoder-decoder segmenter),
  that's a different conversation to have explicitly before building.
