# v8 literature findings — sharp/consistent semantics for the world model

Deep-research run 2026-07-23 (18 claims, all adversarially verified 3-0).
Synthesis below is ours. Sources are arXiv links per finding.

## The headline: our speckle has a *diagnosed cause* in the literature

1. **Speckled label patches are specifically attributed to (a) epsilon/noise-prediction
   on the label channel and (b) missing output squashing (tanh)** — switching to
   predicting the CLEAN label signal (x0/M0-prediction) with a final squash removes
   the noisy patches. (arxiv 2601.02881; PDM 2412.02929 independently: predict M0,
   "predicting eps for the map fails to guide"). Our Wan fine-tune predicts
   noise/velocity for the semantic half → we are running the *diagnosed-broken*
   configuration.
2. **Our exact recipe (UDPDiff-style colorize + shared VAE + channel concat) has NO
   sharpness mechanism** — UDPDiff trains with plain diffusion loss, no CE/boundary
   loss; our speckles are *consistent with the recipe, not a bug in our training*.
   (2503.09344)
3. **Colorize-then-VAE is a measurably weak label encoding**: analog bits (class id →
   binary bits per pixel, scaled to [-1,1]) beat RGB-colorize encodings by large
   margins (IoU 0.432 vs 0.312 on EntitySeg); PDM uses 8-bit analog bits + larger
   noise on the label channel. (2601.02881, 2412.02929)
4. **Decoded-space losses work**: LDSeg trains label autoencoder + denoiser end-to-end
   with CE+Dice computed in decoded pixel space; discrete-diffusion segmentation
   (DDPS) finds plain CE beats hybrid objectives, and diffusion refinement buys
   ~5 pts *boundary* IoU specifically. (2407.12952, 2306.01721)
5. **Consistency needs an explicit mechanism**: SP4D (colorize recipe like ours)
   adds a cross-branch fusion module + contrastive consistency loss — "channel
   sharing alone is not relied on for consistency". IDC-Net argues joint co-denoising
   itself is what aligns modalities (supports single-pass over split passes).
   (2509.10687, 2508.04147)
6. **Family B is live**: SegSplat attaches open-vocabulary semantics to FEED-FORWARD
   Gaussian reconstructions (no per-scene optimization) at ~0.03-0.2 s inference —
   1000x faster than LangSplat. Directly compatible with our WorldMirror-style
   pipeline; candidate replacement for SAM3-fusion as the reward's semantic source.
   (2511.18386)

## v8 experiment shortlist (our synthesis; effort in focused days)

| # | Experiment | Change | Effort | Expected payoff | Confidence |
|---|---|---|---|---|---|
| v8a | **x0-prediction + decoded-space CE for the semantic half** | training_loss: predict clean sem latent (or add x0-space aux CE via VAE decode of sem half at low-noise steps) | 2-4 d | kills speckle per diagnosed mechanism [1,4] | high |
| v8b | **Analog-bits label encoding** (30 classes → 5-8 bits/pixel, bypass palette) | replace labels_to_rgb path; widen _sem modules accordingly | 3-5 d | +large IoU per [3]; independent of v8a | med-high |
| v8c | **SegSplat-style feed-forward semantic field** (family B) | new reward-source branch; ablation vs SAM3-fusion vs diffuser | 4-6 d | view-consistent, hallucination-free reward labels at negligible runtime | med |
| v8d | Data expansion (more RUGD clips + RELLIS-3D remap) | prep scripts exist; SAM3 hint passes | 2-3 d | scaling; benefits all of the above | high |

Recommended order: v8a + v8d together (cheapest, highest-confidence), then v8b,
with v8c explored in parallel as the reward-source ablation.

## Consistency/hallucination trade-off (for the reward)

- Joint co-generation is the literature's consistency mechanism [5]; if the diffuser
  feeds the reward, prefer the single joint pass over split passes.
- SP4D shows even joint passes add explicit consistency losses — worth adopting if
  we keep co-generation.
- Family B (SegSplat-style) sidesteps hallucination entirely (labels only where
  geometry exists) — the honest-reward option; pair with the diffuser-for-
  observations split.

## Caveats

- Verification of several family-B claims (feed-forward vs optimized accuracy gap,
  per-Gaussian blending artifacts, warping losses) was cut off by session limits —
  treat SegSplat numbers as single-source until re-checked.
- 2601.02881 / 2511.18386 are very recent preprints; results not yet replicated.
