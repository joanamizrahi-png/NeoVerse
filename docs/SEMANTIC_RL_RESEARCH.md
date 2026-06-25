# Semantic World Model → RL Navigation — Research Report

*Reconstructed from a deep-research run (104 agents, 22 sources, claims 3-vote
adversarially verified). Findings below passed verification; the one refuted claim is
flagged because it's an important correction.*

## Q1 — Make the diffusion output semantics jointly with RGB

**Validated recipe: colorize → shared-VAE encode → channel-concat → co-denoise.**

- **UDPDiff** — fine-tunes pretrained CogVideoX-5B to jointly predict RGB + entity
  segmentation + depth via channel-concat (doubling to 32-ch latent). **Closest template**
  (RGB+segmentation, finetuning an existing video diffusion).
- **IDC-Net** — joint RGB+depth: `(v0,d0)=EncVAE(I,D)` → concat `[v0;d0]` → one diffusion →
  split → decode each through the **same VAE decoder**. Minimal, clean.
- **Stable Part Diffusion 4D** (arXiv 2509.10687) — dual-branch RGB+segmentation, "spatial
  color encoding" maps discrete masks → continuous colors for the VAE.
- **GeoVideo** (arXiv 2512.03453) — RGB+depth joint denoising.

### ⚠️ REFUTED claim (0/3) — important correction
"Channel-concat the semantic latent and reuse the RGB VAE decoder, no new decoder" is
**wrong**: (1) discrete masks are NOT image-like — must **colorize** before encoding;
(2) the non-RGB modality may need a **small separate decoder** (GeoVideo trains one;
IDC-Net's shared decoder works only with the colormap trick).

**→ Minimal recipe for NeoVerse/Wan:** colorize SAM3 mask → encode via Wan VAE →
channel-concat with RGB latent → finetune DiT to co-denoise → decode (shared first, add a
tiny semantic decoder if needed). Freeze RGB.

## Q2 — Data & SAM pseudo-labels
- **V-STRONG** ("Self-Supervised Traversability Learning with Trajectories and SAM Masks")
  validates SAM-mask pseudo-labels for off-road traversability — direct precedent for the
  label strategy.
- **RELLIS-3D** (20 classes, off-road), **RUGD** = clean pre-labeled fallback.
  **ORAD-3D** (arXiv 2510.16500) = newer large-scale off-road.
- Pitfalls: label noise + temporal flicker (SAM3 video tracking mitigates).

## Q3 — World model as RL simulator
- **World4RL** (arXiv 2509.19080) / **World-Gymnast** — refine a *pretrained* policy with
  *few* high-quality rollouts (not from-scratch RL) → slow diffusion becomes affordable.
- **Epona** — couples trajectory planning with generation (relevant to static→dynamic).
- Consensus: fine-tune pretrained policy, short rollouts; nobody runs millions of slow
  generative steps.

## Q4 — Novelty / positioning
Landscape: driving world models (DriveDreamer/GAIA — no legged nav); traversability
learning (V-STRONG — no world-model sim); world-model RL (World4RL — manipulation, not
outdoor legged). **Uncovered gap = the full combination:** semantic 4D world model from
real video → RL navigation w/ semantic-traversability reward → real legged-robot transfer.
Defensible contribution.

## Q5 — Concrete next steps
1. **Now (Jetson-safe):** SAM3-label a small video set (RELLIS/RUGD + own clips).
2. **Recipe:** UDPDiff-style (colorize, channel-concat into Wan VAE, finetune DiT, freeze
   RGB). IDC-Net = simplest reference.
3. **Training edits** to `train.py` + dataloader (colorized semantic channel + loss). Cluster.
4. **Policy:** static waypoints + SAM3 traversability reward, fine-tune pretrained Go2
   walker, short rollouts → then dynamic.
5. **Risks:** discrete-mask encoding → colormap (validated); label noise → SAM3 tracking;
   slow generation → refine-not-train + few rollouts; cluster dependency → data prep proceeds
   without it.

## Key sources
- Joint RGB+semantics diffusion: arXiv 2509.10687, 2503.09344, 2512.03453, 2407.10937, 2508.04147, 2401.10227
- Datasets / traversability + SAM pseudo-labels: V-STRONG, RELLIS-3D (github.com/unmannedlab/RELLIS-3D), ORAD-3D (2510.16500), 2503.03947, 2312.16016
- World-model RL: World4RL (2509.19080), 2501.10100, 2502.01536, 2412.03572
- Novelty: 2501.06693, 2602.02454
