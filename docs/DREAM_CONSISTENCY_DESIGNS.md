# Dream Consistency: two designs for cache-wide hallucination coherence

Problem (2026-08-18): within one diffusion call (81 frames) hallucinated
content is coherent; across calls it is not. The cache is built from ~300
calls, so unobserved regions hold ~300 unrelated dreams — turning or walking
through them flickers (camera-dream, floor-dream, per-spot re-imagination).
The forest prompt constrains dream CONTENT but not cross-call CONSISTENCY.
Goal: dreams as consistent as if the whole cache came from one long call.

---

## Design 1 — DREAM LIFTING (commit the dream to geometry)

**Mechanism.** Generate a region's dream once (e.g. one spin sweep). Feed the
generated frames back into the feed-forward reconstructor alongside the real
video frames. The reconstructor lifts the dream into Gaussians — the dream
becomes geometry. Every subsequent call rasterizes that geometry as its hint,
so it must stay consistent with the committed dream. Alpha in those regions
becomes 1 honestly-by-construction: "imagined once, then committed." The
dream's semantic labels lift too (labeled-Gaussian fusion), so the gated
reward can READ committed-dream labels instead of voiding them.

One dream per region, permanent, shared by all views. This is the persistent
generated-world idea in miniature (InSpatio-WorldFM direction) built from
parts we already have.

**Implementation delta** (small): `inference_semantic.py --append_views_dir`
— load a rendered sweep's `rgb.mp4` (+ `semantic_labels.npz`) and append the
frames to the input view set before reconstruction. The reconstructor
estimates poses itself (WorldMirror-style), so no pose plumbing. Appended
frames get the anchor's timestamp (static commit).

**Risks.** (a) View budget: reconstructor trained at 81 views; 162 may
degrade or OOM. (b) Pose estimation on dreamed content may fail or drift.
(c) Dream Gaussians may be low-confidence blobs that rasterize as mush.
(d) Real-vs-dream Gaussian conflicts near the alpha boundary.

**Pilot (1 GPU-hour, answers go/no-go):**
1. Input: rugd_trail_00 real clip + the already-rendered `spin_f40_lat+0.00`
   (its 81 frames cover the full ring at spot 40).
2. Run reconstruction with 81 real + 81 appended frames. GATE A (survival):
   reconstruction completes, no OOM, pose estimates sane.
3. Re-render the SAME spin trajectory rasterizer-only from the lifted field.
   GATE B (commit): backward-heading alpha jumps from ~0% to >50%, and the
   rough render shows the dream content (eyeball video).
4. Full diffusion render of the NEIGHBOR spin (`spin_f42`) against the lifted
   field. GATE C (consistency, the point): in the formerly-dream region,
   f42-vs-f40 label agreement was 68–84% cross-call — target >90%, and the
   walk video through f38/f40/f42 stops flickering (her eyes).
5. Metric harness: reuse translation_probe agreement numbers, before/after.

**Cost if pilot passes:** lift pass = one spin render + one reconstruction
per scene region; cache regeneration on the lifted field. ~2 days to
production. If gate A or B fails: stop, cost was 1 hour.

---

## Design 2 — SEQUENTIAL-OVERLAP GENERATION (chain the calls)

**Mechanism.** Order all sweeps in a serpentine through the (lat × heading /
anchor) grid so consecutive calls are spatial neighbors. Seed call k with
call k-1's overlapping tail: initialize the first L target frames from the
previous call's output via partial noising (SDEdit-style: noise the known
frames to an intermediate sigma, denoise the rest from full noise), so the
dream flows across the boundary. The whole cache approximates one long
autoregressive sequence.

**Implementation delta** (medium): pipeline support for per-frame init
latents + per-frame denoise start sigma. Friction: our 4-step distilled
sampler has a coarse sigma ladder — partial noising may land badly between
steps. Generation becomes serial (no 8-way job parallelism; ~300 calls
× 40s ≈ 3.5h serial per scene, acceptable).

**Risks.** (a) Distilled-sampler incompatibility with partial noising —
main risk. (b) Autoregressive drift: errors compound over 300 calls
(mitigate: restart chains per lane; serpentine within lane). (c) Chains
along one axis only — cross-lane consistency still relies on shared seed.

**Pilot (2 GPU-hours):**
1. Implement init-latent seeding for the first L=8 frames only.
2. Chain THREE calls: spin_f40 → spin_f42 → spin_f44, each seeded with the
   previous spin's overlapping heading window. GATE A (mechanics): outputs
   are not corrupted at the seam (no ghosting/blur burst at frame L).
3. GATE B (consistency): same f40/f42 dream-agreement metric as Design 1
   gate C — target >90% at the seam headings, and decaying gracefully, not
   cliff-dropping, away from it.
4. GATE C (drift): chain 10 calls down one lane; compare call 1 vs call 10
   dream statistics (label distribution, RGB tone) — drift < the current
   cross-call disagreement, else chains must be short.

**Cost if pilot passes:** ~3 days to production (sampler work dominates).

---

## Recommendation
Pilot Design 1 first: cheaper pilot, no sampler surgery, converts the alpha
gate from "void the dream" to "trust the committed dream" (a conceptual
upgrade for the reward story), and it IS the persistent-world thesis chapter
in embryo. Design 2 is the fallback if reconstruction rejects generated
views. Neither blocks this week's policy results; both are pre-scoped for
the first free GPU-day.
