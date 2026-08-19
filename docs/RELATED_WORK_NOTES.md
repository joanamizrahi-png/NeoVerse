# Related work landscape (deep-research sweep, 2026-08-18; 21 verified claims)

## Closest works, one line each, and how we differ
- **Vid2Sim** (arXiv:2501.06693) — monocular video -> 3DGS+mesh sim -> RL visual
  navigation. THE closest pipeline. Differs: no generative completion of unseen
  views, no semantics co-generation, no traversability reward — geometry-only sim.
- **VR-Robo** (2502.01536) — 3DGS-mesh hybrid sim, PPO nav+locomotion, RGB-only
  sim-to-real on quadruped. Differs: multi-view capture (not monocular robot
  video), no diffusion completion, task rewards only.
- **World4RL** (2509.19080) — PPO entirely inside a frozen diffusion world model
  (manipulation); +25% absolute on real Franka vs BC/DP. Precedent: policies
  optimized purely in a diffused simulator transfer to hardware. Also: OOD-action
  clipping (sigma<=e^0) = precedent for constraining exploration in learned sims.
- **RAW-Dream** (2605.12334) — RL (GRPO) inside a **Wan 2.1** video-diffusion
  world model (same base family as ours!), VLM reward. Found concrete
  reward-hacking (visually plausible but dynamically false rollouts fooling the
  reward) — mitigated by dual-noise verification. CITE next to our
  backwards-gait exploit.
- **LucidSim** (2411.00083) — quadruped parkour trained on generated images,
  zero-shot real transfer; beats domain randomization 100 vs 70 / 85 vs 35 etc.
  Precedent: generated-image training is not a compromise, it's a win.
- **RWM-U** (2504.16680) — uncertainty-penalized PPO in an autoregressive world
  model, deployed on real quadruped + humanoid. Sibling of our alpha gate
  (statistical uncertainty where ours is geometric evidence).
- **MOPO** (2005.13239) — THE canonical citation for our gate: subtract an
  uncertainty penalty from reward in a learned model; provably maximizes a lower
  bound on true return. Our alpha-gate = the geometric-evidence instantiation.
- **Model-exploitation theory** (2605.15960) — Theorem 1: any imperfect learned
  simulator admits policies that game its errors (exploitation is unavoidable);
  closed-form "safe horizon" bound. Frames our backwards-gait finding as
  predicted-by-theory, handled-by-design.
- **NavDP** (2505.08712) — nav diffusion policy trained in handcrafted sim,
  zero-shot real. Its baseline suite = the reviewer-expected list: GNM / ViNT /
  NoMaD (IL foundation policies), DD-PPO (RL in sim), iPlanner / ViPlanner /
  EgoPlanner (planners).

## Comparison shortlist, ranked by OUR implementation cost
1. **BC on demonstrations** — DONE (bc2 arm).
2. **RL in non-generative sim** (rasterized backend) — DONE (raster control).
3. **Reward-design ablations** (gated vs ungated vs backcost) — DONE (the 2x3+ table).
4. **IL foundation policy zero-shot** (ViNT or NoMaD public checkpoint on our
   scenes / robot) — the one real gap; medium cost (image-goal conditioning
   needs adapting to our position-goal protocol); the single highest-value add.
5. **Classical/learned planner** (iPlanner/EgoPlanner) — high cost on our
   monocular stack (they want depth/LiDAR); cite-and-punt unless time appears.

## Reward-hacking / hallucination-handling citations for the paper
- MOPO (uncertainty-penalized reward, lower-bound theory) -> alpha gate.
- RWM-U (uncertainty propagation, real deployment) -> gate sibling.
- RAW-Dream (reward hacking in Wan-based WM + verification fix) -> our
  backwards exploit + backcost/footprint fix.
- Exploitation theory (2605.15960) -> why gates/penalties are necessary, not
  optional; safe-horizon bound motivates short-episode training.

## Eval-protocol recommendations (ICRA)
- Report per-scene: success rate, steps-to-goal (or SPL), collision steps —
  matches what we already log; add N>=2 seeds (done tonight).
- Position-goal protocol is standard for our class (image-goal is the IL-
  foundation convention — note when comparing to ViNT/NoMaD).
- Real robot: fixed courses, N trials per course with success % (LucidSim
  style); report collision interventions.
- Datasets: RUGD (ours) + GND campus scenes; SCAND as dynamic stress test.
