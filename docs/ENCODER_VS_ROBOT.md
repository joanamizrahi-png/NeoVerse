# Decision: encoder work vs robot deployment (remaining pre-freeze time)

Context (2026-08-12): semantic model v10 + trained reader scores 79.9% /
80.6% on the two held-out scenes (best so far, beats v9's 78.8 / 69.9), and
its RGB is at parity with the pretrained model against real footage
(13.7 vs 13.2 dB / 18.7 vs 19.2 dB). One inference pass now gives aligned
RGB + semantics. Training freeze ~Aug 24, GPU until Sept 22, paper mid-Sept.

Two candidates for the remaining build time — they don't both fit.

## Option A — encoder / representation work

The remaining semantic lever. Three sizes:

| variant | effort | what it buys |
|---|---|---|
| analog-bits semantic latent | 2-3 d + retrain | classes stop being blendable colors → sharper boundaries |
| label autoencoder (replaces colorize+VAE) | 4-5 d + retrain | purpose-built label compression; hardest part is matching the video VAE's 4x temporal squeeze |
| DINOv2 features on Gaussians (no SAM3) | 1.5-2 wk | removes SAM3 errors at the source; reconstructor already computes DINOv2 per frame; needs feature compression + rasterizer path + class probe |

Expected gain: boundary quality / a few accuracy points / SAM3 independence.
Risk: each is a chain (encoder → integration → retrain → eval); a failed link
eats the calendar. And accuracy is no longer the blocker at ~80%.

## Option B — deploy the policy on the Go2W

The paper's core claim is real-world transfer; nothing substitutes a robot
result. Path: export the policy (image + goal vector → action), onboard
inference on the Jetson, goal vector from odometry, camera in. ~1 week of
integration, no GPU-cluster dependency (frees the cluster for the ablations
and the v14 policy retrain in parallel).

Risk: sim-to-real gap is unknown until tried — which is also the argument
for trying before the paper freeze, while there is still time to react.

## Recommendation

B first. A's cheapest variant (analog bits) only if the robot path stalls on
hardware. DINOv2 = post-deadline direction (strongest long-term, doesn't fit).

Reasoning: semantic accuracy stopped being the bottleneck this week; the
robot demo is the one result that can't be produced later, and the deadline
math only fits one of these.
