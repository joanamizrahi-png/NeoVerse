# Class-set proposal — 14 navigation classes (draft for advisor review, 2026-08-04)

Reduces the 30-class taxonomy to 14 behavior-distinct classes. Design rules:
a class exists only if the robot acts differently on it; SAM3 prompts stay
concrete visual nouns (unchanged) and a remap table does the categorizing;
overlaps resolve by priority (function labels beat material labels); safety
classes never merge.

| id | class | fed by (prompts / old classes) | score | rationale |
|----|-------|-------------------------------|-------|-----------|
| 0  | void | unlabeled | 0.0 | conservative default |
| 1  | sky | sky | 0.0 | |
| 2  | trail surface | dirt, gravel, mulch | 0.85 | one walkable loose-surface class; dissolves the measured gravel/sand/dirt confusion |
| 3  | grass | grass | 0.85 | |
| 4  | rough ground | sand, mud, rock, log | 0.35 | passable with care; sand scored as the average over depths (thin sandy trail ok, deep sand not — depth is not visible to semantics; proprioception handles it at deployment) |
| 5  | water | water | 0.0 | hazard: looks like ground, never step |
| 6  | sidewalk | sidewalk, crosswalk | 0.95 | function known: pedestrian |
| 7  | road | road | 0.5 | function known: vehicles. Road danger lives in class 13 + collision, not the surface score |
| 8  | pavement (unknown) | concrete, asphalt where no function prompt fired | 0.7 | honest-uncertainty class: hard surface, function unrecognized (plazas, paths — or a missed road, hence mid score, not 0.95) |
| 9  | stairs | stairs | 0.5 | own class: the Go2 has a dedicated stair gait — neither rough ground nor obstacle |
| 10 | obstacle | tree, building, wall, fence, pole, traffic sign, traffic light, bridge | 0.0 | |
| 11 | vegetation | vegetation | 0.2 | soft/pushable, distinct from solid obstacle |
| 12 | person | person | 0.0 | safety-critical, never merges |
| 13 | vehicle | vehicle, motorcycle, bicycle | 0.0 | safety-critical |

## Points explicitly open for discussion

1. **pavement (unknown)** — new construct: rather than defaulting bare
   concrete/asphalt to sidewalk (unsafe if it's a missed road) or to road
   (needlessly avoids plazas), unknown-function pavement gets its own class
   at 0.7.
2. **sand → rough ground** — asserts the robot handles thin sand but not deep
   sand; 0.35 encodes the average. Sanity-check against platform experience.
3. **traffic lane — cut** — no GT in RUGD or Cityscapes, and no behavioral
   difference from road identified. Argue it back in if lane-level behavior
   is wanted.
4. **bridge → obstacle** — correct for current scenes; wrong if any deployment
   route crosses a walkable bridge.

## Standing rules going forward

- **Prompt growth**: new data domains may ADD prompts (concrete nouns only:
  "bench", "curb", ...); every new prompt declares its class in the remap at
  birth. Class ids never move, so old label files stay valid.
- **Held-out clips**: the next training explicitly excludes 2-3 GT'd clips
  from the training set so all future quality numbers are honest
  (current v7/v8 numbers are training-set numbers and are only used for
  recipe comparison, not absolute claims).

## Implementation notes (after approval)

- One shared constants module becomes the single source of truth (prompts,
  priorities, remap, palette, scores) — this also retires a discovered
  id-ordering mismatch between the SAM3 labeler's class list and
  CLASS_COLORS/traversability.yaml for old ids 15–21.
- Remap tables: RUGD-GT → 14, old-30 (SAM3 npz) → 14; all label files
  regenerate by table lookup, no SAM3 re-run needed.
- num_semantic_classes: 14 end-to-end (CE head, palette, yaml).
- Expected training benefit: 14 well-separated targets vs 30 with
  color-neighbor pairs; the measured v8 confusions (gravel→sand 29%,
  rock→gravel 74%) collapse into single classes.
