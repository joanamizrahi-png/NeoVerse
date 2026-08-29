"""THE single source of truth for the 14-class navigation taxonomy (v14).

Everything class-related imports from here: palette, traversability scores,
SAM3 prompt list (with per-prompt class + priority), and the remap tables from
the legacy 30-class space. This retires the three-hand-synced-lists era that
produced the id-ordering mismatch (SAM3 labeler vs CLASS_COLORS, old ids
15-21) discovered 2026-08-04.

Design (docs/CLASS_SET_PROPOSAL.md): a class exists only if the robot acts
differently on it; SAM3 prompts stay concrete visual nouns and the remap does
the categorizing; function labels outrank material labels; safety classes
never merge.
"""
from __future__ import annotations

import numpy as np

# (name, RGB color, traversability score)
V14 = [
    ("void",       (  0,   0,   0), 0.00),   # 0  unlabeled — conservative
    ("sky",        (200, 225, 245), 0.00),   # 1
    ("trail",      (150, 100,  55), 0.85),   # 2  dirt / gravel / mulch
    ("grass",      ( 75, 190,  80), 0.75),   # 3  urban norm: don't trample lawns
    ("rough",      ( 95,  65,  35), 0.35),   # 4  sand / mud / rock / log
    ("water",      ( 50, 120, 200), 0.00),   # 5  hazard: never step
    ("sidewalk",   (210, 210, 210), 0.95),   # 6  function known: pedestrian
    ("road",       ( 70,  70,  85), 0.50),   # 7  function known: vehicles
    ("pavement",   (235, 205, 150), 0.80),   # 8  hard flat > lawn (2026-08-28)
    ("stairs",     (220, 140,  80), 0.50),   # 9  dedicated Go2 gait
    ("obstacle",   (185,  55,  50), 0.00),   # 10 solid statics incl. tree trunks
    ("vegetation", (170, 200,  55), 0.20),   # 11 soft / pushable
    ("person",     (205,  70, 145), 0.00),   # 12 safety-critical
    ("vehicle",    (110, 130, 220), 0.00),   # 13 safety-critical
]
NUM_CLASSES_V14 = len(V14)
V14_NAMES = [n for n, _, _ in V14]
V14_SCORES = np.array([s for _, _, s in V14], dtype=np.float32)


def v14_palette():
    """[14, 3] float in [0,1] — import torch lazily so cpu-only tools work."""
    import torch
    return torch.tensor([c for _, c, _ in V14], dtype=torch.float32) / 255.0


# ---- remap from any legacy label file, BY NAME (immune to id-order bugs) ----
# Covers: the CLASS_COLORS/GT ordering, the SAM3 labeler's ordering (they
# disagreed for old ids 15-21 — remapping by the names stored inside each npz
# sidesteps the question of which ordering a given file used), and RUGD names.
NAME_TO_V14 = {
    "void": 0, "sky": 1,
    "dirt": 2, "gravel": 2, "mulch": 2,
    "grass": 3,
    "sand": 4, "mud": 4, "rock": 4, "rock-bed": 4, "rock bed": 4, "log": 4,
    "water": 5,
    "sidewalk": 6, "crosswalk": 6,
    "road": 7, "asphalt": 8, "concrete": 8,
    "stairs": 9,
    "tree": 10, "building": 10, "wall": 10, "fence": 10, "pole": 10,
    "traffic sign": 10, "traffic_sign": 10, "traffic light": 10,
    "traffic_light": 10, "sign": 10, "bridge": 10,
    "vegetation": 11, "bush": 11,
    "person": 12,
    "vehicle": 13, "motorcycle": 13, "bicycle": 13, "bike": 13, "car": 13,
}


def remap_array_from_names(class_names) -> np.ndarray:
    """Given a legacy file's own class_names list (index = its id), return a
    lookup array old_id -> v14 id. Unknown names map to void with a warning."""
    lut = np.zeros(max(len(class_names), 32), dtype=np.int8)
    unknown = []
    for i, raw in enumerate(class_names):
        n = str(raw).strip().lower().replace("_", " ")
        if n not in NAME_TO_V14 and " " in n and n.split(" ")[0] in NAME_TO_V14:
            n = n.split(" ")[0]           # e.g. "rock bed" already covered; fallback
        if n in NAME_TO_V14:
            lut[i] = NAME_TO_V14[n]
        else:
            lut[i] = 0
            unknown.append(raw)
    if unknown:
        print(f"[class_taxonomy] WARNING: unmapped names -> void: {unknown}")
    return lut


# ---- SAM3 prompt list for FUTURE labeling runs ----
# (prompt, v14 class id, priority). Priority: higher wins on pixel overlap;
# function labels (sidewalk/road) outrank materials (concrete/asphalt) so the
# "pavement (unknown)" class only catches material-without-function pixels.
SAM3_PROMPTS = [
    ("sky", 1, 10),
    ("dirt", 2, 50), ("gravel", 2, 50), ("mulch", 2, 50),
    ("grass", 3, 55),
    ("sand", 4, 50), ("mud", 4, 60), ("rock", 4, 60), ("log", 4, 90),
    ("water", 5, 60),
    ("concrete", 8, 70), ("asphalt", 8, 70),
    ("road", 7, 80), ("sidewalk", 6, 80), ("crosswalk", 6, 85),
    ("curb", 10, 88),   # 2026-08-28: drop-edge hazard; must beat the ground
    ("stairs", 9, 85),
    ("vegetation", 11, 30),
    ("tree", 10, 90), ("building", 10, 100), ("wall", 10, 100),
    ("fence", 10, 100), ("bridge", 10, 100), ("pole", 10, 95),
    ("traffic sign", 10, 95), ("traffic light", 10, 95),
    ("vehicle", 13, 110), ("motorcycle", 13, 110), ("bicycle", 13, 110),
    ("person", 12, 120),
]


def emit_traversability_yaml() -> str:
    """The nav-rl config/traversability_v14.yaml content, generated (never
    hand-edit the yaml again — regenerate from here)."""
    lines = ["# GENERATED from NeoVerse/diffsynth/utils/class_taxonomy.py — do not hand-edit.",
             "# Per-class traversability for the 14-class navigation taxonomy."]
    for i, (name, _, score) in enumerate(V14):
        lines += [f"{i}:", f"  name: {name}", f"  score: {score}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "yaml":
        print(emit_traversability_yaml())
    else:
        print(f"{NUM_CLASSES_V14} classes:")
        for i, (n, c, s) in enumerate(V14):
            print(f"  {i:2d} {n:11s} rgb{c} score {s}")
