# pose2rot — Learning Joint Positions → Rotations for Arbitrary Skeletons

> A focused study and training recipe for the **Pose-to-Rotation** stage of MoCapAnything V2:
> given a sequence of 3D joint **positions**, predict per-joint **6D rotations** (then forward
> kinematics recovers the full skeletal animation). One model handles **arbitrary skeletons**
> across 72 species — quadrupeds, bipeds, birds, reptiles, dinosaurs, arthropods, even limbless
> snakes — via T5 joint-name embeddings, skeleton graph attention, and rest-pose FiLM conditioning.

This repository is a derivative work of **[MocapAnything](https://github.com/phongdaot/MocapAnything)**
(MIT License, © 2026 Dao Thien Phong; arXiv:2604.28130). Our contributions focus on the `pose2rot`
component — see **Our Contributions** below. The original MIT license is retained (`LICENSE`, `NOTICE.md`).

---

## Why is position → rotation hard?

It is **ill-posed**: the same joint positions can come from many different rotations (bone-axis
twist is unconstrained by position). The model resolves this with a **reference pose-rotation pair**
+ **skeleton structure priors**, turning the multi-valued mapping into a constrained conditional prediction.

## Our Contributions

Starting from the upstream `Pose2RotMemoryRestModel`, we developed a stable, reproducible training
recipe and an honest held-out evaluation:

- **Anti-collapse recipe.** Posterior collapse (model outputs a per-species constant pose) is the main
  failure mode. We fix it with two complementary mechanisms: (a) **memory ablation**
  (`decoder_use_cross_layers=0`) to remove the "copy the species-constant pose from the memory bank"
  shortcut, and (b) a **de-meaned temporal loss** (`tvar`) whose `I − 1/T ≠ 0` gradient actively pushes
  the model to produce time-varying motion. Monitored by `ratio_DYN` (predicted vs GT temporal energy).
- **FK-position loss with corrected conventions.** Adding an FK loss to fix limb drift first *corrupted*
  the model because the FK used a transposed rotation convention + a wrong offset (`FK(GT) ≠ position`).
  Fixed to the **row convention** (`rotation_6d_to_matrix`) + **raw BVH offset**, hard-verified
  `FK(GT) ≈ position` (err ≈ 0) before use. *Lesson: verify `FK(GT) ≈ target` before any FK-based loss.*
- **fk-weight ramp.** A strong constant FK weight from scratch suppresses anti-collapse; a constant
  `fk=30` even **diverges** on continued training (its gradient ≈ grad-clip → metastable). We **ramp
  `fk` from 0 → 10** over epochs 5–15: gentle early so the anti-collapse breaks, then refine.
- **Held-out evaluation protocol.** A proper **seen / rare / unseen** split (`scripts/build_split.py`),
  a **geodesic angle-error metric** in degrees (`scripts/geodesic_eval.py`) to compare head-to-head
  with MoCapAnything, and a fix for a **cache split-tag leakage bug** (the items cache filename did not
  encode the split → a held-out run silently loaded an all-data cache → test motions leaked into training).
- **Visual QA tooling** (`scripts/pose2rot_qa.py`) — GT-vs-pred side-by-side GIFs, because metrics lie
  (both the FK bug and the cache leak were caught by visual/sanity checks, not by the training loss).

## Key Results (geodesic angle error, degrees; MoCapAnything V2 reports 6.54° unseen / V1 ~17°)

| Setting | seen | rare | unseen | overall |
|---|---|---|---|---|
| **all-data (oracle, model saw every species)** | 7.2° | 5.9° | 6.4° | **6.53°** ≈ MoCapAnything 6.54° |
| **true held-out** | 9.8° | 12.7° | 40.9° | 28.0° |

**Finding.** When the model has *seen* a species, it matches SOTA (6.5°). On a **true held-out** test,
cross-topology generalization is a ceiling: unseen species with close training relatives (Goat 17°,
Coyote 19°) generalize partially, while topologically distinctive ones (Pigeon ~67°, Spider ~73°) do not.
A per-species `oracle-vs-held-out` comparison shows the unseen failure is purely a **generalization cost**
(the oracle does 5–8° on the same species), not intrinsic difficulty. See `docs/` for the full analysis.

## Quick Start

```bash
# 1. environment (see also setup.sh / requirements.txt)
pip install -r requirements.txt

# 2. data: Truebones Zoo, preprocessed into per-clip pose npz + per-species memory banks
#    (see preprocess/ ; original data is NOT included — see RUN.md)

# 3. train pose2rot (DDP, 2 GPUs, the held-out decisive recipe)
torchrun --nproc_per_node=2 -m train.pose2rot \
    --config configs/train/train_pose2rot_v10_split_heldout.yaml

# 4. evaluate (per-tier geodesic angle error vs MoCapAnything 6.54°/17°)
python scripts/geodesic_eval.py \
    configs/train/train_pose2rot_v10_split_heldout.yaml exp_pose2rot_v10_split_heldout

# 5. anti-collapse sanity check (ratio_DYN > 0.3 = collapse broken)
python scripts/check_collapse.py \
    configs/train/train_pose2rot_v10_split_heldout.yaml exp_pose2rot_v10_split_heldout

# 6. visual QA — GT vs pred GIFs across species
python scripts/pose2rot_qa.py \
    --config configs/train/train_pose2rot_v9_fk10ramp.yaml \
    --ckpt_dir checkpoints/pose2rot/exp_pose2rot_v9_fk10ramp_60ep \
    --species Horse Elephant Lion Eagle Crocodile Trex Spider KingCobra --n_clips 1
```

Key configs: `configs/train/train_pose2rot_v9_fk10ramp.yaml` (all-data) and
`train_pose2rot_v10_split_heldout.yaml` (held-out, the paper model).

## Checkpoints

Pretrained pose2rot checkpoints are on Hugging Face: **https://huggingface.co/Tevior/pose2rot**.
- `v9` all-data (best for demos / when the species is seen)
- `v10` held-out (the decisive paper model)
- `v8b` earlier converged best

## Docs

- `docs/pose2rot_成功训练讲解.md` — full walkthrough (Chinese): data shapes → preprocessing →
  model design → tensor flow → training → the bug-by-bug success story.
- `docs/pose2rot_experiment_archive.md` — experiment archive (the full history + defensible narrative).
- `RUN.md` / `README_upstream.md` — upstream MoCapAnything pipeline docs.

## Attribution & License

MIT License. Built on [MocapAnything](https://github.com/phongdaot/MocapAnything) by Dao Thien Phong
(© 2026, MIT). See `LICENSE` and `NOTICE.md`. If you use this work, please also cite MoCapAnything (arXiv:2604.28130).
