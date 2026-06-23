# VisTacFusion

Visuo-tactile **multi-task** model: from a **paired (not pixel-aligned)** RGB image and a
vision-based tactile image, predict — in the **tactile frame** —

1. **Dense 3D**: per-pixel **depth** + **surface normal** (DPT head), and
2. **Pose**: object **SE(2)** pose = heading θ + planar translation (Pose MLP head).

Trained on **sim**, deployed on **real** (sim-to-real). The scientific claim is that
**RGB + tactile beats either modality alone**, shown with a *fair* ablation: **one shared
model**, identical trainable params, and an **identical decoder input** across all three
input configs. See [CLAUDE.md](CLAUDE.md) for the full architecture spec (single source of
truth).

## Key design (do not violate — CLAUDE.md §1)

- **Frozen DINOv3 encoders** (sim2real anchor + half the fairness argument). Only
  projection + embeddings + bottleneck + fusion trunk + heads train.
- **No pixel alignment** — RGB↔tactile correspondence is learned by cross-attention only.
- **Fixed decoder input** — always `4×(B×196×768)` spatial taps + `B×1×768` pose token, in
  *every* modality config. 197 queries (196 spatial + 1 pose); only *what fills them* changes.
- **One shared model + modality dropout** — the three configs are inference-time input modes.
- **Tactile is the spatial anchor; RGB is read-only context** (K/V only, no `Q_rgb`).

## Pipeline

```
RGB  ─[frozen DINOv3]─[proj+mod emb]──────────────► Memory M (197, read-only K/V)
                                                          │
tactile ─[frozen DINOv3]─[proj+mod emb+2D pos]─► 196 spatial queries ┐
                          tactile CLS ─────────► 1 pose query        ├─► Fusion trunk ×L
                                bottleneck tokens (m) ◄───────────────┘   ①bn←RGB ②q←bn ③q self
                                                                          tap 4 layers
        4×(B×196×768) ─► DPT (Reassemble→FeatureFusion→CNN) ─► depth, normal
        B×1×768       ─► Pose MLP ─► (cosθ, sinθ, tₓ, t_y)
```

The three configs (CLAUDE.md §4), all with identical decoder input:

| Config | spatial queries (196) | pose query (1) | cross-modal |
|---|---|---|---|
| RGB+tactile | tactile patch tokens | tactile CLS | ①②③ run |
| tactile-only | tactile patch tokens | tactile CLS | no RGB → only ③ self-attn |
| RGB-only | learnable mask tokens | learnable mask token | ①②③ run |

## Repo layout

```
configs/                 model.yaml · train.yaml · data.yaml   (every flag in CLAUDE.md)
vistacfusion/
  models/                encoders.py · projection.py · fusion.py · model.py
    heads/               dpt.py · pose.py
  data/                  dataset.py (synthetic stub + sim skeleton) · transforms.py (TactileAugment)
  losses/                depth.py (SSI) · normal.py (cosine) · pose.py (SE2) · total.py
  engine/                train.py (modality dropout) · eval.py (per-config) · metrics.py
  utils/                 config.py · misc.py
tests/                   test_shapes.py · test_overfit.py
scripts/                 train.sh · eval.sh · ablation.sh
```

## Quick start

```bash
pip install -e .          # installs the `vistacfusion` package + deps
pytest                    # shape invariants + one-batch overfit (uses the mock encoder)
```

**End-to-end smoke run with NO DINOv3 weights** — `configs/data.yaml` defaults to
`dataset: synthetic` and `configs/model.yaml` to `encoder.checkpoint: null`, which swaps in a
`MockEncoder` of identical output shape so the whole pipeline runs on CPU:

```bash
bash scripts/train.sh --epochs 1
# or:
python -m vistacfusion.engine.train --epochs 1
```

`evaluate()` reports metrics **per modality config** (both / tactile / rgb) — that table is
the fairness ablation.

## Going to real DINOv3 + real data

1. **Weights** — DINOv3 is gated. Download/convert as in the original notebook
   (`facebook/dinov3-vitl16-pretrain-lvd1689m` → `.pth`), then set
   `encoder.checkpoint` in `configs/model.yaml`. The real `DINOv3Encoder` loads it via
   `torch.hub` (`pretrained=False`) + `load_state_dict(strict=False)`.
2. **Data** — set `dataset: sim` and fill `configs/data.yaml` (`sim.root`, `sim.rgb_subdir`),
   then implement the two `[TODO/USER]` hooks in `vistacfusion/data/dataset.py`:
   the **paired RGB** path convention and the **SE(2) pose GT** loader
   (`SimVisuoTactileDataset._load_pose`). The tactile/depth/normal loading + `TactileAugment`
   are ported from your notebook.

## DPT tap source: v1 (default) vs v2 (CLAUDE.md §3.7)

- `heads.dpt.tap_source: trunk` (**v1, ship first**) — 4 taps are the spatial queries from
  the 4 fusion-trunk layers.
- `heads.dpt.tap_source: encoder_multiscale` (**v2**) — 4 taps from tactile-encoder layers
  with per-tap **residual RGB injection** through the condensed bottleneck (ReZero gate init 0
  → starts as pure encoder taps). When RGB is absent the injection adds **exactly 0** (skipped)
  → single-modality = pure encoder taps, preserving fairness.

Both are built and flag-switchable from day 1, but are **not checkpoint-compatible** (v2 has
extra params) — **retrain to switch**.

## Ablation knobs (CLAUDE.md §9)

`m ∈ {8,16,32}` · `L ∈ {2,3,4}` · `bottleneck_continuity ∈ {reset,carry}` ·
`fusion_variant ∈ {asymmetric,symmetric_coattention}` · `pose_mode ∈ {regression,classification}` ·
`share_encoder_weights` · `tap_source ∈ {trunk,encoder_multiscale}`. See `scripts/ablation.sh`.

## Status

Architecture skeleton complete and verified on CPU with the mock encoder: all shape
invariants hold (identical decoder input across the 3 configs, v1 **and** v2), v2 injection is
provably 0 without RGB, encoders are frozen, and a single batch overfits (depth+normal+pose).
Open `[TODO/USER]` items: DINOv3 checkpoint, sim/real data loaders, paired-RGB convention, and
SE(2) pose GT (CLAUDE.md §12).
