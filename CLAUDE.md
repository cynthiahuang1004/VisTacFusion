# CLAUDE.md — VisTacFusion (Visuo-Tactile Multi-Task Model)

> Package: `vistacfusion` · Python 3.11 · PyTorch 2.5.1+cu121 · Conda env: `vistacfusion`

---

## 0. Goal

End-to-end multi-task model: paired (non-pixel-aligned) RGB + tactile image →
**depth + surface normal** (DPT head, tactile frame) + **SE(2) pose** (Pose MLP head).
Trained on sim, deployed on real (sim-to-real). RGB provides **pose disambiguation**
(object identity + orientation context), not dense geometry improvement.

---

## 1. Architecture (v3 — decoupled dense/pose paths)

Two fully decoupled data paths share the same frozen encoder:

- **DPT path**: encoder multiscale taps at **native 1024 dim** → DPT decoder.
  Receives tactile taps whenever tactile is present (both/tactile configs, 90% of
  steps); in the `rgb` config it falls back to RGB encoder taps (same ViT-L token
  geometry) and is still supervised. Optional RGB injection
  via ReZero-gated cross-attention through the fusion trunk's bottleneck.
- **Pose path**: encoder → projection (1024→768) → fusion trunk (asymmetric
  cross-attention bottleneck) → Pose MLP. Modality dropout (both/tactile/rgb)
  only affects this path.

### Complete tensor flow

```
INPUT: rgb [B, 3, 224, 224], tactile [B, 3, 224, 224]

═══ ENCODER (frozen DINOv3 ViT-L/16, shared weights) ═══
tactile → tac_patch [B, 196, 1024], tac_cls [B, 1, 1024]
rgb     → rgb_patch [B, 196, 1024], rgb_cls [B, 1, 1024]
tactile → multiscale taps: 4 × [B, 196, 1024] at layers {5, 11, 17, 23}

═══ POSE PATH ═══
rgb_proj(rgb_patch ∥ rgb_cls) + mod_emb        → pose_memory  [B, 197, 768]
tactile_proj(tac_patch) + spatial_pos + mod_emb → spatial_q    [B, 196, 768]
tactile_proj(tac_cls) + mod_emb                 → pose_q       [B, 1, 768]
concat(spatial_q, pose_q)                       → pose_queries [B, 197, 768]
  (+= obj_embedding[object_ids] if provided)

FusionTrunk × 4 layers (D=768, m=16 bottleneck, 8 heads):
  ① bottleneck ← cross-attn(Q=bottleneck [B,16,768], KV=memory [B,197,768])
  ② queries   ← cross-attn(Q=queries [B,197,768], KV=bottleneck [B,16,768]) + FFN
  ③ queries   ← self-attn(queries [B,197,768]) + FFN
  → tap_i = queries[:, :196]  [B, 196, 768]   (4 taps, one per layer)
  → pose_token = queries[:, -1:]  [B, 1, 768]  (after final layer)

PoseHead:
  pool = cat(pose_token.squeeze [B,768], mean(spatial_q) [B,768]) → [B, 1536]
  MLP: 1536 → 512 → 256 → 4
  → se2 [B, 4] = (cos θ, sin θ, tx, ty)

═══ DPT PATH ═══
4 encoder multiscale taps [B, 196, 1024] + dpt_pos [1, 196, 1024]
  + TapInjection (if RGB present):
      cross-attn(Q=tap [B,196,1024], KV=bottleneck [B,16,768]) × gate(init=0) + tap

DPTHead:
  Reassemble: 4 × [B,196,1024] → [B,256,56²], [B,256,28²], [B,256,14²], [B,256,7²]
  FeatureFusion (coarse→fine, each 2× upsample) → [B, 256, 112, 112]
  depth_head: Conv 256→128→32→1, 2× upsample → [B, 1, 224, 224]
  normal_head: Conv 256→128→32→3, 2× upsample → [B, 3, 224, 224]

OUTPUT: {depth: [B,1,224,224], normal: [B,3,224,224], se2: [B,4]}
```

---

## 2. Key dimensions

| Symbol | Value | Meaning |
|---|---|---|
| B | variable | Batch size |
| E | 1024 | DINOv3 encoder dim |
| D | 768 | Fusion trunk dim |
| N | 196 | Patch tokens (14×14) |
| m | 16 | Bottleneck tokens |
| L | 4 | Fusion trunk layers |
| image_size | 224 | Input resolution |
| patch_size | 16 | ViT patch size |
| dpt_features | 256 | DPT internal channel width |
| num_heads | 8 | Trunk attention heads |
| ffn_hidden | 3072 | Trunk FFN dim (D × 4) |

---

## 3. Modules detail

### 3.1 Encoder (`models/encoders.py`)

DINOv3 ViT-L/16, frozen. Checkpoint: `weights/dinov3_vitl16_pretrain_lvd1689m.pth`.
- 24 layers, 1024 dim, 16 heads (head_dim=64)
- Token layout: `[CLS, registers, 196 patches]` → strip registers
- `forward(x)` → `(patch [B,196,1024], cls [B,1,1024])`
- `forward_multiscale(x)` → 4 taps at layers `[5,11,17,23]`, each `[B,196,1024]`
- `share_encoder_weights: true` → single nn.Module instance for both modalities

### 3.2 Projection (`models/projection.py`)

**BranchProjection** (one per modality):
- `Linear(1024, 768)` + learned modality embedding `[1,1,768]`

**SpatialPosEmbedding**:
- `spatial_pos [1,196,768]` — added to tactile spatial queries entering trunk
- `dpt_pos [1,196,1024]` — added to encoder taps entering DPT

### 3.3 Fusion trunk (`models/fusion.py`)

**FusionTrunkLayer** — 3-step per layer:
- Step ①: cross-attn (Q=bottleneck m=16, KV=RGB memory 197), no FFN
- Step ②: cross-attn (Q=queries 197, KV=bottleneck 16) + FFN
- Step ③: self-attn (Q=K=V=queries 197) + FFN

**AttentionBlock**: pre-norm MHA (8 heads, dropout=0.1) + residual. FFN = Linear(768→3072) → GELU → Linear(3072→768).

**Bottleneck**: learned `[1,16,768]`, `trunc_normal_(std=0.02)`. Mode: `carry` (across layers).

**Per-config behavior**:
- `"both"`: all 3 steps run
- `"tactile"`: only step ③ (bottleneck idle, no RGB)
- `"rgb"`: all 3 steps, queries are learnable mask tokens

### 3.4 TapInjection (`models/model.py`)

Per-tap RGB injection into DPT path via cross-attention with ReZero gate:
- Q from encoder tap (1024), K/V from trunk bottleneck (768→1024 projection)
- 8 heads, head_dim=128
- `gate`: scalar param, init=0.0. Output: `tap + gate × cross_attn(tap, bottleneck)`
- When RGB absent: injection skipped, output = pure encoder tap

### 3.5 DPT head (`models/heads/dpt.py`)

Standard DPT decoder at `embed_dim=1024`, `features=256`:
- **Reassemble**: 4 taps → `Conv2d(1024→256)` + bilinear scale `{4×, 2×, 1×, 0.5×}` → `[56², 28², 14², 7²]`
- **FeatureFusion**: coarse-to-fine merge, each block has 2× ResidualConvUnit + 2× upsample
- **Prediction heads**: `Conv(256→128,3) → ReLU → Upsample2× → Conv(128→32,3) → ReLU → Conv(32→C,1)`
  - Depth: C=1, Normal: C=3

### 3.6 Pose head (`models/heads/pose.py`)

Input: `pose_token [B,1,768]` + `spatial_queries [B,196,768]` (mean pooled)
- Concat → `[B, 1536]`
- **Regression mode**: `LN → Linear(1536→512) → GELU → Drop → Linear(512→256) → GELU → Drop → Linear(256→4)`
  - First 2 outputs L2-normalized → `(cos θ, sin θ)`, last 2 → `(tx, ty)`
- **Classification mode**: rotation binned to 72 classes (5° bins), soft-argmax → `(cos, sin)`. Translation separate MLP → `(tx, ty)`
- Output: `se2 [B,4] = (cos θ, sin θ, tx, ty)`

### 3.7 Object embedding

`nn.Embedding(num_objects=20, 768)`. Added to all 197 pose query tokens.
Provides object identity to help pose disambiguation.

---

## 4. Modality dropout (training)

One model, three inference configs. Dropout probabilities: `p_both=0.55, p_tactile=0.35, p_rgb=0.10`.

| Config | Pose queries (197) | Pose memory | DPT taps | RGB injection |
|---|---|---|---|---|
| `both` | tactile-derived | RGB-derived | tactile encoder | Yes (via bottleneck) |
| `tactile` | tactile-derived | None (①② skipped) | tactile encoder | No |
| `rgb` | learned mask tokens | RGB-derived | RGB encoder (fallback)* | Yes |

*DPT uses tactile encoder taps whenever tactile is present; in the `rgb` config (`use_tactile=False`) `model.py` falls back to `rgb_encoder.forward_multiscale` because T3 and MAE share dim/token count, and the depth loss is still applied. Modality dropout otherwise only affects the pose path. RGB injection into DPT is independently sampled (`p_dpt_inject=0.5`).

---

## 5. Losses (`losses/total.py`)

**Grouped Kendall uncertainty weighting**: dense group (depth+normal) and pose group (rot+trans)
auto-balance independently. Fixed `dense_pose_ratio=1.0` between groups.

| Loss | Formula | Group |
|---|---|---|
| Depth | MSE | dense |
| Normal | MSE | dense |
| Pose rot | 1 − cos(θ_pred − θ_gt) via (cos,sin) | pose |
| Pose trans | L1 on (tx, ty) | pose |

---

## 6. Data

### Sim data
- Root: `/media/hdd2/ihsuan/gs_blender/renders` (20 objects, 12 sessions each)
- Meshes: `/media/hdd2/ihsuan/gs_blender/meshes`
- Per sample: tactile PNG + RGB PNG + depth `_gt.npy` + `_pose.json`
- Train/val: every 20th sample = val (`val_every: 20`)
- Augmentation: tactile photometric (gain/bias/noise) + RGB photometric + gel-spin rotation (±180°) + fixed 1/√2 center crop

### Real data
- Root: `/media/hdd2/ihsuan/gs_blender/real_data` (5 objects, session_000 only)
- Per-sample rotation in `rotation_euler[2]` (not session-level). Translation = 0 (centered).

### Pose GT
- `delta_rz = rotation_euler[2] - rz0` (per-sample, works for both sim and real)
- `(cos_rz, sin_rz, x_norm, y_norm)` where `x_norm = R(delta_rz) @ (sx, sy) / half`

---

## 7. Training recipe

| Parameter | Value |
|---|---|
| Optimizer | AdamW, lr=2e-4, weight_decay=0.05 |
| Schedule | Cosine + 1000-step warmup, 150 epochs |
| Batch size | 64 per GPU (2 GPU DDP → effective 128) |
| AMP | fp16 mixed precision |
| Trainable params | ~95M (total ~398M with frozen encoder) |

```bash
# Training
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
  -m vistacfusion.engine.train \
  --model configs/model.yaml --train configs/train.yaml --data configs/data.yaml \
  --output-dir outputs/run_name

# Evaluation
python -m vistacfusion.engine.inference \
  --train-dir outputs/run_name --eval-sim --device cuda:0
```

---

## 8. Repo structure

```
VisTacFusion/
  CLAUDE.md
  configs/
    model.yaml                # architecture config
    train.yaml                # training config (loss, dropout, optimizer)
    train_finetune.yaml       # fine-tune config (lr=2e-5, --finetune flag)
    data.yaml                 # data paths, augmentation, split
  vistacfusion/
    models/
      encoders.py             # DINOv3Encoder (frozen)
      projection.py           # BranchProjection, SpatialPosEmbedding
      fusion.py               # FusionTrunk, FusionTrunkLayer, AttentionBlock
      heads/
        dpt.py                # DPTHead (Reassemble + FeatureFusion + CNN)
        pose.py               # PoseHead (classification or regression)
      model.py                # VisuoTactileModel + TapInjection
    data/
      dataset.py              # SimVisuoTactileDataset, SyntheticDataset
      transforms.py           # TactileAugment, rotate_gel_spin, center_crop
    losses/
      total.py                # MultiTaskLoss (grouped uncertainty)
      depth.py normal.py pose.py
    engine/
      train.py                # DDP training loop
      eval.py                 # precompute_encoder_cache, evaluate
      inference.py            # run_eval_sim, run_real_data_tree, run_single
  weights/                    # DINOv3 checkpoint (frozen)
  outputs/                    # training outputs (checkpoints, history, plots)
  eval_results/               # evaluation outputs (metrics, visualizations)
```

---

## 9. Design principles

1. **Frozen encoders** — sim2real anchor; only projection + trunk + heads are trainable.
2. **No pixel alignment** — RGB↔tactile correspondence learned via cross-attention bottleneck.
3. **Tactile is spatial anchor** — dense prediction in tactile frame; RGB is read-only K/V context.
4. **One model, three configs** — modality dropout at training; any config at inference.
5. **Decoupled DPT/Pose paths** — DPT gets tactile taps on every step tactile is present (no dropout dilution); RGB-only steps fall back to RGB taps.
6. **Grouped uncertainty weighting** — dense and pose auto-balance independently.

---

## 10. Comparison with baselines

### SparshXTwoStreamFusion baseline (`/media/hdd2/ihsuan/SparshXTwoStreamFusion`)
Same encoder (frozen DINOv3), same task heads (DPT + PoseHead), same data/loss/training.
Fusion: **symmetric shared-bottleneck averaging** (MBT, Nagrani et al. NeurIPS 2021).

| Component | VisTacFusion | SparshX baseline |
|---|---|---|
| Fusion | Asymmetric cross-attn bottleneck | Symmetric MBT (bottleneck average) |
| DPT taps | Encoder multiscale @1024 | Post-fusion tactile @768 |
| Pose readout | CLS token (dedicated) | Mean pool (all tactile tokens) |
| Pre-fusion refinement | None | 4 independent self-attn blocks per stream |
| Modality dropout | Yes (both/tactile/rgb) | No |

### Ablation knobs
- `fusion_variant`: asymmetric (default) vs symmetric_coattention
- `bottleneck_continuity`: carry (default) vs reset
- `m` (bottleneck width): 8, 16, 32
- `L` (trunk depth): 2, 3, 4
- `object_embedding`: true/false
- `share_encoder_weights`: true/false
- `pose_mode`: regression vs classification
