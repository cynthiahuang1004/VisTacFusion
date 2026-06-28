# CLAUDE.md — VisTacFusion (Visuo-Tactile Multi-Task Model)

> Project: **VisTacFusion** · Python package: `vistacfusion`.
> Drop this file at the repo root. Claude Code auto-reads it. It is the single source of
> truth for the architecture. Where something is dataset-specific or undecided it is marked
> **[TODO/USER]** — ask me before guessing.

---

## 0. One-line goal

Build a multi-task model that takes a **paired (but NOT pixel-aligned)** RGB image and a
vision-based tactile image and predicts, in the **tactile frame**:
1. **Dense 3D**: per-pixel **depth** + **surface normal** (DPT head).
2. **Pose**: object **SE(2)** pose — heading θ + planar translation (Pose MLP head).

Trained on **sim** data, deployed on **real** data (sim-to-real). The scientific claim is
that **RGB + tactile beats either modality alone**, proven with a *fair* ablation: one shared
model, identical trainable params, identical decoder input size across all three input configs.

---

## 1. Core design principles (do not violate)

- **Frozen encoders.** Both ViT encoders are frozen feature extractors. Only projection +
  embeddings + bottleneck + fusion trunk + heads are trainable. (This is the sim2real anchor
  and half of the fairness argument.)
- **No pixel alignment assumed.** RGB↔tactile correspondence is learned by cross-attention,
  never by coordinate alignment. Do NOT add any warping / homography / grid-wise add/concat
  fusion.
- **Fixed decoder input.** The decoder always receives `4×(B×196×768)` (spatial) + `B×1×768`
  (pose), in **every** modality config. Query count is fixed (196 spatial + 1 pose = 197);
  only *what fills the queries* and *whether RGB exists* changes.
- **One shared model + modality dropout.** Do not train three separate models. Train one model
  and randomly drop a modality each step. The three configs are inference-time input modes.
- **Tactile is the spatial anchor; RGB is read-only context.** Dense output lives in the
  tactile frame, so tactile patch tokens initialize the spatial queries; RGB only ever serves
  as cross-attention K/V (no `Q_rgb`).

---

## 2. Shapes & symbols (memorize these)

- `B` = batch size. `D = 768` = trunk dim. `m` = number of bottleneck tokens (8–16, ablation).
- `L` = number of fusion-trunk layers (default **4**, to match the DPT's 4 taps).
- Image size `224×224`; DINOv3 patch `16` → grid `14×14` = **196 patch tokens** (+1 CLS).
- Token tensors are `B × N × D` (batch, num_tokens, dim).
- **Attention rule that makes everything work:** output rows = number of **queries**; K/V count
  is independent. This is why fixed query count ⇒ fixed decoder input regardless of modalities.

---

## 3. Pipeline (exact, top to bottom)

### 3.1 Inputs
- RGB: `B×3×224×224`. Tactile: `B×3×224×224`. Paired, not pixel-aligned.
- **[TODO/USER]** tactile preprocessing: raw vs background-subtracted (reference image)?
  Default assume raw 3-channel for DINOv3.

### 3.2 Encoders (frozen, shared architecture)
- **DINOv3 ViT-L/16**, frozen, one instance per modality (RGB encoder, tactile encoder).
  - patch 16, embed_dim 1024, returns **196 patch tokens + 1 CLS** (strip the register tokens).
  - Output per branch: `B×197×1024`.
  - **[DESIGN]** RGB and tactile encoders may share weights or be two copies of the same
    frozen DINOv3. Default: two references to the same frozen weights (saves memory). Make it a
    config flag `share_encoder_weights: true`.
  - **[TODO/USER]** DINOv3 checkpoint source / loader (the user already has a working DINOv3
    `get_intermediate_layers` setup in a prior notebook — reuse it).

### 3.3 Projection + embeddings → D=768
- `Linear(1024 → 768)` per branch, applied per token.
- Add a learned **modality embedding** (one vector per modality, broadcast to all its tokens).
- Add **2D positional embedding** to the 196 spatial tokens (REQUIRED — needed for DPT reshape).
  RGB positional emb is optional.
- Output per branch: `B×197×768`.

### 3.4 Roles
- **RGB → Memory `M`** = `B×197×768` (196 patch + 1 CLS). Read-only K/V.
- **Tactile → Query bank** = `B×197×768`:
  - 196 patch tokens = **spatial queries** `Q_spa`.
  - 1 CLS token = **pose query init** `Q_pos` (CLS-style readout token).
- **Bottleneck tokens** = `m` learnable params (shape `m×768`), broadcast to `B×m×768`. These
  are the *only* cross-modal conduit.

### 3.5 Fusion trunk — repeat `L` (=4) layers
Each layer, in order (each sub-block = attention + residual + LayerNorm; ② and ③ also have an FFN):
1. **① Bottleneck ← RGB** (cross-attn): `Q = bottleneck (m)`, `K=V = M (197)` → bottleneck `B×m×768`.
2. **② Queries ← Bottleneck** (cross-attn): `Q = queries (197)`, `K=V = bottleneck (m)` → queries `B×197×768`.
3. **③ Queries self-attn**: `Q=K=V = queries (197)` → queries `B×197×768`.

- **Tap** the queries output of each of the 4 layers → 4 feature maps for the DPT.
- **[ABLATION A]** bottleneck cross-layer continuity: either reset bottleneck each layer, or
  carry the previous layer's condensed bottleneck forward (recurrent-style). Make this a flag.

### 3.6 Split & heads
- Split the 197 queries → **196 spatial** (→ DPT) + **1 pose** (→ Pose MLP).
- **DPT decoder**: 4 taps × `(B×196×768)` → Reassemble (reshape 196→14×14, the grid is square so
  standard DPT Reassemble works) → FeatureFusion blocks → CNN prediction head.
  - Output: **Depth** `B×1×224×224`, **Normal** `B×3×224×224` (tactile frame).
  - Reuse standard DPT decoder (DPT-Hybrid/Large style); `dpt_features=256`.
- **Pose MLP head**: pose token `B×1×768` → `LayerNorm → Linear(768→256) → GELU → Dropout →
  Linear(256→4)`.
  - Output 4 dims = `(a, b, t_x, t_y)`. Normalize `(a,b)→(cosθ,sinθ)`; `θ=atan2(sinθ,cosθ)`
    only for eval. Output **SE(2)** = `(cosθ, sinθ, t_x, t_y)` = `B×4`. 3 DoF.

---

## 3.7 DPT tap source — v1 (default) vs v2 (ablation)

Config flag: `heads.dpt.tap_source ∈ {trunk, encoder_multiscale}`. **Both paths must be
implemented from the start** (gated by the flag) so switching is config-only. They are **NOT
checkpoint-compatible** (v2 has extra params) — switching ⇒ retrain.

**v1 — `trunk` (default).** The 4 DPT taps are the 196 spatial queries taken from 4
fusion-trunk layers (`fusion_trunk.tap_layers`). Spatial multi-scale comes from the DPT
Reassemble `{4,2,1,0.5}`. Simplest, fully fair — **ship this first**.

**v2 — `encoder_multiscale`.** The 4 DPT taps come from 4 different-depth layers of the
**tactile encoder** (`heads.dpt.encoder_tap_layers`, e.g. `{5,11,17,23}` for ViT-L depth 24),
giving abstraction-level multi-scale like the original DPT. RGB is then *injected* into each
tap (not replaced):
1. Tactile encoder → 4 layers, each `B×196×1024` → per-tap `Linear(1024→768)` (+ 2D pos emb).
2. Run the fusion trunk as usual to obtain the **condensed bottleneck** (step ① condenses RGB)
   and the **pose token**. (The trunk's spatial queries are still computed and still feed the
   pose token via ②③; in v2 they simply no longer feed the DPT.)
3. Per-tap **residual RGB injection**:
   `tap_i = tap_i + g · CrossAttn(Q=tap_i, K=V=condensed_bottleneck)`,
   where `g` is a learnable ReZero-style gate (`inject_gate_init=0`).
   **When RGB is absent, skip the cross-attn (add exactly 0)** → the tap is the pure encoder
   feature.
4. DPT consumes the 4 enriched taps. Pose head reads the pose token (unchanged from v1).

**Why v2 stays fair (critical):** the per-tap injection modules ALWAYS exist (same params in
all three configs). With RGB they add RGB context; without RGB they add exactly 0 (skipped), so
single-modality = pure encoder taps and both = encoder taps + RGB. No special-casing for the
single-modality path, and "RGB only via bottleneck / no pixel alignment" still holds. Decoder
input shape stays `4×(B×196×768) + (B×1×768)` in every config.

**Cost vs v1:** +4 tap projections and +4 injection cross-attn blocks (more trainable params),
and v2 must be retrained (not v1-checkpoint-compatible). **Build v1 first, validate the
skeleton, then flip the flag for v2 and retrain.**

---

## 4. The three configs (modality dropout)

One shared model. Decoder input is identical in all three: `4×(B×196×768) + (B×1×768)`.

| Config | spatial queries (196) | pose query (1) | bottleneck / cross-modal |
|---|---|---|---|
| **RGB+tactile** | tactile patch tokens | tactile CLS | ①②③ run (condense RGB) |
| **tactile-only** | tactile patch tokens | tactile CLS | no RGB → bottleneck idle, **① skipped**, only ③ self-attn |
| **RGB-only** | **learnable mask tokens** (196) | **learnable mask token** (1) | ①②③ run (condense RGB) |

- Mask tokens are learned params (one shared 196-set + one pose mask), broadcast per sample,
  carrying the same 2D positional emb as the real spatial queries.
- Train with **modality dropout**: each step sample config ∈ {both, tactile-only, RGB-only}
  with tunable probabilities (e.g. 0.5 / 0.25 / 0.25). Keep dense-task configs sensible per
  **[TODO/USER]** (RGB-only dense may be intentionally weak — see §8).

---

## 5. Losses

- **Depth**: scale-shift-invariant (SSI / MiDaS-style) loss on normalized depth (robust for
  sim2real). Optional gradient-matching term for sharp edges. (A plain L1/MSE fallback is fine
  to start.)
- **Normal**: cosine / angular loss `1 − cos(angle)` (not MSE on raw vectors).
- **Pose**: rotation `1 − cos(θ_pred − θ_gt)` computed on `(cosθ,sinθ)` (no atan2 in the loss);
  translation L1 on `(t_x,t_y)`. `L_pose = L_rot + λ·L_trans`.
- **Total**: weighted sum; start with hand-tuned λ, optionally Kendall uncertainty weighting.
- **[TODO/USER]** confirm GT formats & frames: depth/normal in tactile frame; SE(2) pose
  convention (which axis is θ, what (t_x,t_y) frame).

---

## 6. Training / data

- **[TODO/USER]** sim dataset format + loader. Expected per-sample: RGB image, tactile image,
  GT depth (tactile frame), GT normal (tactile frame), GT SE(2) pose. Provide a `Dataset` stub.
- **Augmentation / domain randomization** is the sim2real workhorse: keep heavy tactile augment
  (gain/bias/gradient/residual/noise + geometric) and standard RGB photometric aug. (User has a
  `TactileAugment` from a prior notebook — port it.)
- Encoders frozen; train everything else. Optimizer AdamW, cosine schedule. Mixed precision.
- Optionally fine-tune trunk+heads on a small amount of real data; report zero-shot too.

---

## 7. Suggested repo structure

```
VisTacFusion/                 # repo root
  CLAUDE.md                 # this file
  README.md
  requirements.txt
  pyproject.toml            # installs the `vistacfusion` package (pip install -e .)
  configs/
    model.yaml              # D, m, L, share_encoder_weights, bottleneck_continuity, tap_source, ...
    train.yaml              # lr, schedule, modality_dropout probs, loss weights
    data.yaml               # paths, image size, aug
  vistacfusion/             # the Python package (import vistacfusion)
    models/
      encoders.py           # frozen DINOv3 wrapper(s); returns patch tokens + CLS
      projection.py         # Linear 1024->768 + modality emb + 2D pos emb
      fusion.py             # Bottleneck tokens + 3-step layer (①②③), repeat L, tap 4
      heads/
        dpt.py              # Reassemble + FeatureFusion + CNN head -> depth, normal (v1/v2 tap_source)
        pose.py             # SE(2) MLP head -> (cos,sin,tx,ty)
      model.py              # full pipeline; handles 3 configs + modality dropout; forward()
    data/
      dataset.py            # [TODO/USER] sim dataset
      transforms.py         # tactile augment + RGB aug
    losses/
      depth.py normal.py pose.py total.py
    engine/
      train.py eval.py metrics.py
    utils/
  scripts/
    train.sh eval.sh ablation.sh
  tests/
    test_shapes.py          # asserts decoder input identical across the 3 configs
```

- `model.py::forward(rgb, tactile, config)` must support `config ∈ {"both","tactile","rgb"}`
  and produce identical-shaped decoder inputs in all three.

---

## 8. Implementation gotchas (read before coding)

1. **DPT Reassemble assumes a square token grid.** Keep the canonical spatial grid square
   (14×14=196). Do not feed a non-square count.
2. **Strip register tokens, keep CLS separately.** DINOv3 returns CLS + registers + patches;
   use patches for the grid, CLS for pose init / memory.
3. **Attention with `n_q ≠ n_kv` is normal.** Don't force equal counts; output length follows
   queries. This is the whole point.
4. **`L=4` matches the 4 DPT taps.** If you want `L≠4`, make the number of taps configurable
   (tap evenly spaced layers, or tap sub-steps within layers).
5. **Pose: never put a raw angle through the loss.** Predict `(cosθ,sinθ)` (after L2-normalizing
   the 2 outputs); loss in `(cos,sin)` space. `atan2` only for the reported metric.
6. **Modality dropout must preserve decoder input shape.** In RGB-only, swap real tactile
   queries for learnable mask tokens — do NOT change counts. In tactile-only, skip ① (no RGB)
   but keep ③; queries still 196+1.
7. **RGB-only dense is expected to be weak** (predicting tactile-frame contact geometry without
   touch). **[TODO/USER]** decide whether RGB-only is in the dense ablation at all, or only in
   the pose ablation. The architecture supports either; just gate it in the dropout sampler.
8. **Bottleneck/mask tokens are shared params broadcast per sample** (like a CLS token), not
   per-image learned.
9. **v2 only — the per-tap RGB injection MUST add 0 when RGB is absent** (skip the cross-attn,
   keep the residual). This is what keeps single-modality = pure encoder taps and preserves
   fairness. The injection modules still exist as params in every config; they just contribute 0
   without RGB. Init the gate at 0 (`inject_gate_init=0`) so v2 starts as pure encoder taps and
   learns to inject.

---

## 9. Ablation knobs (wire these as config flags from day 1)

- `m` ∈ {8, 16, 32} (bottleneck width).
- `L` ∈ {2, 3, 4} (trunk depth). Sweep `L × m`; stop on **real**-data gains, not sim val.
- `bottleneck_continuity` ∈ {reset, carry} (ablation A).
- `fusion_variant` ∈ {asymmetric (default), symmetric_coattention} — symmetric makes RGB also
  produce queries (`Q_rgb` appears); it's the control that justifies "tactile as anchor".
- `pose_mode` ∈ {regression (default), classification} — bin θ into classes if regression is
  unstable.
- `share_encoder_weights` ∈ {true, false}.
- `tap_source` ∈ {trunk (v1), encoder_multiscale (v2)} — DPT taps from fusion-trunk layers vs
  tactile-encoder layers + per-tap RGB injection. Both implemented from day 1; flag-switchable;
  **retrain to switch** (not checkpoint-compatible). See §3.7.

---

## 10. Build order (milestones for Claude Code)

1. Scaffold repo + configs + `requirements.txt`. Stub `Dataset` with synthetic random tensors
   so the model runs end-to-end before real data exists.
2. `encoders.py` (frozen DINOv3 wrapper) → verify output `B×197×1024`.
3. `projection.py` → `B×197×768` each; add embeddings.
4. `fusion.py` (bottleneck + ①②③ + repeat L + 4 taps) → verify `4×(B×196×768)` + `B×1×768`.
5. `heads/dpt.py` and `heads/pose.py` → verify final output shapes.
6. `model.py` wiring all three configs + a unit test asserting identical decoder input shapes.
7. Losses + a one-batch overfit test (loss → 0 on a single synthetic batch).
8. `engine/train.py` + modality dropout + metrics; then plug in the real `Dataset`.

Each step: add a tiny shape-assertion test. Get shapes right before training logic.

---

## 11. References (related work the design draws on)

- **NeuralFeels** (Suresh et al., *Science Robotics* 2024) — visuo-tactile pose+shape, tactile
  refines/disambiguates under occlusion.
- **T3** (Zhao et al., CoRL 2024) — modular per-sensor encoder + shared trunk + task heads.
- **Perceiver IO** (Jaegle et al., ICLR 2022) — fixed query set cross-attends variable memory.
- **MQTransformer** (Xu et al., IEEE TCSVT 2023) — task queries for multi-task dense prediction.
- **MBT / Attention Bottlenecks** (Nagrani et al., NeurIPS 2021) — cross-modal flow through a
  few bottleneck tokens (the fusion trunk's core idea).
- **TokenFusion** (Wang et al., CVPR 2022) — alternative fixed-token-count fusion.
- **Ma et al.** (CVPR 2022) "Are Multimodal Transformers Robust to Missing Modality?";
  **ShaSpec** (Wang et al., CVPR 2023); **UNIC** (Xu & Shirai, ICRA 2026) — modality dropout /
  mask tokens / robustness to missing modality.
- DPT (Ranftl et al., ICCV 2021) for the dense head; 6D/continuous rotation representation
  (Zhou et al., CVPR 2019) specialized here to SE(2) `(cosθ,sinθ)`.

---

## 12. Open questions to confirm with the user before/while building

- [ ] Sim data format + GT (depth/normal frame, SE(2) pose convention & frame).
- [ ] DINOv3 checkpoint / loader (reuse prior notebook).
- [ ] Tactile preprocessing (raw vs background-subtracted).
- [ ] Is RGB-only included in the **dense** ablation, or pose-only?
- [ ] Default `m`, `L`, modality-dropout probabilities for the first run.
