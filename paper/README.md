# Paper Data Provenance

All experimental results in the paper. For each table, this documents the exact source files (history.json or train.log) and how the numbers were obtained.

## Storage locations

| Disk | Path | Status | Contents |
|---|---|---|---|
| `/media/hdd` | `/media/hdd/ihsuan/VisTacFusion_outputs/` | **FULL (0 avail)** — read-only | Old dropout encoder ablation (150 ep each), ratio ladder, baselines |
| `/media/hdd2` | `/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2/` | 1.1 TB free | Moddrop ablation, new dropout re-runs (in progress), seed1 variants |

**Symlinks**: `outputs/` in the repo root contains symlinks to both disks.

**Important**: On 2026-09-08, `history.json` and `latest.pt` were accidentally deleted from many runs on hdd2 by a cleanup script. All `history.json` files were recovered from `train.log` using regex parsing. Checkpoint files (`latest.pt`, `best_pose.pt`) for moddrop runs are **lost** — only `best_depth.pt` survives. The recovered `history.json` files contain identical metrics data.

---

## Current state (2026-09-08)

- **Old dropout** (0.55/0.35/0.10 @ inject 0.5): All 26 encoder ablation runs complete on hdd (150 ep). These are the numbers currently in the paper tables.
- **New dropout** (0.70/0.25/0.05 @ inject 0.75): Re-running all 26 encoder ablation runs on hdd2. 2/26 done, 6 in progress. When complete, all paper tables will be updated.
- **Moddrop ablation**: 22 runs complete on hdd2. history.json recovered from train.log.
- **Ratio ladder**: 14 points complete on hdd.

---

## Table I: Model Comparison (three inference modes)

The main comparison table. All models trained with sim+real co-training at ratio 1:1.27 (sim148), crop 0.816, zoom 1.15, transfilt.

### Data config
- `ablation/simqty_gtac/data_ratio_g3s_sim148_transfilt_zoom115_crop816.yaml`

### Multi-encoder fusion models (†)

| Model | Train config | Model config | History source | Notes |
|---|---|---|---|---|
| **VisTacFusion (Ours)** | `configs/train_bs32.yaml` | `ablation/encoder/tac_sitr_rgb_mae.yaml` | `hdd/ablation_c816_sim148_sitr_mae/history.json` | Will be re-run with new dropout |
| TVL + MAE | `configs/train_bs32.yaml` | `ablation/encoder/tac_tvl_rgb_mae.yaml` | `hdd/ablation_c816_sim148_tvl_mae/history.json` | |
| Sparsh-X + MAE | `configs/train_bs32.yaml` | `ablation/encoder/tac_sparshx_rgb_mae.yaml` | `hdd/ablation_c816_sim148_sparshx_mae/history.json` | Uses SparshX repo code |

### ResNet baselines

| Model | Train config | Model config | History source | Notes |
|---|---|---|---|---|
| ViTaL (ResNet) | `configs/train_bs32.yaml` | (integrated in codebase) | `hdd/ablation_c816_sim148_vital/history.json` | CLIP-pretrained ResNet-18 + DPT |
| MViTac | `configs/train_bs32.yaml` | (integrated in codebase) | `hdd/ablation_c816_sim148_mvitac/history.json` | Pose-only (no dense prediction) |

### Single-encoder baselines (tactile-only and visual-only modes)

These use the same fusion architecture but are evaluated in single-modality mode. Each encoder's row in the tactile/visual sections comes from the corresponding encoder ablation run.

| Encoder | History source | Tactile mode | Visual mode |
|---|---|---|---|
| SiTR | `hdd/ablation_c816_sim148_sitr_mae/` | `val.tactile` | `val.rgb` |
| T3 | `hdd/ablation_c816_sim148_t3_mae/` | `val.tactile` | `val.rgb` |
| DINOv3 | `hdd/ablation_c816_sim148_dinov3_mae/` | `val.tactile` | `val.rgb` |
| DAv2 | `hdd/ablation_c816_sim148_dav2_mae/` | `val.tactile` | `val.rgb` |
| MAE | `hdd/ablation_c816_sim148_sitr_mae/` | — | `val.rgb` (MAE is the visual encoder) |
| CLIP | `hdd/ablation_c816_sim148_sitr_clip/` | — | `val.rgb` |
| SigLIP | `hdd/ablation_c816_sim148_sitr_siglip/` | — | `val.rgb` |
| Sparsh-X | `hdd/ablation_c816_sim148_sparshx_mae/` | `val.tactile` | — |
| Sparsh-DINOv2 | `hdd/ablation_c816_sim148_sparshv2_mae/` | `val.tactile` | — |
| Sparsh-MAE | `hdd/ablation_c816_sim148_sparshmae_mae/` | `val.tactile` | — |
| TVL | `hdd/ablation_c816_sim148_tvl_mae/` | `val.tactile` | — |

### How to extract numbers

```python
import json
h = json.load(open("history.json"))
# Best epoch by pose rotation (both mode):
best = min(h, key=lambda e: e["val"]["both"]["pose_rot_l1"])
# Metrics: depth_mse, normal_mse, pose_rot_l1, pose_rot_deg, pose_trans
print(best["val"]["both"])   # fused mode
print(best["val"]["tactile"]) # tactile-only mode
print(best["val"]["rgb"])     # visual-only mode
```

### Metric definitions

| Paper column | history.json key | Formula |
|---|---|---|
| Depth MSE | `depth_mse` | MSE over contact region |
| Normal MSE | `normal_mse` | MSE of surface normal vectors |
| Rot L1 | `pose_rot_l1` | 1 - cos(Δθ) (via cos/sin dot product) |
| Rot (°) | `pose_rot_deg` | arccos(cos·cos* + sin·sin*) in degrees |
| Trans L1 | `pose_trans` | L1 on normalized (tx, ty) |

### Epoch selection

All results use the **best both-mode epoch** (minimum `val.both.pose_rot_l1`). The same epoch's metrics are reported for all three modes. This is implemented in `scripts/plot_ratio_ladder.py::load_best_per_mode()`.

---

## Table II: Architecture Comparison (VisTacFusion vs Sparsh-X MBT)

Head-to-head comparison of asymmetric bottleneck (ours) vs symmetric MBT fusion, using identical SiTR+MAE encoders.

| Model | History source |
|---|---|
| VisTacFusion | `hdd/ablation_c816_sim148_sitr_mae/history.json` |
| Sparsh-X (MBT) | `hdd2/ablation_c816_sim148_sparshx_mae/history.json` or `hdd/ablation_c816_sim348_sparshx_mae/` |

The Sparsh-X baseline uses the SparshXTwoStreamFusion codebase (`/media/hdd2/ihsuan/SparshXTwoStreamFusion`) with model config `vtf_model_sparshx_sitrmae.yaml` (SiTR tactile + MAE visual, MBT fusion).

---

## Table III: Encoder Ablation (6 tactile × MAE)

All rows use MAE ViT-L as the visual encoder. Only the tactile encoder varies.

| Tactile encoder | Model config | History source |
|---|---|---|
| T3 Large | `ablation/encoder/tac_t3_rgb_mae.yaml` | `hdd/ablation_c816_sim148_t3_mae/` |
| DINOv3 ViT-L | `ablation/encoder/tac_dinov3_rgb_mae.yaml` | `hdd/ablation_c816_sim148_dinov3_mae/` |
| Sparsh-DINOv2-B | `ablation/encoder/tac_sparshv2_rgb_mae.yaml` | `hdd/ablation_c816_sim148_sparshv2_mae/` |
| Sparsh-MAE-B | `ablation/encoder/tac_sparshmae_rgb_mae.yaml` | `hdd/ablation_c816_sim148_sparshmae_mae/` |
| SiTR-B18 | `ablation/encoder/tac_sitr_rgb_mae.yaml` | `hdd/ablation_c816_sim148_sitr_mae/` |
| DAv2 ViT-L | `ablation/encoder/tac_dav2_rgb_mae.yaml` | `hdd/ablation_c816_sim148_dav2_mae/` |

### Full 6×4 grid (not in paper, but available)

The full encoder grid crosses 6 tactile encoders × 4 visual encoders (MAE, DINOv3, CLIP, SigLIP). All 24 combinations are in `hdd/ablation_c816_sim148_{tac}_{rgb}/`. The paper shows only the MAE column because MAE dominates across all tactile encoders.

Additional special encoders: `tvl_mae`, `sparshx_mae` (in separate rows).

### Visualization scripts

- `scripts/plot_ratio_ladder.py` — ratio ladder figures + encoder comparison
- `scripts/gen_encoder_ablation_configs.py` — generates all `ablation/encoder/tac_*_rgb_*.yaml` configs

---

## Table IV: Sim-to-Real Data Mixing (Ratio Ladder)

10 ratio points from real-only to 1:6.52, all using SiTR+MAE.

| Ratio | Run name | History source |
|---|---|---|
| 1:0 (real-only) | `ratio_g3s_realonly_transfilt_zoom115_crop816` | `hdd/` |
| 1:0.30 | `ratio_g3s_sim34_transfilt_zoom115_crop816` | `hdd/` |
| 1:0.58 | `ratio_g3s_sim68_transfilt_zoom115_crop816` | `hdd/` |
| 1:1.04 | `ratio_g3s_sim120_transfilt_zoom115_crop816` | `hdd/` |
| 1:1.27 ★ | `ratio_g3s_sim148_transfilt_zoom115_crop816` | `hdd/` |
| 1:1.63 | `ratio_g3s_sim190_transfilt_zoom115_crop816` | `hdd/` |
| 1:1.97 | `ratio_g3s_sim229_transfilt_zoom115_crop816` | `hdd/` |
| 1:2.98 | `ratio_g3s_sim348_transfilt_zoom115_crop816` | `hdd/` |
| 1:4.89 | `ratio_g3s_sim570_transfilt_zoom115_crop816` | `hdd/` |
| 1:6.52 | `ratio_g3s_sim760_transfilt_zoom115_crop816` | `hdd/` |

★ = selected ratio for all main table comparisons.

### Ratio naming convention

`simXXX` = XXX simulated sessions. Real data is fixed at 4197 train samples. The real:sim ratio is `4197 / (XXX × 12 objects)`.

### Data configs

Each ratio has its own data config in `ablation/simqty_gtac/data_ratio_g3s_simXXX_transfilt_zoom115_crop816.yaml`, varying `train_samples_per_session` and/or `sim_oversample`.

---

## Modality Dropout Ablation (in-text, Section IV-E)

24 configurations testing dropout probabilities and DPT injection rate. All use SiTR+MAE at sim148.

### Settings and sources

| Setting | Train config | Run name(s) | Seeds |
|---|---|---|---|
| Default (0.55/0.35/0.10 @ inject 0.5) | `configs/train_bs32.yaml` | `sitr_mae`, `sitr_mae_seed1` | 2 |
| inject 0 | `ablation/moddrop/train_bs32_md_inject0.yaml` | `sitr_mae_md_inject0`, `_seed1` | 2 |
| inject 0.25 | `ablation/moddrop/train_bs32_md_inject025.yaml` | `sitr_mae_md_inject025`, `_seed1` | 2 |
| inject 0.75 | `ablation/moddrop/train_bs32_md_inject075.yaml` | `sitr_mae_md_inject075`, `_seed1` | 2 |
| inject 1.0 | `ablation/moddrop/train_bs32_md_inject1.yaml` | `sitr_mae_md_inject1`, `_seed1` | 2 |
| 0.80/0.15/0.05 | `ablation/moddrop/train_bs32_md_both80.yaml` | `sitr_mae_md_both80`, `_seed1` | 2 |
| 0.55/0.45/0 (no rgb) | `ablation/moddrop/train_bs32_md_norgb.yaml` | `sitr_mae_md_norgb`, `_seed1` | 2 |
| 0.30/0.60/0.10 | (tac heavy variant) | `sitr_mae_md_tacheavy`, `_seed1` | 2 |
| No dropout | `ablation/moddrop/train_bs32_md_none.yaml` | `sitr_mae_md_none` | 1 |
| No dropout + inject 0.5 | `ablation/moddrop/train_bs32_md_none_inj05.yaml` | `sitr_mae_md_none_inj05` | 1 |
| 0.34/0.33/0.33 (uniform) | `ablation/moddrop/train_bs32_md_uniform.yaml` | `sitr_mae_md_uniform` | 1 |
| 0.40/0.30/0.30 | (rgb heavy variant) | `sitr_mae_md_rgbheavy` | 1 |
| 0.55/0.45/0 (no tactile) | `ablation/moddrop/train_bs32_md_notac.yaml` | `sitr_mae_md_notac` | 1 |
| 0.90/0/0.10 (no tactile-only) | — | `sitr_mae_md_notac` (different naming) | 1 |
| **0.70/0.25/0.05 @ inject 0.75** ★ | `ablation/moddrop/train_bs32_md_b70_inj075.yaml` | `sitr_mae_md_b70_inj075`, `_seed1` | 2 |
| 0.70/0.25/0.05 @ inject 1.0 | `ablation/moddrop/train_bs32_md_b70_inj1.yaml` | `sitr_mae_md_b70_inj1` | 1 |

★ = selected as new default. Both-mode depth −35%, normal −22% vs old default, with no pose degradation.

### Key finding

Both-mode depth is linearly correlated with effective injection rate `p_both × p_dpt_inject` (Spearman ρ = −0.98, n=24). This is the main insight reported in the paper.

### All history.json on hdd2

All moddrop runs: `hdd2/ablation_c816_sim148_sitr_mae_md_*/history.json`
(Recovered from train.log on 2026-09-08 after accidental deletion.)

### Visualization

- `scripts/plot_moddrop_ablation.py` — generates 4 figures in `eval_results/`:
  - `moddrop_inject_sweep.png` — 5-point injection rate sweep
  - `moddrop_vs_default_heatmap.png` — % change vs default for all 15 settings
  - `moddrop_effective_injection.png` — depth vs effective injection rate (ρ=-0.98)
  - `moddrop_inject075_encoders.png` — SiTR+MAE vs DAv2+MAE at inject 0.75

---

## New Dropout Re-runs (in progress)

All 26 encoder ablation runs are being re-run with the new dropout setting (0.70/0.25/0.05 @ inject 0.75) on hdd2. Once complete, all paper tables (I–IV) will be updated with these numbers.

### Train config
`ablation/moddrop/train_bs32_md_b70_inj075.yaml`

### Output location
`hdd2/ablation_c816_sim148_{tac}_{rgb}/` with `train_config.yaml` containing `p_both: 0.7`

### Tracking
The `train_config.yaml` file in each run dir distinguishes new-dropout runs from old ones. Check with:
```bash
grep "p_both: 0.7" $RUN_DIR/train_config.yaml
```

---

## Overleaf

- **Project ID**: `6a6a76066a688253972691e8`
- **Git remote**: `https://git:<token>@git.overleaf.com/6a6a76066a688253972691e8`
- **Local paper dir**: `/media/hdd2/ihsuan/VisTacFusion/paper/`
- **Push**: from the paper/ dir git repo (separate from main VisTacFusion repo)

### Paper files

| File | Contents |
|---|---|
| `main.tex` | Title, abstract, structure |
| `tex/introduction.tex` | Introduction with contribution list |
| `tex/related_work.tex` | Related work (partially TODO) |
| `tex/method.tex` | Sensor design + model architecture |
| `tex/experiments.tex` | All 4 tables + analysis |
| `tex/conclusion.tex` | Conclusions + limitations |
| `references.bib` | BibTeX entries |
| `architecture_diagram.tex` | TikZ architecture figure (standalone) |
| `figures/architecture.pdf` | Compiled architecture figure |

### Remaining TODOs in paper

1. `tex/experiments.tex` — Real data description, qualitative results figure
2. `tex/method.tex` — Sensor photo (Fig. 2), UV switching time, camera resolution, LED specs
3. `tex/related_work.tex` — Visuo-tactile fusion paragraph, sim-to-real paragraph
4. `main.tex` — Acknowledgments
5. Update all tables after new-dropout encoder ablation completes

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/plot_moddrop_ablation.py` | 4 moddrop figures |
| `scripts/plot_ratio_ladder.py` | Ratio ladder + encoder comparison figures |
| `scripts/gen_encoder_ablation_configs.py` | Generate encoder config YAMLs |
| `modal_ablation.py` | Modal cloud deployment (do NOT use — $113 wasted) |

---

## Reproducing paper numbers

```python
# Example: extract Table I "Ours" row
import json

h = json.load(open("/media/hdd/ihsuan/VisTacFusion_outputs/ablation_c816_sim148_sitr_mae/history.json"))
best = min(h, key=lambda e: e["val"]["both"]["pose_rot_l1"])

for mode in ["both", "tactile", "rgb"]:
    v = best["val"][mode]
    print(f"{mode}: depth={v['depth_mse']:.4f}  normal={v['normal_mse']:.4f}  "
          f"rot_l1={v['pose_rot_l1']:.4f}  rot_deg={v['pose_rot_deg']:.3f}  "
          f"trans={v['pose_trans']:.4f}")
```
