# RoboSoft 2027 paper — data provenance

**最新文句修訂（2026-10-06）**：參考 TACTO、Taxim、DeepIM、Sparsh 的段落組織，兩版摘要改為無數字的定性敘述，四項貢獻整合為三項，並明確限定 known-object setting、mesh-derived geometry 與 manipulation scope。數值比較保留在實驗；圖表不變，短版 6 頁、長版 15 頁。詳見 `EDITORIAL_REVIEW.md` 最上方紀錄。

**2026-10-06 作者回饋修訂**：Fig. 1 使用作者最新指定的 `fig_arch.pdf`；Fig. 4 全實線。恢復摘要重點、label-accounting 獨立貢獻、有限定語氣的 pose 論點，以及長版 practical recipe/reporting；保留事實修正。短版 6 頁、長版 15 頁。

**全文修訂紀錄：[`EDITORIAL_REVIEW.md`](EDITORIAL_REVIEW.md)**。`main.pdf` 為六頁匿名稿；`main_long.pdf` 為 extended version。原始結果未改，修訂包含論述、資料來源校正、圖表與排版。

**Current state (editorial revision 2026-10-05, America/New_York): see `PAPER_NOTES.md`** — method name MeshTac, 6-page `main.tex` and
15-page `main_long.tex`, where every number comes from, run-name mapping, conventions and pitfalls.
The sections below are the original (2026-09-24) provenance notes of the renderer study and are kept
for the run tables; where they disagree with `PAPER_NOTES.md`, the notes win.

**Overleaf project**: `6ab1c9afe47f399544728ef3` (branch `main`). Sync with `./sync_overleaf.sh pull|push`
(needs `OVERLEAF_TOKEN`). Deadline date: **2026-10-15**; homepage says 23:59 PT but CFP says GMT—check PaperPlaza for the authoritative cutoff. Notification 2027-01-30.

## Layout

```
main.tex, sections/*.tex, refs.bib, ieeeconf.cls   Overleaf sources
figures/            fig_renders.png, fig_kshot.pdf, fig_fidelity.pdf (+ fig_loo/fig_cross/fig_sensorb, unused)
data/               tables.tex (generated), results_50ep.txt, masked_results.json, renderer_fidelity*.txt
scripts/            build_tables.py, make_paper_figs.py (run from the repo root)
```

## Runs

All downstream runs: `ablation/encoder/tac_sitr_single.yaml` (frozen SITR-B18, tactile only, DPT + pose
head, 12 M trainable), train config `ablation/pilot_robosoft/train_pilot_e50.yaml` (150-ep: `_e150`),
data configs `ablation/pilot_robosoft/data_<run>.yaml`. Outputs in
`/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2/pilot_<run>/` (50 ep) and `pilot150_<run>/` (150 ep),
logs in `outputs/pilot_logs/`. Summaries: `python ablation/pilot_robosoft/summarize.py`.

Sim data: `renders_v3`, 12 pattern objects, sim148 settings (5,345 train contacts; rotation-window
and translation filters, crop 0.816). Real data: `gs_blender/real_filtered`, 12 objects, 4,669 frames,
val = idx % 10 == 0 (472). Held-out objects (A): `pattern_04_3_lines_angle_2`, `pattern_31_rod`,
`pattern_33` (1,244 frames, evaluated in full, never used for selection).

| Table / figure | Runs (50 ep unless noted) | Source |
|---|---|---|
| Table I (LOO) | `A_realonly`, `A_blender`, `A_blender_bgsub`, `A_ganloo`, `A_ganloo_9obj` (geometry control), `A_diffloo`, `A_ganfull` (leaky) | `history.json` → `test.tactile` (depth-best / rot-best epoch selected on `val`); contact MAE/IoU from `data/masked_results.json` (`pilot_<run>/TEST`) |
| Table II (K-shot) | `B_k{0,10,25,100}_{realonly,blender,gank,ganfull}` | `history.json` → `val.tactile`; contact metrics `pilot_<run>/val` |
| Table III (variants) | `B_k25_{realonly,realonly_bgsub,gank,hyb,diff,blender,blender_bgsub}`, `A_{ganloo,hybloo,diffloo,blender,blender_bgsub}` | as above; fidelity column from `data/renderer_fidelity.txt` |
| Fig. 3 (K-shot curves) | Table II runs | `scripts/make_paper_figs.py` |
| Fig. 4 (fidelity vs downstream) | `B_k*_{gank,hyb,ganfull}` + `data/renderer_fidelity.txt` | `scripts/make_paper_figs.py` |
| Fig. 1 (renders) | `renders_v3/<obj>/session_00{2,3}/sensor_0000/{samples,samples_g_k25,samples_g_k100,samples_g}` | `scripts/make_paper_figs.py` |
| 150-ep confirmations | `pilot150_{A_realonly,A_ganloo,B_k100_realonly,B_k100_gank}` | `masked_results.json` keys `pilot150_*` |

Metric conventions: depth MSE over all pixels in mm² (as in the ICRA paper); contact MAE = MAE inside
GT depth > 0 (mm); IoU = (pred > 0.05 mm) vs (GT > 0); peak error = |max pred − max GT| per frame;
rotation = mean angular error (deg); translation = normalized L1.

## Renderers

| Name in paper | Sim tactile subdir | Renderer training data | Weights |
|---|---|---|---|
| Blender (physical) | `samples` | 1 real no-contact frame, BO on 16 params (`gs_blender/calibration/bo_tactile_v2.py`, `bo_results/tactile_v2/best_params.json`) | — |
| GAN-K (strict) | `samples_g_k{10,25,100}` | K evenly spaced train pairs per object (same subset as downstream `train_samples_per_session`) | `outputs/tactile_gan_k{10,25,100}` |
| GAN-LOO (strict) | `samples_g_loo` | all train pairs of the 9 non-held-out objects | `outputs/tactile_gan_loo` |
| GAN-full (leaky) | `samples_g` | all 4,197 train pairs | `outputs/tactile_gan` |
| Hybrid | `samples_h_*` | GAN pretrained on 30k (depth, Blender) pairs (`outputs/tactile_gan_simpre`), fine-tuned on K pairs | `outputs/tactile_gan_hyb_*` |
| Difference-image | `samples_d_*` | as GAN-K/LOO, target = image − session background, composited on `outputs/session_bg/ref.png` | `outputs/tactile_gan_diff_*` |

Renderer code: `scripts/tactile_render/{train_tactile_gan,generate_tactile,eval_renderer,compute_session_bg}.py`.
Fidelity = contact-region L1 of `G(depth)` vs real image on the real val pairs
(`eval_renderer.py`; images in [0,1]).

## Known caveats to state or fix before submission

- Real depth references are mesh projections at the recorded pose, not membrane measurements.
- One capture session per object: object identity and session appearance are confounded
  (session backgrounds differ by 0.05–0.10 L1, more than the best renderer's error).
- Real frames receive no augmentation; the one run with real rotation augmentation
  (`B_k25_realonly_aug`, aborted) showed ~89° rotation error after one epoch — possibly a
  label-handling bug for real rot-aug, unverified.
- Single seed; 50-epoch pilot numbers (few-shot depth converges by epoch ~10; pose keeps
  improving slowly to 150 epochs, rankings unchanged).
