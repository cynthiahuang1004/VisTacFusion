# MeshTac paper — notes for revising the manuscript

Companion to `main.tex` (6-page RoboSoft 2027 submission) and `main_long.tex` (15-page extended
version, same story, every result). Everything that is *in* the paper is not repeated here; this file
says where the numbers come from, how paper terms map to code names, and what must not be changed
without re-checking.

Deadline date: RoboSoft 2027, **2026-10-15**. Official homepage says 23:59 PT but CFP says GMT; confirm the time in PaperPlaza. Page limit 6 + up to 2 paid pages; `main.pdf` is
exactly 6 pages including references.

## Short-version readability and argument revision (2026-10-08; latest)

Revised the abstract in `main.tex` and all six short sections, keeping the renderer-free MeshTac main setting, current title, author-updated Fig. 1 (0.9 text width), and all raw results. `main.pdf` remains six pages. See the newest entry in `EDITORIAL_REVIEW.md` for papers read, before/after content corrections, and verification. Long-version sources/PDF, architecture/figure assets, shared `data/tables.tex`, and `refs.bib` are unchanged. This round has not been pushed to Overleaf.

The prose now distinguishes the observed gains from their interpretation: corrected contact state improves labels and student predictions; synthetic augmentation has small within-session gains at K=10 but no consistent additional gain across budgets, and helps session 2 at K=100. Self-training does improve contact metrics at larger budgets. Sensor-B pose gains are less consistent than contact gains. Table-II students share architecture/schedule but use different teachers; rows 3 and 5 share the real-only teacher and directly compare unrefined/refined mesh labels. The L1-search half-depth gate and all-frame pose supervision are explicit. Section E now uses **Label Budgets**, avoiding an implication of validation/test leakage. Do not restore either the renderer-assisted main setting or the retired terminology.

**Checkpoint policy (2026-10-08, unified; supersedes the exceptions note and the final-epoch policy):** every table and figure reports dense metrics at `best_depth.pt` (lowest validation depth MSE) and pose metrics at `best_pose.pt` (lowest validation rotation error). Sensor A: `build_tables.hist()` / `make_paper_figs.pose_ckpt()` take the best-validation-rotation epoch from history.json (no final-epoch checkpoint is saved). Session 2 and sensor B: re-evaluated with `eval_cross_session.py --ckpt best_pose.pt` into tags `cross_session_fv_bp` and `sensorB_mesh_bp` (dense metrics still from `cross_session_fv` / `sensorB_mesh`). `eval_pose_refine.py` already loads `best_pose.pt`. Translation is the mean absolute coordinate error in mm everywhere: `eval_cross_session.py` sums |dx|+|dy|, so `trans_l1 * 41 / 2`; history.json and the refinement script already average the two coordinates. Final-epoch vs selected-epoch rotation differ by <= 0.25 deg on sensor A.

**Translation aggregation (now stated in setup/captions):** sensor-A budget metrics (`engine/eval.py`) and test-time refinement (`eval_pose_refine.py`) average absolute errors over x and y. Sensor-B/session-2 evaluation (`eval_cross_session.py`) sums the two axes and averages over frames. Fig. 4(c)/(f) and Tables III/IV therefore use different aggregation; the text now makes this explicit without changing shared table data or rescaling results. Direct cross-table comparisons would require a unified evaluation protocol.

The abstract keeps the existing quantitative results and explicitly identifies the real-only comparison, two sensor-specific models and 12 ridge-pattern objects. Discussion now includes contact-state accuracy, the distinction between full-image and contact metrics, simulation coverage, pose ambiguity, validation-label cost and future work in connected paragraphs. PBR contact-tuning split/K-budget provenance remains unconfirmed; the author has been asked, and no train-only/K-only guarantee was invented.

## Short-version full-text revision (2026-10-07; historical)

This round edits only `sections/*.tex` and rebuilds `main.pdf` (6 pages). See the newest entry in `EDITORIAL_REVIEW.md` for primary papers read, section-level changes, before/after factual corrections, captions, and verification. The existing abstract, title and five experiment headings remain. Method now specifies the actual student losses; setup explicitly states that the object embedding conditions the pose head. Cross-session renderer benefits are budget-dependent (K=100 improves; at K=25 renderer-free MAE is slightly lower). Test-time refinement's 0.4–1.2° range applies to K >= 25, whose session-2 medians are 1.5–1.8°. The held-out-object caption now explicitly exempts the all-pairs renderer from the exclusion of those objects' real training contacts. For the half-gated L1-search control, depth is gated per object but pose supervision uses all search results (`pose_all=True`).

No long-version or shared figure/table/bibliography files were edited by this round. An external update to the long experiment section and shared plots was detected during the work and preserved; the final short captions were checked against the current six-panel K-shot plot. No Overleaf synchronization was run during editing; the author subsequently requested a push, and the six short-section files were pushed and verified at commit `a233bb8` (latest remote base: `89d0ac5`). Physical-renderer contact-tuning split provenance remains unconfirmed.

## Reframing (2026-10-07, author decision after an external review)

- **Fig. 1 training/inference version (author-approved and pushed):** the short version now includes `figures/fig_arch_training_inference.pdf`, built from `architecture_diagram_training_inference.tex`. Both teacher and student are expanded as the same frozen SITR-B/18 encoder, DPT decoder and Pose MLP; object embedding connects only to pose. Training shows registered and mesh-aligned labels supervising the student; inference separates student-only prediction from optional mesh-assisted refinement, including real-image and mesh inputs. The short caption describes panels (a) and (b). Pushed to Overleaf and verified at `180a0f0`, on remote base `05717b2`; local and remote short manuscripts compile to six pages with no unresolved references or overfull boxes. The original `figures/fig_arch.pdf` and long-version sources remain unchanged. `main_fig1_preview.pdf` is a preview copy of the updated short manuscript.
- **L1-search explanation (author-requested clarification):** the short-version ablation now explicitly states teacher-pose initialization, fitting the mesh to teacher depth, 567 nearby candidate poses, learned-renderer RGB generation, and selecting the pose with minimum full-image RGB L1 against the real tactile image as the pose pseudo-label. The adjacent results paragraph was shortened to avoid repeating table values and retain six pages. This is the search baseline only; the renderer-free MeshTac framing is unchanged. Long-version sources and shared figures/tables were not edited in this clarification. Pushed to Overleaf as `83753bb`; the remote commit was verified and the short version still compiles to six pages.
- **MeshTac = renderer-free pipeline** (code names `rf` / `maplnr_m`: real-only teacher, aligner trained on real perturbations only, student without synthetic frames). The former main configuration (`al2` / `mapl_m`, K-pair learned renderer in aligner, teacher and student) is now the variant "+ synthetic frames" and appears as an indented row under MeshTac. Fig. 3, Fig. 5 (predictions of the rf models), Table I-IV and all numbers in the text follow this. Do not swap back.
- **Checkpoints**: dense metrics at the best-validation-depth checkpoint; pose metrics (rotation, translation) at the best-validation-rotation checkpoint, uniformly (see the 2026-10-08 policy note above). The text still reports final-epoch depth MSE for the seed comparison.
- **Table I** has Normal (deg) and Trans. (mm) columns; translation = normalized L1 x 41 mm (`HALF_MM`).
- **"leaky" is retired** -> "all-pairs renderer / all-pairs reference" everywhere (tables, figure legends, text).
- **Fig. 5** frames = the object's median MeshTac contact MAE (not the median improvement).
- **Aligner convergence** (`scripts/aligner_convergence.py` -> `ablation/pilot_robosoft/aligner_convergence.json`, `fig_conv.pdf`, shown in both versions): from the teacher pose 5.6 -> 2.1 (1 it.) -> 1.1 deg (3 it.), min 1.05 at 5; six updates recover perturbations up to 15 deg (0.6 deg), 20 deg -> 0.8, 25 -> 1.5, 30 -> 3.2.
- Contribution 2 is phrased as an empirical finding (renderer study), contribution 3 no longer claims MeshTac on held-out objects. Title: "Contact Geometry" (was "Tactile Geometry").
- Test-time refinement table uses the rf models (`pose_refine_results.json` keys `pilot150_B_k{K}_rf_realonly/*`).

## Abstract and terminology (2026-10-07, author decision)

The abstract now leads with the registration needed to obtain mesh-derived contact labels, introduces MeshTac in the second sentence, and groups its mechanism, experimental evidence, and implications in that order. It specifies sensor-specific models and ridge-pattern objects, adds the cross-session finding, and ends with the implications for alignment and renderer supervision. The quantitative findings remain: contact depth error -20% on sensor A and -36% on GelSlim 4.0 at K=25; all-pairs renderer 24-26% versus matched 0-3%. Keep these numbers; this supersedes the 2026-10-06 note below. The abstract structure was revised after directly reading seven related abstracts; see the latest entry in EDITORIAL_REVIEW.md. Both abstracts contain approximately 207 words; the compiled short and extended versions remain 6 and 15 pages.
The coined term "label accounting" is retired everywhere; use "matched-budget protocol / matched budgets / renderer supervision counted in the label budget". Do not reintroduce it.

## Latest prose revision (2026-10-06)

The author now prefers an abstract without quantitative results. Both abstracts use qualitative findings; exact effect sizes and half-budget comparisons remain in Experiments. This supersedes the earlier request to restore numbers in the abstract. The contributions are consolidated into three: the MeshTac method, matched-budget label accounting, and systematic empirical characterization (including optional test-time refinement). The qualified pose-bottleneck conclusion and extended practical recipe/reporting remain. Introduction, related work, method transitions and experimental paragraphs were revised using TACTO, Taxim, DeepIM and Sparsh as structural references (see `EDITORIAL_REVIEW.md`). The paper explicitly targets known-object tactile perception and distinguishes mesh-projected geometry from physical elastomer deformation. No figure, result table or bibliography changed. Short: 6 pages; extended: 15 pages.

## Author-directed follow-up (2026-10-06, America/New_York)

Fig. 1 uses the author's latest `fig_arch.pdf` (superseding the earlier v1 request); Fig. 4 retains Claude's four solid lines.
Restored the abstract's label-accounting motivation and concrete half-budget comparisons,
a separate label-accounting contribution, the qualified pose-bottleneck interpretation,
and the extended version's practical recipe/reporting sections. All protocol corrections remain.
Short version: 6 pages; extended: 15 pages.

## Editorial revision (2026-10-05, America/New_York)

See `EDITORIAL_REVIEW.md` for the full change log (its open items on PBR provenance and the translation/checkpoint protocol were resolved on 2026-10-08; see the policy note above).
Both documents default to anonymous review; define `MeshTacAuthorVersion` for an author copy.
K is a **training-label** budget; additional validation labels select checkpoints.
GAN-full exceeds K in the few-shot protocol, but never trains on validation images. In LOO it sees real frames of evaluation objects.
The cross-session `A_realonly` reference uses nine training objects, not all twelve; its comparison with K-shot methods is not matched for object coverage.
Main sensor-A aligner logs include synthetic examples (24 real + 40 synthetic per batch), all generated by the same-K renderer. Renderer-free and sensor-B aligners use real examples only. No additional real training labels are consumed.

## 1. Files

| What | Where |
|---|---|
| 6-page paper | `main.tex`, `sections/01_intro … 06_conclusion.tex` |
| Extended version | `main_long.tex`, `sections_long/01_intro, 02_related, 03_setup, 04_mapl, 05_experiments, 06_conclusion.tex` |
| Shared | `refs.bib`, `ieeeconf.cls`, `data/tables.tex` (generated), `figures/` |
| Overview figure (Fig. 1) | Short: **`figures/fig_arch_training_inference.pdf`**, author-approved TRAINING/INFERENCE figure with expanded teacher/student architectures. Long: original **`figures/fig_arch.pdf`** (preserved). |
| Tables | `scripts/build_tables.py` → `data/tables.tex`; one `\newcommand{\tab<name>}` per table body |
| Figures | `scripts/make_paper_figs.py`; `scripts/make_arch_assets.py` (thumbnails of Fig. 1); `scripts/dump_qual_preds.py` (predictions for Fig. 5) |
| Overleaf | project `6ab1c9afe47f399544728ef3`; `OVERLEAF_TOKEN=… bash sync_overleaf.sh push "<msg>"` (also `pull`) |

Compile from this directory: `pdflatex main; bibtex main; pdflatex main; pdflatex main` (same for
`main_long`). Regenerate tables/figures from the repo root (`/media/hdd2/ihsuan/VisTacFusion`):

```
conda run -n vistacfusion python paper_RoboSoft/scripts/build_tables.py
conda run -n vistacfusion python paper_RoboSoft/scripts/make_paper_figs.py paper_RoboSoft/figures
cd paper_RoboSoft && pdflatex -output-directory=figures -jobname=fig_arch architecture_diagram.tex
```

Short-paper tables use the `*short`/`*main` macros (`\tabkshotshort`, `\tabrcablmain`, `\tabcrossshortb`,
`\tabsbshortb`, `\tabposeref`, `\tabloomain`); the long version uses the full ones (`\tabkshot`,
`\tabseeds`, `\tabrcabl`, `\tabcross`, `\tabsbmapl`, `\tabsensorb`, `\tabloo`, `\tabvariants`).

## 2. Paper term ↔ code / run name

The method was called MAPL (and briefly SnapTac) during development; code, run directories and result
keys still use the old names. Do **not** rename anything in code.

| Paper | Runs / dirs (sensor A) | Runs (sensor B) |
|---|---|---|
| Real only | `B_k{K}_realonly` | `S_k{K}_realonly_m` |
| Physical renderer | `B_k{K}_blender` | — |
| Learned renderer (GAN-K) | `B_k{K}_gank`; renderer `outputs/tactile_gan_k{K}` | `S_k25_gan40m`; `outputs/tactile_gan40_k25` |
| Self-training (teacher depth as label) | `B_k{K}_semipm_pseudo` | `S_k25_semipm_m` |
| Teacher depth + teacher pose | `B_k25_tp` | — |
| L1 render-and-compare search (½ gated) | `B_k25_rcp` (also `rc50`, `rc100`, `rcp00/25/75`) | — |
| **MeshTac** (aligner labels, renderer retrained on aligned pairs) | `B_k{K}_al2`; labels `gs_blender/real_pseudo_k{K}_al`; renderer `outputs/tactile_gan_al_k{K}`, sim images `renders_v3/*/samples_a_k{K}`; aligner `outputs/aligner_k{K}` | `S_k25_mapl_m`; labels `real_pseudo_S_k25_m_al`; renderer `tactile_gan40m_al_k25`; aligner `outputs/aligner_S_k{K}_m` |
| MeshTac, no renderer (real-only teacher, no synthetic frames) | `B_k{K}_rf`; labels `real_pseudo_k{K}_ro_al`; aligner `outputs/aligner_k{K}_realonly` | `S_k{K}_maplnr_m`; labels `real_pseudo_S_k{K}_m_al` |
| MeshTac, sim-free student only | `B_k{K}_al0` | — |
| Unlabeled fraction 25/50 % | `B_k25_al2u25`, `B_k25_al2u50` | — |
| Second round (student as teacher) | `B_k{10,25}_al2_r2` | — |
| Leaky renderer (GAN-full) | `B_k{K}_ganfull` | — |
| Teacher f0 | the `gank` model of the same K (sensor A); the `realonly_m` model (sensor B) | |

Prefix `pilot150_` = 150-epoch run (all main sensor-A numbers), `pilot_` = 50-epoch run (seeds, ablation
table, all sensor-B runs). Seeds: `<run>_seed1`, `<run>_seed2` (seed 0 has no suffix). Checkpoints in
`/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2/<prefix><run>/{best_depth.pt,best_pose.pt,history.json}`
(`latest.pt` was deleted to free disk; nothing in the paper needs it). Logs: `outputs/pilot_logs/`.

Method code: `scripts/tactile_render/train_aligner.py` (aligner), `scripts/tactile_render/pseudo_label.py`
(teacher labels), `scripts/tactile_render/pseudo_label_rc.py` (alignment / L1 search → mesh labels),
`scripts/tactile_render/train_tactile_gan.py` + `generate_tactile.py` (learned renderer),
`scripts/tactile_render/fit_mesh_pose.py` + `scripts/canonicalize_gs40_poses.py` + `scripts/build_gelslim40_mesh.py`
(sensor-B mesh registration), `ablation/pilot_robosoft/eval_masked.py`, `eval_cross_session.py`,
`eval_pose_refine.py` (evaluation).

## 3. Where every number comes from

All result files are in `ablation/pilot_robosoft/` (repo root). Keys are `<prefix><run>/<split or tag>`.

| Quantity | File / key | Notes |
|---|---|---|
| Depth MSE, normal MSE, rotation, translation (val) | `<outputs>/<run>/history.json` via `build_tables.hist()` | depth metrics at the best-val-depth epoch, pose at the best-val-rotation epoch |
| Contact MAE, IoU, normal angular error (val, TEST) | `masked_results.json`, keys `…/val`, `…/TEST` | TEST = the three held-out objects (A runs) |
| Second session (sensor A) | `cross_session_results.json`, tag **`cross_session_fv`** only | tags `cross_session_curated` / `curated361` / `curated854` are from a label-scale bug and must not be used |
| Sensor B | `sensorB_results.json`, tag `sensorB_mesh` (mesh-registered labels, used in the paper) and `sensorB_test` (original public labels, mentioned once) | |
| Test-time alignment | `pose_refine_results.json`, keys `pilot150_<run>/{val,session2}`, fields `rot`, `rot_al`, medians, IoU/cMAE of mesh@aligned | |
| Label quality of pseudo labels (IoU / contact MAE / pose error vs withheld GT) | `outputs/pilot_logs/pseudo_k{K}_al.log` (sensor A, K=10,50,100), `pseudo_S_k{K}_m_al.log` (sensor B), lines `teacher (all)` / `mesh (kept)`; also `stats` in `gs_blender/real_pseudo_*/pseudo_manifest.json` | the sensor-A K=25 log was overwritten; its values (teacher 0.55/0.32/5.3°, aligned 0.71/0.24/1.1°) are hard-coded in `make_paper_figs._label_quality` and in `\tabrcablmain` via `labq("real_pseudo_k25_al", …)` |
| Renderer fidelity | `outputs/pilot_logs/renderer_fidelity_all.log`, `data/renderer_fidelity*.txt` | |
| Seeds (mean ± std) | `build_tables.seeds()` over seed 0/1/2 of `pilot_<run>` | |

Numbers quoted in prose (percentages, "halves the labels", "93 %") are derived by hand from the tables;
if a table changes, re-derive them. Current canonical values:

**Sensor A, 150 ep (depth MSE / contact MAE / IoU / rot°), real only → MeshTac**
K=10: 0.0642/0.330/0.625/9.6 → 0.0559/0.288/0.682/4.0 · K=25: 0.0558/0.312/0.664/5.6 → 0.0450/0.249/0.727/3.7 ·
K=50: 0.0482/0.283/0.690/4.3 → 0.0307/0.206/0.769/3.7 · K=100: 0.0369/0.235/0.744/3.2 → 0.0232/0.168/0.803/3.7.
Leaky K=100: 0.0274/0.205/0.777. No-renderer MeshTac: 0.0521/0.0433/0.0291/0.0227.
Seeds (50 ep) K=25: 0.0447±0.0009 vs 0.0551±0.0010; K=100: 0.0229±0.0003 vs 0.0366±0.0008.
Derived: depth MSE −13/−19/−36/−37 %, contact MAE −13/−20/−27/−28 %.

**Sensor B, 50 ep, mesh-registered labels, real only → MeshTac (no renderer)**
K=10: 0.0314/0.180/0.780/14.0 → 0.0170/0.121/0.856/9.5 · K=25: 0.0182/0.127/0.847/13.4 → 0.0076/0.081/0.911/9.2 ·
K=50: 0.0088/0.083/0.899/9.0 → 0.0041/0.061/0.933/9.8 · K=100: 0.0051/0.066/0.932/8.4 → 0.0034/0.056/0.944/9.1.
K=25 with renderer: 0.0075/0.078/0.904/9.9; learned renderer 0.0148/0.111/0.877; self-training 0.0175/0.135/0.846.
Seeds K=25: real 0.0179±0.0004 / 0.127±0.001 / 0.849; MeshTac 0.0075±0.0001 / 0.078±0.0003 / 0.903.
Original public labels, K=25: contact MAE 0.137 → 0.098. Derived: contact MAE −33/−36/−26/−15 %, depth MSE −46/−58/−53/−33 %.

**Label quality (IoU, teacher → aligned)**: sensor A 0.53→0.65, 0.55→0.71, 0.64→0.76, 0.66→0.81 (pose 1.6/1.1/1.1/0.7°);
sensor B 0.78→0.85, 0.85→0.91, 0.90→0.94, 0.93→0.95 (pose 2.3/1.4/0.9/0.8°, teacher 6.6–11.4°).

**Cross-session (sensor A, `cross_session_fv`, contact MAE / IoU)**: nine-object real-only reference 0.315/0.675; K=25 real 0.368/0.598,
self-training 0.335/0.598, MeshTac 0.310/0.660 (no renderer 0.306/0.654); K=100 real 0.351/0.657,
self-training 0.300/0.707, MeshTac 0.252/0.743 (no renderer 0.288/0.707).

**Test-time alignment (rot°, pred → aligned)**: val, real only 9.6/5.6/4.3/3.2 → 2.8/1.2/1.2/0.4; MeshTac
4.0/3.7/3.7/3.7 → 1.5/0.9/1.2/0.8. Session 2, real only 20.2/16.2/13.8/12.9 → 11.4/5.5/4.8/4.9; MeshTac
8.8/9.0/10.6/10.2 → 4.9/4.8/5.2/4.8. Unseen objects (aligner trained on 9 objects): 4.8–6.8° → 0.6–1.0°.

## 4. Terminology and conventions (keep)

- **GT** = ground truth = the object mesh projected at the recorded pose (sensor A) or at the pose fitted
  to the public label (sensor B). Defined once in Setup ("ground-truth (GT) depth and normals are …");
  use "GT" everywhere afterwards, not "reference" (the word "reference" is kept only for "leaky
  reference" / "as a reference for common practice").
- Method name **MeshTac**; expanded as "mesh-aligned pseudo-labeling". Earlier names MAPL/SnapTac must
  not reappear (MAPL and TacAlign collide with existing papers).
- `$\Kreal$` = K, labeled frames per object. Few-label regime = K ≤ 25 (3 % and 7 % of sensor-A training
  frames); K = 50, 100 are "the trend". Headline claims use K = 10–25; do not advertise "10–100".
- Table rows are **methods** ("Real only", "Learned renderer", "Self-training", "MeshTac (ours)",
  "Leaky renderer"), no "+" prefix; column header "Method". Leaky rows italic, never bold.
- Caption style: short bold noun-phrase title ("Overview of MeshTac.", "K-shot results on sensor A."),
  details after, conclusions in the text.
- Teacher = f0, student = f1, aligner = A; the same aligner is used for labeling and for test-time
  refinement (same weights). Student architecture = teacher architecture = standard frozen SITR-B/18
  encoder + DPT decoder + pose MLP; the contribution is the labeling, not the network.
- Sensor A = GelSlim 5.0 (`\sensor`), sensor B = GelSlim 4.0 (`\sensorB`, public dataset, Sipos et al.).
- Metrics: depth MSE over all pixels (mm²); contact MAE inside the GT contact region (mm); IoU over the union of predicted (>0.05 mm) and GT (>0) masks;
  normal angular error inside the contact (deg); rotation error (deg). Dense metrics come from
  `best_depth.pt`; all pose metrics use `best_pose.pt` (tags `*_bp` for session 2 and sensor B). The old `cross_session_fv` / `sensorB_mesh` rotation and translation values were taken at the depth checkpoint and are no longer used in the paper.

## 5. Facts that are easy to get wrong

- **Sensor B labels were re-registered to the meshes.** The public labels are mesh depth clipped at
  1 mm and Gaussian-blurred, not exact projections. We fitted the mesh pose that best explains each label
  (0.018 mm mean residual) and canonicalized it against the object's rotational symmetry. Every method is
  trained and tested on these labels (`real_gelslim40_{train,test}_mesh`); the paper also reports
  MeshTac vs real-only on the original labels (`sensorB_test`) as a check. An earlier sensor-B result
  (MeshTac contact MAE 0.227/0.210, "MAPL does not help on sensor B", "condition: aligned mesh must beat
  the teacher") came from a gel-view bug and is **invalid**; it survives only in git history and in
  `*_bugview` directories. Nothing in the paper may cite it.
- **Second session**: only tag `cross_session_fv` (labels re-rendered at the full view by
  `scripts/fix_session2_view.py`) is correct; earlier tags scored against labels magnified 1.225×.
- **"93 % of the frames"**: at K = 25 the few-shot renderer's render at the *true* pose is farther from
  the frame (image L1) than its render at the pose the search selected, on 93 % of frames — i.e. the
  renderer scores a wrong pose above the true one. It does not mean the search is wrong on 93 %.
- **"halves the labels"**: MeshTac at K lands at or below real only at 2K (sensor A: 0.0307 at K=50 vs
  0.0369 at K=100; sensor B contact MAE 0.081 at 25 vs 0.083 at 50, 0.061 at 50 vs 0.066 at 100).
- **Rotation on sensor B** improves only at K = 10, 25 (14.0→9.5°, 13.4→9.2°); at K ≥ 50 it is ≈9° either
  way. Sensor-A rotation at K = 100: real only 3.2° < MeshTac 3.7° until test-time alignment.
- **No K = 0 points** in Fig. 3: real only, learned renderer, self-training and MeshTac need labels;
  only the physical and leaky renderers exist at K = 0 (reported in the long version only).
- **Fig. 5 frames** are chosen by rule (per object, the frame whose contact-MAE improvement is the
  object's median), not hand-picked; the sensor-B objects shown are pattern_04, pattern_33, pattern_06
  (pattern_37, the hardest object, is not shown but is in every average). Predictions are cached in
  `/media/hdd2/ihsuan/VisTacFusion/outputs/robosoft_qual/qual_{A,B}.npz` (130 MB each; keep them out of
  `data/`, which is synced to Overleaf).
- **Fig. 1**: the author selected the latest `fig_arch.pdf`. Its student-output thumbnails are actual predictions (`make_arch_assets.py`, `pilot150_B_k25_al2/best_depth.pt`), not the earlier GT placeholders. The caption also notes the omitted synthetic training branches. **Fig. 4 must retain four solid lines.**
- **Second round** (student as teacher) gives no gain (K=10: 0.0570 vs 0.0559; K=25: 0.0445 vs 0.0450).
  Unlabeled fraction 25/50/100 %: 0.0453/0.0452/0.0447 (long version only).
- **Leave-object-out**: held-out objects pattern_04_3_lines_angle_2, pattern_31_rod, pattern_33
  (1,244 frames); runs `A_*` (prefix `pilot150_` for the main rows).
- The renderer-free variant is equal or better *within* a session but worse *across* sessions
  (K=100 contact MAE 0.288 vs 0.252); that is why the renderer stays in the main method.
- Real data counts: sensor A 4,669 frames (222–568 per object, 472 val, one session per object), session 2
  854 curated frames of 11 objects; sensor B 1,620 test frames, ~270 training frames per object.
  Simulated contacts: 5,345. Trainable parameters: 12 M. Unlabeled pool per K: 4,077 / 3,897 / 3,597 /
  2,997 frames (sensor A), 3,120 / 2,940 / 2,640 / 2,040 (sensor B).

## 6. Known weaknesses a reviewer will raise (address in wording, not by over-claiming)

1. Only 12 ridge-pattern objects; no curved or textured objects. Not fixable without new data.
2. Needs the mesh and the identity of the object in every unlabeled frame; test-time alignment needs
   them at test time too. Scope: known objects (industrial parts, calibration objects).
3. No downstream robot task.
4. Sensor-B GT is our re-registration of public labels (see §5); three seeds on sensor B at every budget (real-only and AlignTac).
5. GT is a geometric projection, not a measured membrane deformation (both sensors).
6. RoboSoft audience: keep the first paragraph of the introduction readable for non-ML reviewers; avoid
   unexplained ML jargon.

## 7. Authors and metadata

I-Hsuan Huang*, Xiaohao Xu*, Xiangdong Wang, Xiaonan Huang, Nima Fazeli (University of Michigan;
* equal contribution). Correspondence `{ihsuan, xiaohaox}@umich.edu`. Check the author list before
submission.

- PBR calibration provenance (confirmed by the author 2026-10-08): one no-contact image for the automatic illumination fit, one contact image of one object outside the training set for manual gel/contact tuning; no labels used, nothing counted in K.

- 2026-10-08: short version replaces the pose-refinement table with paired bars (`figures/fig_poseref.pdf`, `make_paper_figs.fig_poseref`, 2x2 single column: rotation/translation x session 1/2; hollow = predictor, filled = +align). The long version keeps the table (`\tabposeref`).

- 2026-10-08 (late): short version drops Fig. 3 (renderer examples, `fig_renders_small.pdf`), Fig. 5 (label quality vs K, `fig_labels.pdf`) and the sensor-B block of Table III (`\tabsbT`); their content is summarized in one sentence each (Sec. B label IoU numbers, Sec. C sensor-B IoU). Files and macros are still generated for the long version. Freed space went to Fig. 2 (0.92 columnwidth), Table II (scriptsize, no resizebox, 7pt group gap, row "Aligner (ours)") and larger Figs. 3/4/6.

- 2026-10-09: CLEAN PROTOCOL IN THE SHORT VERSION. All sensor-A within-session numbers (Table I, Table II student columns, Table III session-2 rows, Fig. 3 a-c, Fig. 5 sensor A, Fig. 6 a,c and the text) now come from the T_* runs (trained without the 530 held-out frames) evaluated on `gs_blender/real_filtered_testA`: dense from best_depth.pt (`testA_results.json` tag testA), pose from best_pose.pt (tag testA_bp), refinement keys `pilot150_T_k{K}_{realonly,rf}_realonly/testA` in pose_refine_results.json, session-2 keys `pilot150_T_*/cross_session_fv(_bp)`. Self-training and Table II rows are renderer-free (T_k*_semipm, T_k25_{tp,rf0,rcp} with sim train_samples_per_session 0). Headline numbers: contact MAE -15/-20/-31/-29% (K=10/25/50/100), depth MSE -18/-22/-30/-36%; full renderer 17-23% vs budget-matched 2-3%; session 2 MeshTac -15/-20%, +synth K=100 0.248 (-31%). PBR students (T_k*_blender) retrained 2026-10-09 09:01 for Fig. 3. Old B_* numbers remain in the long version.

## 2026-10-09 — v6 review round (GPT review of the clean-protocol PDF)
- **Verified, not a leak:** the full-set learned renderer (`outputs/tactile_ganT`) was retrained with `--exclude-frames testA_frames.json`; its log reports `pairs: train=3667 val=472`. The short-version text still said 4,197 (legacy count) — fixed to 3,667 in Sec. IV; long version now states both counts (legacy 4,197 / clean 3,667).
- Sec. V-E percentages were stale (0–3% / 24–26%): now 2–3% (budget-matched) vs 17–23% (full-set) at K=10–25, computed from `testA_results.json` (gank 2.5/1.7%, ganfull 22.6/16.7%), matching the abstract.
- Single-checkpoint reference added to Sec. V-A: the final-epoch checkpoint of MeshTac K=25 gives MSE 0.041 / IoU 0.74 / rot 3.2° / trans 0.88 mm simultaneously (`testA_final` tag).
- Wording softened: abstract first sentence (contact-state estimation, not "image registered to mesh"), "no synthetic tactile images", contribution 2 ("critical for" instead of "determines"), Sec. V-B closing sentence, Sec. V-E held-out-object interpretation (object-specific pose supervision, not rendering fidelity), related-work contrast with inference-time refinement, aligner first-layer init (depth filters = mean of RGB filters, per `train_aligner.py`), sensor-B per-frame mesh fitting defines labels only.
- Table II rows 2–3 (Teacher/Teacher = `T_k25_tp`, Mesh/Teacher = `T_k25_rf0`) were single-seed; seeds 1–2 launched 2026-10-09 16:34 on GPUs 0–1 (`worker_T.sh`, jobs `train50 … e50_pm 1|2`), evaluated by `finisher_seeds.sh` → tags testA/testA_bp. `build_tables.py` already averages whatever seeds exist (n=3 for both rows). After eval: rebuild tables and drop "except rows 2–3" from Sec. V-B.
- Row 2 (Teacher/Teacher) is the "stronger self-training" baseline the review asked for (teacher depth + normals + pose pseudo-labels); the text now says so.
- **Seeds done (2026-10-09 17:26):** Table II rows 2–3 now average three 50-epoch seeds on the 530 test frames. Teacher/Teacher: IoU 0.63 / cMAE 0.30 / rot 6.1° / trans 1.2 mm (was 6.2° single seed); Mesh/Teacher: IoU 0.48 / cMAE 0.36 / rot 6.0° / trans 1.2 mm (IoU was 0.51 single seed), depth MSE 0.058 (was 0.060) vs 0.049 real-only. Aligner row remains best in every column; all Sec. V-B statements re-checked. Runs: `pilot_T_k25_{rf0,tp}_seed{1,2}` under `/home/shared/vtf_outputs_T`.

## 2026-10-10 — co-author pass (X. Xu on Overleaf, commit 34f2646) + eight-page version
- Pulled the co-author's Overleaf edits: method renamed **MeshTac → AlignTac**, new title ("AlignTac: Aligning Real Tactile Images with Simulated Geometry for Label-Efficient Contact Learning"), new abstract, expanded intro/method (problem formulation, numbered equations), pastel rank colors in tables (`table_style.tex`: \rankfirst/\ranksecond/\rankthird, grey "reference" rows), section titles. `data/tables.tex` now carries **manual** rank-color edits — `build_tables.py` refuses to run without `--force`; re-apply the coloring after any regeneration.
- RoboSoft allows 6 + 2 purchased pages; the user approved 8 pages. Added: intro teaser `figures/fig_teaser.pdf` (`scripts/fig_teaser.py`: real registration vs. simulated appearance vs. AlignTac, frame pattern_06 #0479), restored Fig. labels (pseudo-label quality vs K, both sensors), the sensor-B block of Table III (`\tabsbT`), and the renders figure (`fig_renders_small.pdf`) in Sec. V-E.
- Fig. 4 caption now states why PBR / full-set renderer are absent on sensor B (PBR scene calibrated to sensor A; full-set renderer is a non-matched reference).
- Sensor naming per co-author: sensor A = "marker-free variant of the GelSlim 4.0 design (17.5 mm view, six colored LEDs)", sensor B = "the public GelSlim 4.0 with markers". GelSlim 5.0 stays only in the author-version macro.
- Figures regenerated with the AlignTac legend (`make_paper_figs.py`); Fig. 1 rebuilt from the local diagram source. `sync_overleaf.sh` FILES now include `table_style.tex` and `architecture_diagram_training_inference.tex`. Long version renamed too and compiles (17 pages).

## 2026-10-10 (later) — co-author revision 2 pulled; K=5 and no-identity runs
- Pulled X. Xu's Overleaf revision (20a2cfb): his own intro figure (`figures/intro_figure/`, replaces our `fig_teaser`), TikZ pipeline (`figures/pipeline.tex`), `notation.tex` (f^t, f^s), caption/paragraph/bibliography layout files, `build.py`; both sensors named as GelSlim 4.0 variants (marker-free = A, dot-marker = B). All numbers verified unchanged.
- K=5 (clean protocol, 530 test frames; runs `pilot150_T_k5_{realonly,blender,semipm,rf}`, labels `real_pseudo_k5_ro_{ng,al}`, aligner `outputs/aligner_k5_realonly`): real only MSE 0.0656 / cMAE 0.348 / IoU 0.599 / rot 14.0 / trans 2.05; PBR 0.0656 / 0.368 / 0.591 / 6.5 / 1.21; self-training 0.0580 / 0.346 / 0.543 / 14.7 / 2.20; AlignTac 0.0553 / 0.313 / 0.642 / 6.3 / 1.24. Label quality: teacher IoU 0.475, rot 13.84 -> aligned 0.580, 5.96. Reported as one sentence in Sec. V-A only (user decision: no table/figure change). `pilot150_T_k5_tp` = an accidental teacher-pose variant (root _ro_ng without skip_pseudo), not used.
- No-object-identity students (`pilot150_T_k{25,100}_{rf,realonly}_noid`, model `tac_sitr_single_noobj.yaml`, `eval_cross_session.py --model`): dense metrics unchanged, rotation worse (AlignTac 7.0/7.1 deg, real-only 17.2/7.8 deg at K=25/100). User decision: NOT included in the paper.

## 2026-10-10 (evening) — story reframing, Fig. 1 mock, session-2 unlabeled-pool experiment
- Story (agreed with the user): images are easy to collect, mesh-registered labels are not. Problem = image–geometry pairs need calibration + verification (real) or sensor-response modeling (sim). Obstacle to "just project the mesh" = label sensitivity to the contact state, no direct depth-vs-RGB alignment signal, only K registered contacts to learn from. Intro paragraphs 1, 2, 4 and two Setup sentences edited accordingly (the co-author's \FinalLine box at the end of intro paragraph 2 was removed because the paragraph ending changed). Renderer figure width 0.9\columnwidth to stay at eight pages.
- Setup now states: registered = recorded pose after per-object calibration and an image–mesh agreement check (`gs_blender/filter_real_data.py`); in session 2 the session-1 calibration no longer held, per-run corrections averaged 11.4 deg and 1.6/1.7 mm (`new_real_test/validation/pose_calibration_new.json`, 48 runs).
- Fig. 1 reference mock for the PPT revision: `figures/intro_figure/mock_fig1_v2.{tex,pdf,png}` (NOT used by main.tex); overlay assets `figures/intro_figure/v2_assets/` from `scripts/fig1_assets.py` (mesh outline on the real image at teacher / aligned state for frame pattern_06 #0479; second-session frame at the old calibration vs verified pose). The #0479 example was chosen for a large alignment gain (teacher translation error 4.6 mm), not a median sample.
- Session-2 unlabeled-pool experiment (running): `ablation/pilot_robosoft/make_s2_split.py` alternates the 194 presses per object -> pool (1,208 contact frames, stride 2, `gs_blender/real_s2_pool`, poses never used; 1,066 train-split frames labeled) and test half (464 curated frames, `real_s2_testhalf`). Labels: `real_pseudo_s2_k{25,100}_{ng,al}` (teacher `T_k{K}_realonly`, aligner `outputs/aligner_k{K}_realonly`); merged roots `real_merge_k{K}_{rf,semipm}_s2`; runs `pilot150_T_k{K}_{rf,semipm}_s2`; eval tags `s2half`, `s2half_bp` in cross_session_results.json (`finisher_s2.sh`). Pool label quality on 162 frames with a curated label (`s2_label_quality.py`): K=25 teacher IoU 0.51 / rot 16.2 (median 15.3) -> aligned 0.58 / 8.6 (2.7); K=100 0.57 / 19.6 (18.6) -> 0.63 / 4.4 (2.0).
- 2026-10-10 22:30: Fig. 1 is now the user's PowerPoint export `figures/intro_figure/AlignTac_Fig1.png` (transparent background, 2400 px wide); caption rewritten to match. Superseded versions removed: co-author's `AlignTac_Figure1_Updated.{pdf,pptx}` + caption, the v2 mock (`mock_fig1_v2.*`, `AlignTac_Figure1_v2.pptx`, `scripts/fig1_pptx.py`), and the earlier TikZ teaser (`teaser_diagram.tex`, `figures/fig_teaser.*`). Kept: thumbnail/overlay assets (`figures/teaser/`, `figures/intro_figure/{assets,v2_assets}`, `scripts/fig_teaser.py`, `scripts/fig1_assets.py`) and the co-author's generator files (`motivation_*`, `export_motivation_pptx.py`, used by `build.py --figures`). Co-author's title edit ("…Label-Efficient Contact Learning") merged.
- Session-2 unlabeled-pool RESULT (2026-10-10 22:21; held-out half of session 2, 464 curated frames; dense at best_depth.pt, pose at best_pose.pt; single seed; tags `s2half`, `s2half_bp`). cMAE / IoU / rot / trans(mm):
  K=25: real only 0.356 / 0.625 / 14.8 / 1.79; self-training 0.347 / 0.603 / 17.4 / 1.96; self-training + s2 pool 0.340 / 0.592 / 19.6 / 1.98; AlignTac 0.304 / 0.663 / 10.2 / 1.49; AlignTac + s2 pool 0.288 / 0.652 / 7.8 / 1.32.
  K=100: real only 0.366 / 0.658 / 12.4 / 1.64; self-training 0.336 / 0.679 / 14.9 / 1.85; self-training + s2 pool 0.347 / 0.668 / 14.7 / 1.84; AlignTac 0.294 / 0.710 / 9.4 / 1.54; AlignTac + s2 pool 0.269 / 0.707 / 7.4 / 1.34.
  Reading: unregistered second-session images labeled by the session-1 aligner lower contact MAE by a further 5% (K=25) / 8% (K=100) and rotation by 2.0-2.4 deg, translation by 0.2 mm; IoU unchanged (slightly lower at K=25); teacher-labeled pool gives no gain. Not yet in the paper (awaiting the user's decision).
- 2026-10-10 22:50: session-2 unlabeled-pool result IN THE PAPER (user decision; presented as a contribution). Table III bottom block = `\tabstwoadapt` (hand-added to data/tables.tex; the sensor-B block `\tabsbT` was dropped from the short version, sensor-B numbers remain in Fig. 4d-f and the text). New Sec. V-C paragraph "Using a new session without re-registration"; abstract clause (5-8% further MAE reduction); new contribution bullet 3; Table III caption rewritten. To stay at eight pages: renderer figure (old Fig. 9, `fig_renders.pdf`) removed from the short version (Fig. 1b shows PBR / learned / real at one contact; the removed block is saved in scratchpad `removed_fig_renders_block.tex` and remains in the long version), closing sentence of Sec. V-C shortened. Numbers are single-seed; seeds 1-2 of `T_k{25,100}_{rf,rf_s2}` (150 ep, `train150s`, `train_pilot_e150_pm_seed{1,2}.yaml`) started 22:26 on GPUs 0-1, evaluated by `finisher_s2seed.sh` (tags s2half / s2half_bp) -> update the block and text with three-seed means when done.

## Session-2 label-free adaptation: three seeds (2026-10-11)
Runs `pilot150_T_k{25,100}_{rf,rf_s2}{,_seed1,_seed2}` (150 epochs), tags `s2half` / `s2half_bp` in `ablation/pilot_robosoft/cross_session_results.json`, 464 held-out session-2 frames.
Three-seed means (cMAE / IoU / rot / trans mm): K=25 AlignTac 0.309 / 0.665 / 10.4 / 1.51 -> + session 2 0.290 / 0.651 / 7.9 / 1.33 (-5.9% cMAE); K=100 0.286 / 0.710 / 9.7 / 1.54 -> 0.270 / 0.708 / 7.3 / 1.35 (-5.8%).
Per-seed cMAE change: K=25 -5.5 / -5.5 / -6.7 %; K=100 -8.4 / -8.4 / -0.5 % (lower in every seed; one K=100 seed marginal). Rotation gain is the most stable (seed std 0.06-0.5 deg).
Paper: abstract and Sec. V-C say "a further 6%"; Table III bottom AlignTac rows are three-seed means (asterisk), the other rows single seed.

## Additional (non-ridge) objects on sensor A (2026-10-22)
Paper: Sec. V-C paragraph "Additional objects" (three-seed means), Fig. 6 middle group (layout wide, seed-0 models); long version Sec. "Additional Objects" + Table tab:addobj. Ridge wording removed from setup/conclusion (user: do not stress ridge objects).
Data: button, edge, hex_key, marble (meshes `gs_blender/meshes/*_v2.obj`, 3,156 contacts, `gs_blender/real_filtered_plus4`); GT = mesh at the robot pose after a per-object placement calibration (theta offset + xy, fitted to image contact masks; mean press fixed at 1.0 mm), `aligntac_newobj/calibrate_newobj.py`. 489 test frames (`real_filtered_testP4`, same index rule as testA, excluded from every component).
Runs (16 objects, K=25): `pilot150_P4_k25_{realonly,rf}{,_seed1,_seed2}`, `pilot150_P4_k25_semipm` (seed 0 only); eval `aligntac_newobj/eval_newobj.py` -> `aligntac_newobj/eval_newobj.json` (object-ID map from build_datasets!).
Three-seed means, new objects (cMAE / IoU / rot / trans mm): real only 0.167 / 0.867 / 7.65 / 0.83; AlignTac 0.146 / 0.889 / 7.16 / 0.70 (-12% cMAE; lower in every seed: 0.163/0.170/0.168 vs 0.159/0.139/0.141). Self-training (1 seed) 0.174 / 0.724 / 8.0 / 0.88.
Per object cMAE: button 0.202 -> 0.186, edge 0.131 -> 0.115, hex key 0.218 -> 0.180, marble 0.078 -> 0.083 (no gain). Button rotation ~23 deg for both (near-symmetric); edge along-edge translation unobservable.
Same models on the 12 ridge objects (testA): real only 0.287 / 6.3 deg -> AlignTac 0.243 / 2.9 deg.
Label quality (unlabeled train frames, teacher -> aligned IoU): button 0.69 -> 0.85, edge 0.84 -> 0.94, hex key 0.69 -> 0.81, marble 0.39 -> 0.93 (`aligntac_newobj/label_quality_newobj.json`).
Fig. 6 candidates: `paper_RoboSoft/scripts/fig_qual_plus4.py` (layouts wide / tall).

### 2026-10-24 correction: z-map bug, all additional-object numbers re-run
`build_real_filtered_new.get_zmap` gave every triangle its lowest vertex z (no interpolation); the coarsely meshed wedges (edge_v2 = triangular prism on a plate, hex_key_v2) became flat tops, marble off by ~0.15 mm. Fixed with exact barycentric rasterization for `*_v2` meshes only (`exact_zmap`, `use_exact_zmap`); the 12 session-1 objects keep the per-face minimum used for every published result (their pseudo-labels / aligner inputs are flattened by ~0.05 mm on ridge flanks; session-1 GT is not, session-2 GT ~0.06 mm). Recalibrated (`calib_newobj.json`; old fit `calib_newobj_facemin_zmap.json`), GT rebuilt, full chain re-run (`aligntac_newobj/chain_p4_v3.sh`, 4 GPUs); old runs kept as `*_facemin`.
Three-seed means, new objects (cMAE / IoU / rot / trans): real only 0.235 / 0.701 / 7.7 / 0.82; AlignTac 0.204 / 0.774 / 7.5 / 0.74 (-13%; lower in every seed: 0.236/0.234/0.237 vs 0.205/0.204/0.202). Self-training (1 seed) 0.225 / 0.544 / 8.3 / 0.86.
Per object cMAE: button 0.215 -> 0.186, hex key 0.317 -> 0.266, edge 0.292 -> 0.268, marble 0.090 -> 0.074 (real-only marble varies 0.074-0.114 over seeds). Label IoU teacher -> aligned: 0.62 -> 0.85, 0.43 -> 0.61, 0.41 -> 0.66, 0.33 -> 0.93 (pooled 0.43 -> 0.76). 12 session-1 objects with the same models: 0.304 -> 0.232, rot 6.1 -> 3.0.
Eval: `aligntac_newobj/eval_newobj_v2.json`. Fig. 6 middle group: author-chosen frames in `figures/qualitative_assets/fig_qual_picks.json` (button, hex key, marble; edge = median until chosen).

### 2026-10-24 session-2 GT re-rendered with the exact z-map
Session-2 GT (`real_filtered_new_curated_pred_r2_fv`, `real_s2_testhalf`, `real_s2_poolcheck`) came from `gt_depth_mm` with the per-face-minimum z-map; re-rendered at the same poses with exact rasterization (`aligntac_newobj/regen_gt_exact.py` -> `*_exact` roots; face-min re-render reproduces the stored GT to 1e-7, change 0.056 mm over contact). All 138 `cross_session_fv` / `s2half` entries re-evaluated (`aligntac_newobj/reeval_s2_exact.sh`); pose metrics identical (< 1e-3 deg), contact MAE -0.006 to -0.023 mm. `ablation/pilot_robosoft/cross_session_results.json` now holds the exact-GT values (old: `cross_session_results_facemin_gt.json`; `_bp` entries unchanged).
Paper: zero-shot AlignTac vs real only 14-21% (was 15-20%); + learned synth. K=100 0.235 mm, -32% (was 0.248, -31%); adaptation a further 5% (three-seed: K=25 0.295 -> 0.280, -5.1%; K=100 0.271 -> 0.256, -5.5%; per seed -4.3/-4.3/-6.4 and -7.8/-6.1/-2.7; was 6%), abstract and long conclusion too. Table III MAE/IoU and ranks updated by hand in data/tables.tex; long tabcross regenerated (was not hand-edited); pool label IoU 0.50 -> 0.58 (K=25), 0.56 -> 0.62 (K=100); legacy session-2 paragraph numbers updated.
- Author decision (2026-10-25): the per-face-minimum z-map inside the 12-object pseudo-labels / aligner inputs is left as is and NOT mentioned in the paper.

### 2026-10-25 figures folder layout
`figures/fig1_teaser.pdf` ... `figures/fig8_poseref.pdf` = conference-paper figures in order; `figures/long/` = extended-version-only figures; `figures/src/` = TikZ wrappers, grids, curves and PPT sources (`python build.py --figures` rebuilds fig4-6 from them). Everything else moved to `figures_archive/` (local only, not synced to Overleaf). Fig. 6 grid: `scripts/fig_qual_plus4.py paper_RoboSoft/figures/src wide`, rename to `fig6_image_grid.pdf`.

### 2026-10-25 co-author (X. Xu) figure suggestions
- Fig. 6: last row of each group now shows surface normals (mean angular error over GT contact) instead of a fourth depth row — depth MAE can understate differences (marble: MAE 0.089 vs 0.098 but normals 9.7 -> 4.0 deg). Set-level normal error real only -> AlignTac: A 27.1 -> 23.1, B 19.0 -> 15.9, new objects 20.0 -> 15.5 deg. Dumps now store normals (`dump_qual_preds.py`, `eval_newobj.py`); grid `fig_qual_plus4.py`.
- Fig. 7(b): "after 6 iterations" replaced by curves after 1/2/3/6 updates on a log axis with the no-update diagonal (`make_paper_figs.fig_conv`); same data (`aligner_convergence.json`).
- 2026-10-26 (author): Fig. 6 examples now chosen by largest real-only minus AlignTac error among frames whose AlignTac error is at most the object median (`FIG6_SELECT=gain`, default; `median` restores the old rule; the 2026-10-23 manual picks are kept in `figures/src/fig6_picks_author_2026-10-23.json`). Axis titles simplified: Fig. 4/8 x-axis "K" (caption defines K), Fig. 4 panel titles carry units, Fig. 7 y-axis "Rotation error [deg]", (b) titled "After n updates" with n at the curve ends. Fig. 7 1.5 in tall, Fig. 8 1.85 in (page budget).
- 2026-10-26 (author, round 2): Fig. 6 numbers sit in the corner with the fewest contact pixels (`free_corner`); hex key example = frame 145 (`figures/src/fig6_picks.json`). Fig. 7 fonts reduced, (b) colours saturated, n=0 label, grid ends at 30 deg (labels outside the axes). Fig. 4 tick labels 5.8 pt and headers closer; Fig. 5 now the plain matplotlib export (wrapper no longer re-typesets labels); Fig. 8 tick labels 5.8 pt.
