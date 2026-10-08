# MeshTac 全文修訂紀錄

## 2026-10-08 六頁稿：可讀性、論證與比較條件

依作者要求閱讀目前六頁稿，修訂 `main.tex` 的 abstract 及六個短版章節。保留標題、三項貢獻的研究範圍、renderer-free 主設定、作者最新的 Fig. 1 與 0.9 text-width 顯示尺寸，以及所有既有實驗結果。長版、繪圖原始碼、圖檔、共用 `data/tables.tex` 和 `refs.bib` 未修改。本輪未執行 Overleaf pull/push。

### 文獻參考

直接閱讀以下原始論文的相關段落，參考論述組織而非複製用語；沒有把引用次數當作寫作品質排名。

| 原始論文 | 本輪閱讀與採用重點 |
|---|---|
| [DeepIM](https://arxiv.org/html/1804.00175v4) | 引言建立 initial estimate → refinement 的必要性；方法分清輸入、更新與 supervision；消融明示比較的設定；本稿突出「修正 state → 改善 label → 改善 student」這條證據鏈。 |
| [Self6D](https://arxiv.org/html/2004.06468v3) | 區分訓練使用 RGB-D 與推論只用 RGB；本稿分清 labeling、student inference 和 optional refinement，並說明相對於 pose refinement，MeshTac 的主要用途是建立 dense/pose training labels。 |
| [Sparsh](https://arxiv.org/html/2410.24090v1) | 各任務先交代評估條件，再討論結果；Discussion 分開實驗觀察、可能解釋及未驗證能力。本稿避免把 contact reconstruction 改善等同 pose 或 manipulation 都改善。 |
| [SITR](https://arxiv.org/html/2502.19638v2) | 架構、sensor-specific 評估、transfer 與 limitations 分開；本稿保持 frozen backbone 固定，明示本研究的 sensor-B 結果是重新訓練，而不是零樣本 transfer。 |
| [TACTO](https://arxiv.org/html/2012.08456v2) | 實驗先說資料、模型與比較方法，再解釋 sim-to-real 結果；本稿把數字留在關鍵比較，不逐項重述整張表。 |
| [Taxim](https://arxiv.org/html/2109.04027v2) | 區分 optical response 與 marker motion，並交代量測條件；本稿 Related Work 不再把不同 simulators/transfer methods 都視為同一種 depth-to-RGB renderer。 |
| [Church et al.](https://proceedings.mlr.press/v164/church22a.html) | 核對 real tactile RGB → simulated depth 的 transfer 方向，避免與本稿的 depth → RGB renderer 混淆。 |
| [TactGen 摘要重刊](https://ojs.aaai.org/index.php/AAAI/article/view/41424) | 核對其以 visual RGB-D 生成 tactile data、cGAN 在 simulation 中訓練的設定，移除「image translation 都需 target-sensor real labels」的概括。保留共用 bibliography 的原期刊引用。 |

### 寫法與結構

- **Abstract / Introduction**：拆開多重子句，從 real-contact registration 的成本，接到 teacher initialization、residual alignment 和 student supervision。摘要保留現有量化結果，補明 relative-to-real-only 的基準、12 ridge objects 和 sensor-specific models；跨 session 的 synthetic benefit 限定在較大的 label budget。
- **Related Work**：分成 simulation/transfer、representations、pseudo-labeling/pose refinement 三段。明確說明本稿比較的是 fitted-on-real-pairs 的 renderer，而非聲稱所有 tactile transfer 方法都需真實標註。
- **Method**：明示 perturbation 時 real RGB 固定、只有 mesh state 改變；區分四維 contact state 和三維 student planar pose。Teacher/student 的相同架構與分開訓練也直接寫出。
- **Experimental Setup**：將 L1 search 的 coarse mesh fit、567 candidates、learned-renderer RGB L1、half-depth gate 和 all-frame pose supervision 集中說清楚，保留 renderer-free 主設定。
- **Experiments**：移除連續的「This experiment asks …」模板，改成比較設定 → 關鍵結果 → 意義。保留不同 teacher、50/150 epoch 和 three-seed 的條件；不把全表數字逐一重述。
- **Discussion and Conclusion**：改為連貫段落，先回到已驗證的能力，再討論 contact-state accuracy、contact metrics 與 simulation coverage；最後接限制、uncertainty/symmetry、dynamic contact/manipulation 與 unknown-mesh future work。不再使用「Why it works / What we learned / Conclusion」提綱式小標，也不重複摘要的改善百分比。

### 實質修正／補充（不只是 rephrase）

| 修改前 | 修改後與依據 |
|---|---|
| MeshTac 有「lowest contact MAE, IoU and normal error」。 | IoU 改為 **higher**；MAE 和 normal error 才是越低越好。比較限定在 budget-matched baselines。 |
| Self-training 降低 depth MSE，「but not the contact metrics」。 | 改為 K ≤ 25 未一致改善 contact MAE/IoU；K = 50、100 確實改善，依 Table I。 |
| Adding renderer frames「changes nothing within the session」。 | 改為 K = 10 有小幅改善，跨 budgets 沒有一致的額外提升。Table I 的 0.293 → 0.288 MAE、0.660 → 0.682 IoU 不能寫成完全沒幫助。 |
| 消融表 caption 說 rows「differ only」in labels，且寫「same student」。 | 改為 student architecture/schedule 相同；正文保留 rows 1、2、4 的 renderer-augmented teacher 與 rows 3、5 的 real-only teacher 差異。指出 **rows 3 vs. 5** 才直接控制 teacher、測 alignment。 |
| Table II 的 L1-search depth source 只列 Mesh。 | Caption 補明低-error 半數用 mesh depth，另一半仍用 teacher depth；Setup 補明所有 frames 使用 search pose。表格數據／macro 未改。 |
| Setup 概括所有 pose metrics 都是 final epoch。 | 按 `build_tables.hist()`、`eval_cross_session.py`、`eval_pose_refine.py` 分開：sensor-A budget/held-out pose 是 final epoch；sensor-B/session-2 tests 使用 depth checkpoint；refinement 使用 validation rotation 選出的 pose checkpoint。主結果的 final-epoch policy 未改回 best-rotation selection。 |
| 不同表／sensor 的 translation 都叫 Trans. (mm)，未定義差異。 | `engine/eval.py` 與 `eval_pose_refine.py` 使用兩軸 absolute error 的平均；`eval_cross_session.py` 累加兩軸、除以 frame count。Setup、Fig. 4 和 Table III/IV captions 明示 mean-coordinate 與 summed-L1 的差別；沒有擅自重算或改動 raw results。 |
| Sensor B 的 learned renderer 改善「at most 10%」。 | 移除此錯誤上限；`sensorB_results.json` 的 K = 25 mesh-label baseline 0.127 → 0.1112，約 12.4%。改為比較 MeshTac、renderer、self-training 在各預算的 contact MAE。 |
| Original public-label robustness 的範圍未限定。 | 明示是 K = 25 的檢查；不暗示每個 budget 都重做。Renderer-free MAE 0.1366 → 0.1021 支持改善方向。 |
| Test-time refinement 結果推出「aligner is independent of the predictor」。 | 改為同一 aligner 改善 **兩個受測 predictor**；沒有外推到所有 predictors。Session-2 median range 也限定 K ≥ 25。 |
| 「only 30° starts leave 3°」忽略中間 perturbation 的殘差。 | 改為 15° → 0.6°、30° → 3.2° 的兩個具體比較，不聲稱只有 30° 才有殘差。依 `aligner_convergence.json`。 |
| Held-out rotation 初始數字為 64.7°。 | 依目前 Table V 的 final-epoch 值改為 **64.8°**；加入 no-held-out-mesh control 的 **50.5°**，讓 coverage 解釋有直接證據。 |
| 小節 E 稱 Label Leakage，容易被讀成 validation/test images 進入 training。 | 改為 **Renderer Baselines: Unseen Objects and Label Budgets**；正文明示 full-set reference 只用額外的 real training pairs，不是 validation images。保留此 renderer study 的獨立貢獻。 |
| 短版最後一句列 renderer fidelity 的 ρ = 0.95，但短版沒有呈現該 study 的範圍與圖。 | 從六頁版刪除此孤立的次要結果，集中在 mesh-exclusion control 和 matched budgets；沒有否定該觀察，長版和相關資料保持原樣。 |

### 仍需資料或新實驗支持的部分

- [已解決 2026-10-08] PBR 校正只用一張 no-contact 影像（自動擬合）與一張 training set 以外、單一物件的 contact 影像（手動調整），無標籤，不計入 K；已寫入兩版 setup。
- [已解決 2026-10-08] 評估協定已統一：所有表格的 pose 取 validation rotation 最佳 checkpoint，translation 為兩座標平均誤差；session-2 與 sensor-B 以 best_pose.pt 重算（tag *_bp）。
- Two-sensor evidence 是 sensor-specific training；12 ridge objects 與 quasi-static contacts 的結果不等同 unknown-object、curved contact、shear/slip 或 manipulation validation。正文維持此範圍，Future Work 不寫成已完成能力。

### 驗證

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` 成功，`main.pdf` **6 頁**；沒有 undefined references、overfull boxes 或 oversized floats。
- 檢視六頁 PDF，所有文字框均在頁面範圍內，圖、表和 captions 無截斷／重疊。保留既有 ieeeconf/caption class-recognition warning，未更改 conference class、字級或 margins。
- Hash 比對確認長版、所有圖檔、共用 table data 和 bibliography 未修改。摘要保留 21%、36%、24–26%、0–3% 等核心結果；沒有新增實驗或重建結果圖表。
- 修改前備份與逐檔 diff：`/tmp/meshtac-readability-before-n0ow7iad/`。編譯 log：`/tmp/meshtac-readability-build.log`。本輪尚未推送 Overleaf。

## 2026-10-07 Overleaf 六頁稿同步

依作者後續要求，已推送至 Overleaf commit `a233bb8c581c6511907131e5db8824fa63495dfd`。從最新遠端 `89d0ac5` 建立 checkout，確認並保留 Fig. 3 六個 panels 的更新；本次 commit 只包含 `sections/01_intro.tex`、`02_related.tex`、`03_setup.tex`、`04_method.tex`、`05_experiments.tex`、`06_conclusion.tex`。長版、摘要／main.tex、共用圖檔、table data 與 bibliography 均未納入變更。推送前在該 checkout 完成 LaTeX 編譯：6 頁，無 undefined references、overfull boxes 或 oversized floats；推送後查詢遠端，確認 main 指向此 commit。本機 `.overleaf_head` 已更新。

## 2026-10-07 六頁版逐節修訂與圖表校對（本輪）

本輪依作者要求只編輯 `sections/*.tex` 並重編 `main.pdf`。長版的 TeX/PDF、共用圖檔、`data/tables.tex`、`refs.bib` 未由本輪編輯。保留目前摘要、論文標題、三項貢獻，以及作者選定的五個實驗標題；Fig. 1 繼續使用 `figures/fig_arch.pdf`。這輪沒有執行 Overleaf pull/push。

### 閱讀的原始論文與使用方式

本輪閱讀 Introduction、Related Work、Method、實驗段落及 captions，不只閱讀摘要。參考的是問題如何收斂、比較條件如何交代，以及結果如何支持解釋，沒有搬用原文句子，也沒有把引用量當作寫法優劣的依據。

| 原始論文 | 閱讀重點與本稿採用方式 |
|---|---|
| [DeepIM](https://arxiv.org/html/1804.00175v4) | 引言先說初始估計的不足；方法逐步定義觀測、render、更新與監督。本稿依此先定義 contact state，再交代 aligner 訓練、迭代標籤生成及 student。 |
| [Self6D](https://arxiv.org/html/2004.06468v3) | 明確區分訓練時可用的 RGB-D 與推論輸入，實驗交代監督類型。本稿同樣區分 mesh-assisted labeling、一般 student inference 與額外 test-time refinement。 |
| [Sparsh](https://arxiv.org/html/2410.24090v1) | 各任務交代評估條件後討論比較；Discussion 將觀察、解釋與限制分開。本稿實驗先說怎麼比較，結論再提出 geometry/pose ambiguity 與指標選擇的 insight。 |
| [SITR](https://arxiv.org/html/2502.19638v2) | 架構 caption 交代資料如何流動；資料與訓練設定和結果分開。圖表明示 dataset、metrics、units。本稿將方法機制與 sensor-specific 實作設定分開，補 object-conditioned pose head。 |
| [TACTO](https://arxiv.org/html/2012.08456v2) | 模擬架構、實驗條件與使用範圍分開敘述；不把模擬圖像的外觀直接等同 downstream 效益。本稿對 simulation 的解釋限定在實測 budget 與任務。 |
| [Taxim](https://arxiv.org/html/2109.04027v2) | 方法分清 optical response 與 marker motion，實驗 caption 交代比較對象與量測。本稿避免將 calibration-based simulator 與 learned image translation 混為一談。 |
| [GelSlim 4.0](https://arxiv.org/html/2409.19770v1) | 核對 public geometric labels 的來源與意義；本稿保留 mesh-derived geometry 和 membrane deformation 的區分。 |
| [Church et al.](https://proceedings.mlr.press/v164/church22a.html) | 核對 transfer 方向是 real tactile image → simulated depth；Related Work 明確交代。 |
| [Xu et al.](https://proceedings.mlr.press/v205/xu23b.html) | 核對 differentiable simulation 的輸出是 tactile force field；避免籠統寫成 RGB renderer。 |
| [TactGen](https://ojs.aaai.org/index.php/AAAI/article/view/41424) | 核對以 visual RGB-D 產生 tactile images、模型在 simulation 中訓練；避免誤寫為需要 real tactile pairs 的 depth-to-RGB renderer。 |

六頁稿的 Related Work 集中在直接相關的 tactile simulation/transfer、representation 與 geometry-based refinement。刪去原本同一句中的四個泛用 vision transfer 引用，保留 tactile transfer 工作；僅短版的引用集合改變，共用 bibliography 沒有修改。

### 逐節改動

- **Introduction**：需求 → mesh registration 成本 → synthetic data 的 real-supervision budget → teacher errors 如何使 mesh labels 錯位 → MeshTac。刪去重複的方法介紹，保留三項貢獻，避免為了強調重要性增加泛泛的形容詞。
- **Related Work**：依三個問題組織，而不是連續列論文名字。具體區分各種 simulation/transfer 的輸入輸出，以及 MeshTac 與 representation learning、DeepIM、Self6D 的關係。
- **Method**：先定義四維 alignment state \((\theta,x,y,z)\) 與三維 planar pose；再說 perturbation supervision、六次更新、student 訓練及推論。補上實際使用的 depth/normal MSE、cosine rotation、L1 translation 及組內 learned weighting。
- **Experimental Setup**：分成 sensors/targets、split/budget、architecture/training、aligner、renderers/computation、metrics/checkpoint。補上 224×224 輸入與 pose head 的 object embedding；保留 real/synthetic aligner batch、50/150 epoch 差異及額外 validation-label 成本。
- **Experiments**：每節先說比較方法與控制條件，再挑關鍵結果，最後解釋意義。保留負面或不一致結果，例如高 budget 的 rotation 不一定改善，以及跨 session 的 renderer 效益取決於 budget。
- **Discussion and Conclusion**：保留 contact-state accuracy 的主要論點，新增「real image 已有 sensor appearance，因此 alignment 可補 geometry」、「repeated ridges 導致 geometry 準但 pose 不唯一」、「all-pixel MSE 受 background 主導」三個讀者可帶走的理解。提供 renderer-free 起點與使用 simulation 的條件，接回 limitations 和 future work。

### 內容上的更正／補充（不只是 rephrase）

| 原本的敘述 | 本輪修訂 |
|---|---|
| Unseen-object 表 caption 說 evaluation objects 的所有 real contacts 都未用於 training。 | 明示 matched methods 排除這些 real contacts，但 italic leaky renderer 會用這些 objects 的 real training pairs；validation 仍未拿去訓練。 |
| Test-time refinement 無條件寫 within-session mean 為 0.4–1.2°。 | 限定 **K ≥ 25**；K=10 的數字並不在此範圍。 |
| Session-2 medians 寫為 1.5–2.2°，但同段已限定 K ≥ 25。 | 依 `pose_refine_results.json` 對應六個 rows，修正為 **1.5–1.8°**；2.2° 不屬於這個 budget 範圍。 |
| 跨 session 籠統說 renderer 提供改善。 | K=100 有益；K=25 移除 renderer 的 contact MAE 略低（0.306 vs. 0.310）。結論改成收益依 budget 而異。 |
| Predictor architecture 沒說 object identity 實際如何進入模型。 | 依 `SingleEncoderModel` 與 YAML 補上 learned object embedding **conditions the pose head**，沒有宣稱 DPT 同樣接受此 embedding。 |
| Method 沒說 student 使用哪些 loss。 | 依 training YAML 與 `vistacfusion/losses/` 補齊實際 objective；沒有修改訓練。 |
| L1-search 半數 gate 的 caption 只說 mesh targets。 | 明示每個 object 的低 render-error 半數用 mesh depth、其他用 teacher depth，但所有 unlabeled frames 都用 L1-search pose。實際 `real_pseudo_k25_rcp/pseudo_manifest.json` 為 `keep_frac=0.5, pose_all=True`。 |
| Fig. 3 與 qualitative caption 對 sensor-B 的指引容易混淆。 | 按目前六-panel 圖核對 (a–c) sensor A、(d–f) sensor B，並區分 B 曲線的 renderer-free 模型與 qualitative examples 的含-renderer 模型。 |

原始 table macros、數值、training split、run/checkpoint 與實驗程式未由本輪修改；新文字中的百分比來自現有結果。

### 圖表呈現

- Fig. 1 保留作者指定 PDF；caption 說明 teacher/aligner/student、共享架構、optional refinement 與圖中省略的 synthetic branch。
- Fig. 2 明示欄位、K、sensor、median-frame selection 規則與 depth 單位。
- Fig. 3 的短版顯示寬度由 0.86 增至 0.90 text width；caption 補 panel/split、label budget、epochs 和 leaky 例外。
- Fig. 4 由 0.90 增至整個 column width，提高字與曲線的可讀性；caption 明示 withheld GT 僅評估 label quality。
- Fig. 5 保留原顯示寬度；交代 median-gain selection、MAE 單位、共用 color scale，以及模型 variant。
- Table II 縮小 column padding，使表格縮放後字級更大；區分 pseudo-label quality 與 student validation。
- Table III 用兩個明確 group headings 區分第二 session 與 sensor-specific sensor-B training，再列 Method/metrics；three-seed 統計只套用 sensor-B 部分。
- Table IV/V 的 captions 補足 refinement iterations、metric aggregation，以及 held-out meshes 和 leaky reference 的資料例外。
- 短版 table floats 允許在欄內、欄首或欄尾排放，使表格接近相應文字。本文字級、頁面邊界及 caption 字級沒有調小。

### 驗證與並行更新紀錄

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` 成功；`main.pdf` **6 頁，含 23 項 references**。
- 無 undefined references、overfull boxes、oversized floats；文字框均在頁面內；逐頁檢視六頁 PDF，表格與 captions 無重疊或截斷。
- `caption` package 保留原有的 ieeeconf class-recognition warning；沒有因此改 conference class 或 typography。
- `main.tex` 與本輪開始時相同，摘要、title、匿名 metadata、字級、margins 均保留。
- 初始 hash 比對偵測到 16:49–16:50 的另一次更新：`sections_long/05_experiments.tex`、`main_long.pdf` 及若干共用 figure PDF 有變動。本輪未寫入這些檔案，也未撤回更新；重新從目前圖檔與生成 script 核對 captions。以該更新後狀態再次核對 22 個 protected files，後續沒有變更。
- 原稿與初始 hashes：`/tmp/meshtac-short-full-prose-before/`。編譯 log：`/tmp/meshtac-short-full-prose-build.log`。

### 仍待作者確認

Physical renderer 的手動 contact tuning 使用哪些影像、屬於哪個 split，仍沒有已確認的來源。本輪保留手動調參說明，沒有替它宣稱嚴格的 K-label isolation。這是資料來源問題，文字修飾無法代替確認。



## 2026-10-07 摘要結構與流暢度

依作者要求直接閱讀相關論文的摘要，重新組織短版及長版 abstract。參考的是各篇如何銜接問題、機制及證據，沒有移植原文句子。

| 論文 | 摘要結構的觀察 |
|---|---|
| [DeepIM](https://arxiv.org/abs/1804.00175) | 問題與現有估計限制之後，依序交代初始估計、迭代匹配和驗證範圍。 |
| [Self6D](https://arxiv.org/abs/2004.06468) | 從標註成本接到方法，訓練階段的介紹直接服務於如何利用未標註資料。 |
| [Sparsh](https://arxiv.org/abs/2410.24090) | 將標註與感測器差異接到學習方式，再用 benchmark 和結果支持主張。 |
| [SITR](https://arxiv.org/abs/2502.19638) | 圍繞感測器間的 transferability 組織限制、方法和驗證；不展開所有訓練細節。 |
| [TACTO](https://arxiv.org/abs/2012.08456) | 從模擬需求及缺口接到系統能力與實際驗證。 |
| [Taxim](https://arxiv.org/abs/2109.04027) | 交代模擬問題後介紹模型及校正成本，較詳細地描述光學與 marker 機制。 |
| [SimGAN](https://arxiv.org/abs/1612.07828) | 從 synthetic--real 差異接到方法目標，再說明設計如何支持該目標及 downstream 驗證。 |

本稿採用 registration 問題 → MeshTac 機制 → matched-budget 比較與結果 → alignment 與 renderer supervision 的意義。MeshTac 提前至第二句；test-time refinement 接在方法介紹後，摘要最後以實驗 insight 收束。保留既有兩組量化結果，新增跨 session 的定性發現，明確限定 ridge-pattern objects 及分別訓練的 sensor-specific models。圖表及實驗數字未修改。兩版摘要約 207 字；編譯後短版維持 6 頁、長版 15 頁，無 overfull 或未解析引用。

## 2026-10-06 三項貢獻與風險陳述（目前版本）

- 將原本四項貢獻整合為三項較完整的主張：**MeshTac 方法**、**matched-budget label accounting**、以及涵蓋兩個 sensors、第二個 session、held-out objects、消融與 test-time refinement 的**系統性實證**。原第 4 點 test-time refinement 併入實證，不再獨立包裝為同等層級的方法貢獻。
- 摘要與引言在開頭明確限定為 **known-object tactile perception**：object identity 與 mesh 已知，而昂貴的是把每張真實 contact 註冊到 mesh。
- 與一般 pose refinement 的差異寫得更明確：aligner 的主要用途是把 refined mesh state 轉成 depth、normal、pose training targets；student 在標準推論時不需要 mesh input。相同 aligner 的 test-time refinement 是額外用途。
- 限制段明確交代：evaluated predictor 使用 object identity；mesh 用於 pseudo-labeling 與 optional refinement；mesh-derived targets 衡量 registered object geometry，不是實際 elastomer deformation。
- 將 ridge objects 說明為受控的 contact-localization/pose 測試，不宣稱可推廣到 curved、textured 或 unknown objects；也直接指出本文驗證 perception，尚未證明 downstream manipulation 效益。
- 未修改實驗數字、結果表格或圖檔。短版維持 6 頁，長版維持 15 頁。
- 已同步至 Overleaf commit `58fa81c`。

## 2026-10-06 文獻寫法參考與全文修飾（最新）

依作者最新要求，兩版摘要改為約 198 字的定性敘述，不列改善百分比、K 值、誤差或測試角度。原先數字仍在實驗正文與表格；「一半標註」的具體比較保留於 label-efficiency 段落。這項選擇取代前一輪「在摘要恢復數字」的編輯方向。

### 本輪參考的原始論文與採用方式

這些是與研究主題直接相關的代表性論文，未按引用次數排名；參考的是論述組織，沒有移植原文句子。

| 原始論文 | 閱讀重點 | 本稿採用方式 |
|---|---|---|
| [TACTO](https://arxiv.org/html/2012.08456v2) | 摘要從模擬需求、缺口帶到系統能力；實驗分清設定與結果；結論回到已驗證能力 | 每個實驗先交代比較目的，再給結果和解釋；結論不再重複一串指標 |
| [Taxim](https://arxiv.org/html/2109.04027v2) | 方法先交代整體流程，再分解光學與 marker 模型；結論銜接能力及延伸方向 | 方法先建立 contact state 與 mesh labels 的關係，再說 aligner 訓練、標籤生成與 student |
| [DeepIM](https://arxiv.org/html/1804.00175v4) | 摘要以問題、迭代修正機制與驗證範圍為主，不列性能百分比；引言解釋初始估計為何需要 refinement | 摘要提早介紹 MeshTac，保留 pose bottleneck 論點；引言建立 pose error → dense-label error 的因果鏈 |
| [Sparsh](https://arxiv.org/html/2410.24090v1) | 從觸覺資料與標註限制建立動機；Discussion 分開主要結果、解釋與限制 | 區分實驗觀察與可能原因，保留 GT 定義、validation 成本、失敗及幾何範圍限制 |

這些論文對摘要是否列數字的選擇不同；本稿採定性摘要是配合作者偏好與論述主線，不是宣稱頂會論文一律不放數字。

### 修改內容

- **摘要**：mesh supervision 的問題 → teacher / residual aligner / student → qualitative evidence → renderer label accounting 與適用情境。保留 label-accounting 研究發現，縮減支線細節。
- **引言**：從 tactile geometry 的需求收斂至註冊成本，再說明 renderer budget 的問題與 alignment 動機。本輪當時保留四項貢獻，已由上方最新修訂整合為三項。
- **Related work**：由逐項列舉改成依問題組織，明確交代 simulation、representation、pseudo-labeling 與 pose refinement 和本稿的關係。
- **方法／設定**：新增簡短流程導讀，改善長句及不完整句子；主 aligner 的 real/synthetic 資料、訓練預算與 checkpoint 規則均保留。
- **實驗**：各節先說比較目的；詳細版將原本一段內的六項 ablation 拆成 aligner training、synthetic student data、unlabeled data/filtering、repeated alignment 等段落。合併重複跨 session 敘述，保留實驗數值及比較。
- **結論**：以方法貢獻和實驗支持的理解收束，不再重述摘要數字；保留 mesh-pose bottleneck、synthetic data 跨 session 的作用、限制與後續方向。長版 practical recipe/reporting 未刪除。
- **順手修正的內容範圍**：長版 cross-session 表 caption 原仍寫「full session-1 real set」，補正為九物體並註明 depth checkpoint；few-shot renderer 的「不改善 dense error」改成「不同時改善 contact MAE 和 IoU」，避免與 MSE 小幅下降矛盾。若干未經單獨控制實驗驗證的機制解釋改為明示推測。
- **圖表**：所有圖檔、`data/tables.tex` 和 bibliography 均逐檔 hash 確認未變；Fig. 1 持續使用作者指定的 `fig_arch.pdf`，Fig. 4 保留全實線。
- **驗證**：短版 6 頁、長版 15 頁；兩版 LaTeX 編譯成功，無 undefined references、overfull boxes、oversized floats；文字框均在頁面內，並檢視短版首頁、Fig. 1 頁及結論頁。
- **Overleaf**：已同步本輪修訂 `279b89f` 與最後校對 `445059e`。
- **修訂前備份**：`overleaf_backups/meshtac-before-prose-2026-10-06.zip`（含本輪修改前的 TeX 與說明文件）。


## 2026-10-06 作者回饋後的修訂（前一輪；摘要由上方更新）

- 依作者最後指定，兩版 Fig. 1 均直接引用最新的 `figures/fig_arch.pdf`（取代本輪稍早的 v1 選擇），PDF 本身未修改。Caption 補充圖中省略的 synthetic training 分支；新版 student-output thumbnails 是 `make_arch_assets.py` 產生的實際預測，保留這項說明。
- 保留 Claude 已恢復的 Fig. 4 四條實線，未改動該圖或生成 script。
- 摘要恢復 renderer label cost 的切入點與 24–26% depth-MSE 結果，明確指出 all-pairs renderer 超出 K 的訓練標籤預算，因此不能當成只用 K labels 的 simulation gain。
- 摘要恢復「一半訓練標註」的具體證據：sensor A 50 vs. 100（depth MSE），sensor B 25 vs. 50（contact MAE），不擴張為所有 budgets 和 metrics。
- Introduction/contributions 恢復 label-accounting study 作為獨立貢獻，不宣稱其他研究普遍洩漏測試資料。
- 結論恢復「the results indicate that mesh-pose accuracy is a key bottleneck」的詮釋，限定為利用未標註 real contacts 產生監督的問題。
- Extended version 恢復 practical recipe 和 reporting，納入 renderer-free 起點、跨 session synthetic augmentation、known-mesh pose coverage，以及各元件的資料與 checkpoint accounting。
- 五項已查證錯誤的修正、50/150 epochs 區別、validation-label 成本和 physical tuning 的待確認來源均保留。
- Introduction 進一步把方法動機接回實驗發現：pose 誤差會使 mesh contact 錯位並污染 dense labels，因此 alignment 是利用額外真實影像的關鍵。
- 編譯結果：短版 6 頁、長版 15 頁；無 undefined references、overfull boxes 或 oversized floats。
- 已同步至 Overleaf：`ee664fe`（引言與所選 Fig. 1 的圖說）；指定的 Fig. 1 PDF 與 Fig. 4 均未改動。

## 2026-10-05 首輪修訂（歷史紀錄；部分編輯取捨已由上方更新）

本輪以 RoboSoft 六頁稿為主，同步修訂 extended version。原始實驗數值、模型 checkpoint、資料切分與訓練程式未改；圖表由既有結果重建。這份文件記錄編輯判斷和仍需作者確認的實驗來源，不是新的實驗報告。

## 文章主線與寫法

- 摘要改以 MeshTac 的問題、方法、主要結果為主，移除無法統一驗證的「2–4×」、籠統的「halves the labels」和對領域慣例的批評。
- Introduction 從軟膠觸覺的物理量測與 pose registration 成本切入，再介紹如何用 mesh 補足未標註影像的幾何資訊。
- Related work 分清 optical rendering、elastomer simulation、image translation、representation 與 pose refinement。Church et al. 的工作不再被當作 depth-to-real-image renderer 的直接例子；Taxim 也不再被引作僅需一張背景影像的校正法。
- 方法補上 residual normalization、indentation clipping、sensor-B teacher 與 renderer-free 分支。明確區分 student inference 和額外的 mesh-assisted pose refinement。
- 實驗先說比較條件，再給結果與解釋；移除「obvious」「snaps」「exact geometry」「cost nothing」「earns its place」等含糊或帶推銷感的寫法。負面結果限制在本資料與已測設定。

參考的原始論文：

- [TACTO](https://arxiv.org/html/2012.08456v2)：方法敘述與設計範圍的組織方式。
- [Taxim](https://arxiv.org/html/2109.04027v2)：把光學與形變模型分開交代，並明確說明校正資料。
- [GelSlim 4.0](https://arxiv.org/html/2409.19770v1)：從機器人感知需求銜接感測器與驗證；並核對 William van den Bogert 的姓名。
- [Church et al.](https://proceedings.mlr.press/v164/church22a.html)：真實觸覺影像到模擬深度的轉換方向。
- [SITR](https://openreview.net/pdf?id=RnJY9WcpA3)：本稿使用的表示學習 backbone。

這些是結構與技術定位的參考，沒有搬用原文段落，也沒有以引用量作為論證依據。

## 實質更正

1. **K-shot 圖 caption 貼錯**：短版原寫 unseen objects，長版原寫 qualitative results；兩版皆改為 label-efficiency curves。
2. **兩種資料越界不同**：GAN-full 在 K-shot 使用超出 K 的 training labels，並未使用 validation frames；在 LOO 則使用了 held-out objects 的真實資料。刪除「memorized the test contacts」等無依據說法。
3. **跨 session 的 `all` baseline 實際只有九個物件**：由 `data_A_realonly.yaml` 的 `test_objects` 核對。修正生成表格的 script、表格與文字，移除把它當成所有物件全部標籤 baseline 的主要結論。
4. **A 主實驗 150 epochs，seed/ablation/B 為 50**：補齊 caption；K=10 GAN-full 亦為 50 epochs。A 的統計結果不再看似來自 150-epoch 多 seed 實驗。
5. **sensor B 是重新訓練**：不再寫成未重新訓練的跨感測器遷移。主要曲線/小表是 renderer-free，而三 seed 數字與 qualitative 圖中的 B 模型含 renderer；已逐處區分。
6. **validation 標籤不在 K 內**：明示 K 是 training-label budget，checkpoint selection 額外使用 validation labels。不能把 K 解讀為整個專案總共取得的標籤數。
7. **L1 search 有較小幅的 contact 改善**：刪除「只有 aligner 是第一個改善 contact 的方法」；93% 數字描述 image-score 排序，不是 pose 失敗率。
8. **renderer-free 跨 session 仍比 self-training 有較低 MAE**：0.288 < 0.300，IoU 同為 0.707；改成它比含 renderer 的 MeshTac 差，而不是完全失去對 self-training 的優勢。
9. **aligner 訓練也有 synthetic 分支**：`real_pseudo_k25_al/pseudo_manifest.json` 指向 `outputs/aligner_k25/aligner.pt`；`aligner_k{10,25,50,100}.log` 均有 finite synthetic loss。與程式預設及加權 loss 相符，A 主模型每批 24 real + 40 synthetic；renderer-free 與 B log 為 real-only。原筆記的「主 aligner 只看 K 張 real frames」不夠精確；正確說法是只使用同一組 K 個 **real labels**，主模型另有來自該 K-pair renderer 的合成訓練例子。
10. **評估 checkpoint**：A main/pose-refinement 的 pose 用 best_pose；legacy cross-session 表的 pose 由 best_depth checkpoint 計算。避免把這两套角度混用。

## 圖表與排版

- 重畫 vector overview，明確標示迭代、student、optional synthetic data 與 optional test-time mesh refinement；移除原先用 GT 冒充輸出外觀的示意 thumbnail。
- Qualitative 與 pseudo-label 圖加入共同深度色階（0–1.2 mm，超過上限以端點色表示），保留既有影像與按規則選擇的 frame；數字加底色提高可讀性。
- Label-quality 圖以虛線/實線區別 teacher/aligned labels，曲線標出較佳方向。
- 縮短 ablation 表的 row labels，增加數字實際顯示字級。
- 長版完整 K-shot 表原超出頁面；改成兩張跨欄表，保留所有 row。多 seed 表也改為跨欄。
- PDF 隱藏超連結彩框；匿名版不輸出作者姓名、email 或 PDF author metadata。

## 還需要作者確認的事

**Physical renderer 的 contact tuning 來源尚未確定。** 舊 README 說只用一張 no-contact frame，但長版同時記錄 gel/contact parameters 曾根據幾張 contact frames 手動調整。現在稿件保留「背景影像自動校正 + contact 參數手動調整」，不再宣稱整個 renderer 完全沒看過 contact。仍需確認那些影像的物件與 split，才能判定 physical baseline 的嚴格資料隔離。已在對話中詢問；沒有替作者猜測答案。

現有資料仍有 12 個 ridge-pattern objects、mesh-derived GT、逐幀 validation、沒有 manipulation task 等限制。這些已在稿件保留；文字修訂不能替代額外實驗。若要進一步增強投稿，較有價值的是獨立少標籤 validation 的預算檢查、對齊失敗案例/成功率，以及一個機器人任務，而不是增加宣傳性措辭。

## 投稿模式

[RoboSoft 2027 CFP](https://robosoft27.com/call-for-papers/) 要求 double-anonymous review；六頁含 references，最多兩張付費額外頁。`main.tex` 和 `main_long.tex` 預設匿名，保留作者分支，尚未執行 PaperPlaza 投稿。

```sh
# 投稿 PDF
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
# 作者閱讀版，輸出到另一個 jobname，不覆寫投稿 PDF
pdflatex -jobname=main_author '\def\MeshTacAuthorVersion{1}\input{main.tex}'
bibtex main_author
pdflatex -jobname=main_author '\def\MeshTacAuthorVersion{1}\input{main.tex}'
pdflatex -jobname=main_author '\def\MeshTacAuthorVersion{1}\input{main.tex}'
```

官網截止時間文字互有出入：[首頁](https://robosoft27.com/) 寫 2026-10-15 23:59 Pacific，CFP 寫 all times GMT。日期一致，時區應以 PaperPlaza 實際顯示為準；不要僅依舊筆記的 PT 安排最後上傳。

## 驗證與同步

- `main.pdf`：6 頁；`main_long.pdf`：14 頁；皆為 US Letter。
- 兩版成功完成 LaTeX/BibTeX 編譯；無 undefined references、overfull boxes 或 oversized floats。PDF text bounding boxes 均在頁面內。
- 原有 18 個 table macros 的小數數值與修訂前一致；新增兩個 macro 只拆分長表。
- Overleaf 已同步至 commit `6a5fcdf`。編輯紀錄與生成 script 保留於本機 repo。
- 原稿備份：`../overleaf_backups/meshtac-before-editorial-2026-10-05.zip`。
