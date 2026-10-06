# MeshTac 全文修訂紀錄

## 2026-10-06 作者回饋後的修訂（目前版本）

- 兩版 Fig. 1 均直接引用作者指定的 `figures/fig_arch_v1.pdf`，PDF 本身未修改。Caption 補充圖中省略的 synthetic training 分支與 schematic output thumbnails。
- 保留 Claude 已恢復的 Fig. 4 四條實線，未改動該圖或生成 script。
- 摘要恢復 renderer label cost 的切入點與 24–26% depth-MSE 結果，明確指出 all-pairs renderer 超出 K 的訓練標籤預算，因此不能當成只用 K labels 的 simulation gain。
- 摘要恢復「一半訓練標註」的具體證據：sensor A 50 vs. 100（depth MSE），sensor B 25 vs. 50（contact MAE），不擴張為所有 budgets 和 metrics。
- Introduction/contributions 恢復 label-accounting study 作為獨立貢獻，不宣稱其他研究普遍洩漏測試資料。
- 結論恢復「the results indicate that mesh-pose accuracy is a key bottleneck」的詮釋，限定為利用未標註 real contacts 產生監督的問題。
- Extended version 恢復 practical recipe 和 reporting，納入 renderer-free 起點、跨 session synthetic augmentation、known-mesh pose coverage，以及各元件的資料與 checkpoint accounting。
- 五項已查證錯誤的修正、50/150 epochs 區別、validation-label 成本和 physical tuning 的待確認來源均保留。
- 編譯結果：短版 6 頁、長版 15 頁；無 undefined references、overfull boxes 或 oversized floats。

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
