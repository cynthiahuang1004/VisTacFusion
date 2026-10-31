# AlignTac 去掉已知 mesh：文獻評述與可行路線

日期：2026-10-11。目的：回應「把 known reference object geometry 去掉，改用 visual 3D foundation model 重建的幾何」這個方向，整理 2024–2026 的相關文獻，評估可行性，並提出實驗路線。數字皆取自各論文自述，未經獨立驗證。

---

## 1. 現在 AlignTac 哪裡用到 mesh

| 階段 | mesh 的用途 | 去掉後需要的替代 |
|---|---|---|
| GT label（K 個 labeled contacts） | mesh 在校正後的 q* 投影出 depth / normals / pose | 仍需要某種幾何來產生 K 個 label；或改成由感測器校正直接量 depth |
| Aligner 訓練 | mesh 在 q*+δ 的投影當作「放歪的候選 depth」 | 任何可投影的幾何表示（mesh / SDF / 點雲） |
| Pseudo-label | mesh 在 q_T 的投影 | 同上 |
| 可選 test-time refinement | 同上 | 同上 |

重點：AlignTac 真正依賴的是「一個可以在任意接觸狀態 q 下投影出 depth 的幾何表示」，不一定要是 CAD。只要重建的幾何在**接觸區域**的局部精度夠，pipeline 其餘部分不用動。因此問題拆成兩個：

1. **幾何從哪裡來、精度多少**（視覺基礎模型、掃描、觸覺重建）。
2. **幾何不夠準時怎麼辦**（讓觸覺把幾何修到夠準）。

## 2. 我們需要的幾何精度

- 論文的 contact IoU 用 depth > 0.05 mm 當接觸閾值；K=25 時 pseudo-label contact MAE 0.23 mm、student contact MAE 0.23–0.29 mm。
- Aligner 修正的是 4 維狀態（θ, x, y, z），修正範圍 ±8°、±1.5 mm、±0.15 mm。
- 現有 12 個物件是 3D 列印的 ridge pattern，ridge 的寬度與高度都是 mm 等級；壓入深度約 1 mm。

推論：要在**不改動 aligner** 的情況下換掉 mesh，重建幾何在接觸區域的誤差需要壓到約 0.1–0.2 mm；這比目前任何純視覺重建在物件尺度上做到的精度（約 1 mm，見第 3 節）小一個數量級。對特徵較大的日常物件（邊緣、把手、按鍵、凹槽），要求會寬鬆一些，但 ridge 類物件肯定不夠。這決定了第 5 節的路線設計。

## 3. 文獻整理

### 3.1 視覺 3D 基礎模型（幾何的來源）

**多視角 feed-forward 模型**（輸入數張到數百張照片，直接輸出 depth / point map / 相機）

- 綜述：[Review of feed-forward 3D reconstruction: from DUSt3R to VGGT](https://arxiv.org/pdf/2507.08448)；[Feed-forward 3D reconstruction with point-cloud representations (2026)](https://www.sciencedirect.com/science/article/pii/S2096579626000203)。
- [Depth Anything 3 (DA3, 2511.10647)](https://arxiv.org/abs/2511.10647v1)：任意張數、有無相機位姿皆可；DTU 物件重建 Chamfer（mm）：DA3-Large 1.23（有位姿）/ 2.08（無位姿），VGGT 1.44 / 2.05，π³ 1.72 / 3.28，MapAnything 3.97 / 7.91，DUSt3R 約 7.6–8.0。主模型輸出**相對尺度**，metric 版本另有變體。
- [VGGT](https://arxiv.org/pdf/2605.15195)（及 2026 的 VGGT-Ω、VGG-T³）、π³、MapAnything：同一類。第三方分析在 DTU 上報 VGGT 0.59–0.97 mm、MASt3R 0.37 mm、COLMAP 0.53 mm（[Uncertainty Quality of VGGT](https://arxiv.org/pdf/2606.16479)，作者提醒各家評測子集不同、不能直接比）。
- 機器人應用上已有人用 VGGT 做數位分身，但都要**另外處理尺度**（例如用已知相機基線對齊，或 [DREAM](https://arxiv.org/pdf/2608.29078) 用校正過的 wrist camera 做 metric 對齊）。
- 結論：DTU 的物件約 10–20 cm，1–2 mm 的 Chamfer 相當於 1% 尺度誤差；足以給出物件的整體形狀、ridge 的佈局與方向，但**不足以給出 ridge 的剖面**。

**單張影像生 3D（生成式）**

- [SAM 3D / SAM 3D Objects (2511.16624, CVPR 2026)](https://arxiv.org/pdf/2511.16624)：單張照片＋mask → 帶材質的 mesh、含相機座標下的位姿與佈局；評測以人類偏好為主（5:1 勝率），幾何指標在 SA-3DAO 上 F-score 約 0.23（二手來源）。一篇實驗室機器人的評測指出它「幾秒內給出合理幾何，但細節不符會影響操作」（[2608.26383](https://arxiv.org/abs/2608.26383)）。
- [TRELLIS / TRELLIS 2](https://arxiv.org/pdf/2510.23880)、[Hunyuan3D 2.1](https://arxiv.org/pdf/2506.15442)、[Hunyuan3D-Buffalo 1.0 (2608.02711)](https://arxiv.org/pdf/2608.02711)、InstantMesh、Wonder3D：以外觀為目標訓練，幾何通常**過度平滑**（Tactile DreamFusion 的動機正是「diffusion 生成的 3D 缺乏真實細緻幾何，表面過平、細節被塞進 albedo」）。[MM-TRELLIS](https://arxiv.org/pdf/2606.24301) 報告 TRELLIS 單視角有比例失真（車輛偏瘦）。
- 尺度：生成模型輸出在 [-1,1] 的正規化框內，**沒有 metric 尺度**；需要外部資訊（RGB-D、已知物件尺寸、機器人運動學）。

**單張 metric 深度**

- [MoGe-2 (2507.02546, NeurIPS 2025)](https://arxiv.org/abs/2507.02546)：單張 → metric point map，另有 global scale 分支；適合給上面的模型補尺度，或直接給接觸區域一個粗深度。

### 3.2 用「重建 / 掃描的幾何」取代 CAD 做觸覺任務的先例

- [YOTO: You Only Touch Once (2606.28899, 2026)](https://arxiv.org/html/2606.28899v1)：兩個 GelSight Mini 單次接觸估 6-DoF 物件位姿。**直接比較 CAD mesh 與消費級掃描 mesh**（KIRI Engine app、iPad，手持 20–30 cm）：平均接觸定位誤差 CAD 5.45 mm vs 掃描 4.78 mm，掃描甚至略好（作者歸因於列印瑕疵被觸覺感到）。每個物件仍需 20 個真實接觸做 fine-tune。這是最直接的證據：**對 mm 等級的任務，掃描幾何可以取代 CAD**；但他們的精度要求比我們鬆 20 倍以上。
- [Any6D (CVPR 2025)](https://arxiv.org/html/2503.18673v2)：單張 RGB-D anchor → InstantMesh 生成正規化形狀 → 用深度做 bounding-box 對齊與逐軸尺寸搜尋取回 metric 尺度 → FoundationPose 的 render-and-compare 估位姿。作者自列的限制：「初始 3D 形狀不準時無能為力，因為沒有 shape updating」。這正好是觸覺能補的缺口。
- [ViTa-Zero (ICRA 2025)](https://arxiv.org/pdf/2504.13179)：以視覺位姿估計（FoundationPose/MegaPose）為骨幹，用觸覺與本體感覺的物理約束（彈簧質量模型）在測試時修正位姿；比 FoundationPose 的 ADD AUC 高 60%。思路與 AlignTac 的 refinement 相近，但沒有產生 dense label。
- [TouchScale (2610.10288)](https://arxiv.org/abs/2610.10288)：500 小時人類視觸覺資料；物件 mesh 由多張 RGB 丟給 Meshy 多圖轉 3D API 重建，隨資料集釋出；沒有報告 mesh 精度，也沒用它產生觸覺 label。顯示「用生成式 API 重建物件」已是資料集建置的常規做法。
- [Shared visuo-tactile interactive perception (IJRR 2025)](https://journals.sagepub.com/doi/10.1177/02783649241301443)：不假設物件模型，邊探索邊重建並估位姿與尺度。

### 3.3 用觸覺補足視覺幾何的細節（我們最該借鏡的一類）

- [Tactile DreamFusion (2412.06785)](https://arxiv.org/html/2412.06785v1)：Wonder3D 生 base mesh（附錄也試 InstantMesh、RichDreamer），GelSight Mini（約 85 µm/pixel）手動壓 5 個 patch，Poisson 積分得高度、高通濾波留紋理，轉成 normal 紋理用 image quilting 貼到 UV，再以 diffusion 做視觸覺聯合最佳化。**沒有把觸覺 patch 註冊到 mesh 的特定位置**，細節是「紋理層級」而非「幾何層級」。限制：粗形狀受生成器限制、UV 接縫。
- [GelSLAM (2508.15990, 2025)](https://arxiv.org/html/2508.15990v2)：純觸覺 SLAM。GelSight Mini，用 6.31 mm 金屬球 50 張影像校正一個三層 MLP（RGB＋像素座標 → 梯度），Poisson 求高度、曲率做特徵；連續接觸追蹤＋loop closure。10 個列印物件（平均 18 mm）Chamfer 平均 0.6 mm（0.26–0.91）。限制：需要連續滑動接觸、低紋理物件（蛋）無法閉環、重複圖案造成假閉環、細節因解析度與位姿誤差被略微平滑。作者自己建議「加入視覺」。
- [InvariantCloud (2605.25216, 2026)](https://arxiv.org/pdf/2605.25216)：GelSight Mini 多次離散接觸重建輪廓，不需外部動捕。
- [NeuralFeels (Science Robotics 2024)](https://arxiv.org/pdf/2312.13469)：多指手上視觸覺線上建 neural field，F-score 81%、位姿漂移 4.7 mm（已知模型時 2.3 mm）。
- [Touch-GS (2024)](https://arxiv.org/html/2403.09875v3)、[Touch-Augmented Gaussian Splatting (2508.07717, 2025)](https://arxiv.org/html/2508.07717v1)：用觸覺監督 3DGS；後者在模擬中把 Chamfer 從 7–30 降到 0.7–5（單位依場景），但觸覺是從 GT mesh 取樣的理想訊號，沒有真實硬體。
- [TouchSDF (RA-L 2024)](https://arxiv.org/pdf/2311.12602)、[Touch2Shape (2505.13091, 2025)](https://arxiv.org/abs/2505.13091)：觸覺條件下的 DeepSDF / 3D diffusion 形狀重建與探索。
- [Proactive tactile exploration from minimal visual priors (2505.11975)](https://arxiv.org/abs/2505.11975)：從單一 primitive template 開始，以接觸修正 mesh 的局部變形，主動挑選接觸點降低不確定性。
- [VisTa3D (2608.20740, 2026)](https://arxiv.org/abs/2608.20740)：薄物件重建基準；現有視覺與點雲重建模型在薄結構上保真度低，提出以觸覺響應圖補充。對我們的 ridge 物件是直接的警訊。
- [TaRF (CVPR 2024)](https://arxiv.org/html/2405.04534v1)：NeRF＋條件 diffusion，從視覺預測場景中任意點的觸覺影像。方向相反（視覺→觸覺外觀），但證明視觸覺可以在共同 3D 座標下對齊。

### 3.4 感測器端的「去校正」（和去 mesh 互補）

- [3D-Cal (2511.03078)](https://arxiv.org/abs/2511.03078)：把 3D 印表機改成自動探針收校正資料，訓練 CNN 做 depth；DIGIT 156 µm、GelSight Mini 205 µm。這是「直接量 depth、不用 mesh」的路線，但需要每顆感測器做一次探針校正。
- [Calibration-free surface normals via universal photometric stereo (2609.31754, 2026)](https://arxiv.org/abs/2609.31754)：用純合成資料訓練的通用光度立體網路估觸覺 normal，三種感測器免校正；金屬球上與校正法相當（6.56°），自然紋理 10.66°。可當作觸覺 patch 的局部幾何來源，免去 3D-Cal 式校正。
- [SITR (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/f05a11e682bf9c57ab62f7cff08fbe5b-Paper-Conference.pdf)、Sparsh、[TacVerse (2026)](https://www.researchgate.net/publication/408046993_TacVerse_A_Multi-Sensor_Dataset_and_Benchmark_for_Cross-Sensor_Vision-Based_Tactile_Perception)、UniTac：跨感測器表徵，不直接解決幾何來源。

### 3.5 文獻上的空缺（我們可以站的位置）

1. 沒有人把「視覺重建的粗幾何」當作**可投影的 label 來源**，再用對齊後的觸覺觀測把它修到 label 可用的精度。Any6D 明說沒有 shape updating；Tactile DreamFusion 不做註冊；Touch-GS 只在模擬；GelSLAM 沒有視覺先驗。
2. 沒有人量化「幾何誤差 → pseudo-label 品質 → student 準確度」的關係。這是決定整條路線可不可行的第一個實驗，而且**用現有 CAD 資料就能做**。
3. 觸覺 pseudo-label 工作（含我們自己）都綁定已知 mesh；一旦幾何來自重建，K 個 label 的 GT 本身也變成「重建幾何在校正位姿的投影」，GT 的定義需要重新說清楚。

## 4. 候選模型

| 用途 | 候選 | 輸入 | 輸出 | 尺度 | 備註 |
|---|---|---|---|---|---|
| 多視角重建 | DA3-Large / VGGT / π³ | 繞物件一圈的照片（wrist cam 或手機） | depth、點雲、相機 | 相對；需用機器人運動學或已知基線對齊 | 精度最好、可直接出接觸區的局部 depth；需要再做 mesh 化（Poisson / TSDF） |
| 單張生成 | SAM 3D Objects / TRELLIS 2 / Hunyuan3D 2.1 | 一張照片＋mask | 封閉 mesh | 無；用 Any6D 的對齊法或已知尺寸 | 一張就能用、拓樸完整，但細節平滑 |
| 單張 metric 深度 | MoGe-2 | 一張照片 | metric point map | 有 | 補尺度，或給接觸面一個粗深度 |
| 掃描基準 | KIRI Engine / Polycam / 結構光掃描器 | 手機繞拍 | mesh | 有（app 內建） | 當作日常物件的「參考幾何」或 pseudo-GT |
| 觸覺局部幾何 | 校正式（3D-Cal、現有 GelSlim 校正）或免校正（universal PS） | 單張觸覺影像 | normal / depth patch | 有（感測器解析度已知） | 用來修正粗 mesh |

授權狀態各家不同（Meta 的 SAM 系列、VGGT 為研究授權；DA3、MoGe-2、TRELLIS 較寬鬆），實作前要確認。

## 5. 三條路線

### 路線 A：直接代入（AlignTac-V，最小改動）

視覺重建 mesh → 對齊到機器人座標（尺度＋初始位姿）→ 原樣跑 teacher / aligner / student。

- 需要解決：尺度（機器人運動學：已知 gel 視野 17.5 mm 與壓入深度；或 Any6D 式深度對齊）、物件座標系（重建 mesh 的 frame 取代 CAD frame，planar pose 相對它定義）。
- 預期：對特徵大的日常物件可能直接可用；對 ridge 物件 label 品質會明顯下降。
- 價值：是路線 B 的 baseline，也是「幾何精度門檻」實驗的載體。

### 路線 B：觸覺修正幾何的迴圈（核心貢獻候選）

```
粗 mesh（視覺）
  → AlignTac 對齊每個接觸（teacher 初始 + aligner 修正，得到 q_T）
  → 在 q_T 把該接觸的觸覺 normal/depth patch 貼回 mesh 表面
  → 以 displacement field / SDF 微調 mesh（loop closure 由視覺先驗提供，不需 GelSLAM 的連續滑動）
  → 用修正後的 mesh 重新對齊，迭代
```

- 等於「有視覺先驗的 GelSLAM」或「做在幾何而非紋理上的 Tactile DreamFusion」。視覺提供全域形狀、拓樸與閉環，觸覺提供接觸區域 0.1 mm 等級的剖面。
- Aligner 本來就在學「影像 vs 候選 depth 的差異」；mesh 不準時可以多加一個 head 預測局部 depth 殘差，當作修正 mesh 的訊號（或直接用感測器的 normal 積分）。
- 評估：用現有 12 個 CAD 當 GT，量「修正後 mesh 的 Chamfer / 接觸區 depth 誤差」隨迭代下降；再量 pseudo-label 與 student 指標是否回到 CAD 水準。
- 風險：觸覺 patch 的 normal 需要感測器校正（我們現有 GelSlim 的校正可用；免校正版本精度約 10°，可能只夠修 ridge 位置不夠修剖面）；ridge 對稱物件的對齊歧義會把錯的幾何貼進去，需要 confidence 篩選（論文 Sec. V-D 已提到這個需求）。

### 路線 C：完全無幾何的自監督（不建議先做）

只靠多次按壓之間的一致性（同一物件、不同位姿）與機器人位姿做自監督。文獻裡沒有可直接參考的做法，且失去「幾何正確」這個 AlignTac 的核心優勢。

## 6. 建議的實驗順序

1. **幾何退化研究（現在就能做，不需新資料）**：把 CAD mesh 依序退化（高斯平滑 ridge 剖面、頂點噪聲 0.1 / 0.3 / 0.5 / 1 mm、簡化到 1k / 5k 面、整體尺度 ±2%、ridge 高度 ±20%），固定 K=25，量 label IoU / MAE / rot 與 student 指標。結果給出「視覺重建需要多準」的門檻，也直接是論文的一張圖。
2. **視覺重建品質量測**：對 12 個物件各拍一圈（wrist camera 或手機），跑 DA3 與 SAM 3D / TRELLIS，對 CAD 量 Chamfer 與接觸區 depth 誤差；對照步驟 1 的門檻。
3. **路線 A**：把步驟 2 的 mesh 代入現有 pipeline，看 label 與 student 掉多少。
4. **路線 B**：實作觸覺修正迴圈，看能否把 label 品質拉回 CAD 水準；報告迭代次數與每次的 mesh 誤差。
5. **日常物件**：你要收的新物件如果沒有 CAD，先用手機掃描（KIRI / Polycam）當參考幾何，或把掃描 mesh 3D 列印一份做受控實驗；YOTO 的經驗顯示掃描幾何在 mm 等級夠用。

## 7. 對新資料收集的影響

- 每個物件補拍一組多視角 RGB（手機繞拍 30–60 張即可，或 wrist camera 自動繞拍），背景乾淨、光線均勻。
- 若可能，記錄相機內參與機器人位姿，metric 尺度就不用猜。
- 新的日常物件挑特徵大小不同的幾種（大特徵：把手、按鍵；小特徵：螺紋、刻字），讓精度門檻的結論有層次。

## 8. 參考文獻（連結）

視覺 3D 基礎模型：[DA3](https://arxiv.org/abs/2511.10647v1)、[VGGT-Ω](https://arxiv.org/pdf/2605.15195)、[feed-forward 綜述](https://arxiv.org/pdf/2507.08448)、[DTU 比較](https://arxiv.org/pdf/2606.16479)、[MoGe-2](https://arxiv.org/abs/2507.02546)、[SAM 3D](https://arxiv.org/pdf/2511.16624)、[Hunyuan3D 2.1](https://arxiv.org/pdf/2506.15442)、[Hunyuan3D-Buffalo](https://arxiv.org/pdf/2608.02711)、[TRELLISWorld](https://arxiv.org/pdf/2510.23880)、[MM-TRELLIS](https://arxiv.org/pdf/2606.24301)、[實驗室機器人重建評測](https://arxiv.org/abs/2608.26383)、[DREAM real-to-sim](https://arxiv.org/pdf/2608.29078)

重建幾何用於觸覺／位姿：[YOTO](https://arxiv.org/html/2606.28899v1)、[Any6D](https://arxiv.org/html/2503.18673v2)、[FoundationPose](https://nvlabs.github.io/FoundationPose/)、[ViTa-Zero](https://arxiv.org/pdf/2504.13179)、[ViTaSCOPE](https://arxiv.org/pdf/2506.12239)、[TouchScale](https://arxiv.org/abs/2610.10288)、[Shared visuo-tactile perception](https://journals.sagepub.com/doi/10.1177/02783649241301443)

觸覺補幾何：[Tactile DreamFusion](https://arxiv.org/html/2412.06785v1)、[GelSLAM](https://arxiv.org/html/2508.15990v2)、[InvariantCloud](https://arxiv.org/pdf/2605.25216)、[NeuralFeels](https://arxiv.org/pdf/2312.13469)、[Touch-GS](https://arxiv.org/html/2403.09875v3)、[Touch-Augmented GS](https://arxiv.org/html/2508.07717v1)、[TouchSDF](https://arxiv.org/pdf/2311.12602)、[Touch2Shape](https://arxiv.org/abs/2505.13091)、[Proactive tactile exploration](https://arxiv.org/abs/2505.11975)、[VisTa3D](https://arxiv.org/abs/2608.20740)、[TaRF](https://arxiv.org/html/2405.04534v1)、[FingerSLAM](https://arxiv.org/abs/2303.07997)、[Smith et al. 2020](https://proceedings.neurips.cc/paper/2020/file/a3842ed7b3d0fe3ac263bcabd2999790-Paper.pdf)

感測器端：[3D-Cal](https://arxiv.org/abs/2511.03078)、[Calibration-free normals](https://arxiv.org/abs/2609.31754)、[SITR](https://proceedings.iclr.cc/paper_files/paper/2025/file/f05a11e682bf9c57ab62f7cff08fbe5b-Paper-Conference.pdf)、[Sparsh](https://arxiv.org/pdf/2410.24090)、[TacVerse](https://www.researchgate.net/publication/408046993_TacVerse_A_Multi-Sensor_Dataset_and_Benchmark_for_Cross-Sensor_Vision-Based_Tactile_Perception)、[ObjectFolder 2.0 / Taxim](https://export.arxiv.org/pdf/2306.00956)
