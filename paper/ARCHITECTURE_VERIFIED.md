# VisTacFusion architecture figure

2026-09-11. Original TikZ diagram, checked against the paired-input asymmetric configuration in `ablation/encoder/tac_sitr_rgb_mae.yaml` and the current model implementation.

## Files

- [Vector PDF](figures/architecture_verified.pdf): publication asset, with embedded Type 1 fonts. Only the input/output example thumbnails are raster images.
- [PNG preview](figures/architecture_verified.png).
- [Editable TikZ source](architecture_verified.tex).
- [Caption](figures/architecture_verified_caption.tex).
- [IEEE-width figure and caption proof](figures/architecture_verified_preview.pdf).

Build from `paper/` with `make architecture-verified architecture-proof`. The manuscript's current figure is not replaced by these independent targets.

## Preserved architectural details

- SITR-B/16 is frozen, with 12 sequential transformer blocks. The drawing groups blocks into four sets of three; each tap is taken after blocks 3, 6, 9, and 12. The four tap grids all contain 196 tokens of width 768.
- The final tactile encoder output is projected 768→768 to initialize 196 spatial queries and one pose query. The pose query comes from the tactile CLS token. MAE-L/16 has 24 blocks; its 1024-dimensional features are projected to 768-dimensional, read-only visual memory.
- Each of four fusion layers updates the bottleneck from visual memory, then updates the queries from the updated bottleneck. The query cross-attention and self-attention each have an FFN. The bottleneck cross-attention has no FFN. All attention and FFN sublayers use pre-normalization and residual connections.
- Bottleneck and query states carry between layers. Learned B0 initializes the first layer; the two final states are B4 and Q4. B4 supplies dense conditioning; Q4 alone supplies the pose head. The bridge in the outgoing query line explicitly avoids a junction with the B4 conditioning line.
- Each dense tap has its own cross-attention and zero-initialized scalar gate. The displayed equation includes its residual addition. The four injection modules share B4 as context, not their parameters.
- Four independent DPT reassembly operations produce 256-channel maps of sizes 56², 28², 14², and 7². Four fusion operations and their 2× upsampling stages yield a shared 112² feature map. Only the final fusion output feeds the heads. Each output head upsamples once more to 224².
- The pose head concatenates the final pose token with mean-pooled spatial queries: 768 + 768 = 1536. The regression head outputs four values for planar rotation and translation; the sine/cosine pair is normalized in code.

## Validation

Used the actual `FusionTrunk`, `DPTHead`, and `PoseHead` implementations with PyTorch meta tensors in the `vistacfusion` environment. This checks tensor shapes without loading encoder checkpoints or producing numerical predictions. For a batch size of 2:

| Tensor | Shape |
| --- | --- |
| Final spatial queries | (2, 196, 768) |
| Final pose token | (2, 1, 768) |
| Final bottleneck | (2, 16, 768) |
| Reassembled maps | (2, 256, 56, 56), (2, 256, 28, 28), (2, 256, 14, 14), (2, 256, 7, 7) |
| Shared dense feature | (2, 256, 112, 112) |
| Depth and normal outputs | (2, 1, 224, 224), (2, 3, 224, 224) |
| Pose pooled input and output | (2, 1536), (2, 4) |

Both standalone figure and IEEE-style proof compile without overfull/underfull boxes or LaTeX warnings. Inspected the rendered figure and the full-width proof. `pdffonts` confirms embedded Type 1 fonts.

## Scope and image provenance

The diagram shows paired inputs. Calibration and modality/position/object-ID embeddings are omitted graphically and disclosed in the caption. Missing-modality routes remain in the method text. Optional CORAL and symmetric fusion variants are not represented.

Input thumbnails use `assets/tactile_sim.png` and `assets/rgb_sim.png`. Geometry thumbnails use `assets/depth_gt.png` and `assets/normal_gt.png`: they illustrate output types and are **ground truth, not model predictions**. The caption states this explicitly. Actual checkpoint predictions can replace them when their provenance is available.
