# Architecture figure design study

2026-09-09. Compared the actual architecture figures in six primary-source PDFs. Figure numbers and page numbers below refer to the downloaded versions; pages are one-based PDF pages. Design recommendations are editorial judgments, not claims made by the cited authors.

## Reference figures

| Paper and primary source | Figure / PDF page | Observed visual approach | Application to VisTacFusion |
| --- | --- | --- | --- |
| [TVL — A Touch, Vision, and Language Dataset for Multimodal Alignment](https://tactile-vlm.github.io/files/tvl.pdf#page=5) | Fig. 4, right / p. 5 | Input thumbnails, encoder shapes, colored latent strips, and frozen/trainable icons distinguish data, computation, and training status. This is the user's example. | Represent tokens graphically; identify frozen encoders consistently. Its latent averaging and LLM pathway do not describe our fusion. |
| [Sparsh-X — Tactile Beyond Pixels](https://akashsharma02.github.io/sparsh-x-ssl/sparshx.pdf#page=4) | Fig. 2 / p. 4 | Aligned modality columns retain their colors through tokenization and unimodal layers; bottleneck tokens are a separate group beside a shared fusion stage. | Give visual memory, tactile queries, and bottleneck states distinct visual identities. Do not imply Sparsh-X's multisensory topology is our one-way visual conditioning. |
| [Attention Bottlenecks for Multimodal Fusion](https://papers.nips.cc/paper/2021/file/76ba9f564ebbc35b1014ac498fafadd0-Paper.pdf#page=2) | Fig. 1 / p. 2; Fig. 2 / p. 3 | Token nodes and connectivity distinguish fusion strategies; the application overview is separate. | Use a small module detail to show which stream provides queries versus keys/values, and how bottlenecks carry across layers. |
| [SITR — Sensor-Invariant Tactile Representation](https://arxiv.org/pdf/2502.19638#page=3) | Fig. 2 / p. 3 | Patch grids and distinct tokens expose calibration/input construction; separate outgoing paths show class-token and normal-map supervision. | Separate pose-token and spatial-feature roles. Calibration belongs in the encoder description/caption when omitted from the overview. Do not import SITR pretraining losses into our downstream model. |
| [Transferable Tactile Transformers (T3)](https://arxiv.org/pdf/2406.13640#page=2) | Fig. 1 / p. 2 | Sensor-specific encoders, a shared trunk, and task-specific decoders occupy distinct stages; dashed regions identify tasks. | Establish a consistent left-to-right encoder / processing / output hierarchy. Our dense branch bypasses the pose queries, so it cannot simply be drawn as another head on the final trunk output. |
| [DPT — Vision Transformers for Dense Prediction](https://openaccess.thecvf.com/content/ICCV2021/papers/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.pdf#page=3) | Fig. 1 / p. 3 | Overview, token reassembly, and fusion details appear in separate panels; intermediate transformer outputs feed a decoder hierarchy. | Separate the overall model from fusion/injection details. Show intermediate encoder taps explicitly, with spatial resizing confined to DPT. |

## Recommended layout

Use three panels: **(a) complete paired-input pathway**, **(b) one asymmetric fusion layer**, and **(c) one gated dense-feature injection**. The overview should establish the two encoder streams, direct tactile-to-dense branch, compact fusion trunk, and two output tasks before introducing attention details.

The current figure's large formula-filled trunk gives equations more visual weight than the information flow. Its many similarly shaped boxes also make tensors and operations look interchangeable. Use token strips for sequences, equal-sized patch grids for the four intermediate tactile taps, boxes for operations, and small thumbnails for inputs/output types. Keep long tensor shapes and full equations in the method section.

Use orange for tactile-derived features, blue for visual features, and purple for bottleneck states/conditioning. Use restrained outlines and pale fills, consistent sans-serif labels, and a white background. Frozen labels identify the two backbones; learned projections remain visibly outside those backbones. Avoid using color alone to identify a pathway: retain names and directed arrows.

## Draft artifacts

- Editable source: [architecture_redesign.tex](architecture_redesign.tex)
- Vector text and geometry, with raster example thumbnails: [architecture_redesign.pdf](figures/architecture_redesign.pdf)
- Preview: [architecture_redesign.png](figures/architecture_redesign.png)

This is an original design study. It is not included by `main.tex`. Reference-paper artwork is not embedded in the draft. The overview uses the repository's existing simulation inputs and ground-truth geometry thumbnails to illustrate the output types; those thumbnails are **not model predictions**. The draft is compiled independently of the submission PDF.

Build from `paper/`:

```sh
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=figures architecture_redesign.tex
pdftoppm -scale-to 2200 -singlefile -png figures/architecture_redesign.pdf figures/architecture_redesign
```

## Implementation constraints retained in the draft

- `BranchProjection` is a learned linear map plus modality embedding. The tactile 768-to-768 map is not an identity; the visual map is 1024-to-768.
- `Q_0` contains 196 spatial queries and one pose query initialized from the tactile CLS token. The dark orange token depicts that pose query. Position and object-identity conditioning are omitted graphically, not removed from the method.
- Visual memory `M` contains 197 tokens and remains fixed across the four asymmetric fusion layers. Learned `B_0` initializes 16 bottleneck tokens; `B_l` thereafter denotes an input-dependent state.
- Within a layer, bottlenecks attend to visual memory without an FFN; queries attend to the updated bottlenecks with an FFN, then undergo self-attention with an FFN. Attention and FFN sublayers use pre-normalization and residual connections.
- `T_i` comes directly from tactile encoder blocks 3, 6, 9, and 12, with 196 tokens of width 768 per tap. The four grids have the same spatial resolution. They are different encoder depths; DPT subsequently reassembles/resizes them into a feature hierarchy.
- Each tap has independent attention parameters and a zero-initialized scalar gate. All four use final visual bottleneck state `B_4` as context. The injection does not use the final spatial queries `Q_4`.
- Pose input is a concatenation, not a sum: final pose token plus mean-pooled spatial queries gives 1536 features. The four output values encode planar rotation and translation.
- Only the paired-input route is drawn. Tactile-only inference skips visual cross-attention/injection; visual-only inference uses learned queries and adapted MAE taps. Keep these routes explicit in the text rather than adding many conditional edges to the overview.

Checked against `vistacfusion/models/projection.py`, `fusion.py`, and `model.py`. The draft describes the main asymmetric configuration, not optional symmetric/CORAL variants now available in the repository.

## Proposed caption if the draft is adopted

> **VisTacFusion architecture.** (a) Frozen tactile and visual encoders provide features for pose estimation and dense geometry prediction. Learned projections form tactile-initialized queries and fixed visual memory. Four intermediate tactile features feed the dense decoder through gated visual conditioning. (b) Each fusion layer first updates the bottleneck from visual memory, then updates the queries through cross-attention and self-attention. Attention and FFN sublayers include pre-normalization and residual connections; the bottleneck is initialized by learned tokens and carried across four layers. (c) Each dense-feature tap has its own cross-attention module and zero-initialized gate, conditioned on the final bottleneck. The pose head concatenates the final pose token with mean-pooled spatial queries. The paired-input configuration is shown; calibration inputs, embedding additions, and missing-modality routes are described in the text. Token glyphs are schematic, and geometry thumbnails illustrate output types using ground-truth examples.

Before publication, actual predictions from the reported checkpoint could replace the illustrative output thumbnails, with their provenance recorded. A final full-paper compile and page-layout inspection are needed when the figure is adopted, because its height differs from the currently included PNG.
