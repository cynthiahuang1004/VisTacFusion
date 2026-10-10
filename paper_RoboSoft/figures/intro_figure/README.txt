Figure 1: From labeling bottlenecks to aligned geometric supervision.

The three stacked routes are designed for the introduction's single-column
placement. The upper routes explain the motivation; the larger lower route
retains the approved AlignTac concept and its label-scaling output.

(a) Register real contacts: supplied robot-press photograph, real RGB image,
    and registered mesh depth. One registered pose is needed per labeled press.
(b) Simulate contacts: supplied simulated sensor-contact scene, geometric target,
    and PBR/learned appearance examples. The two RGB renderings are alternatives,
    grouped together, not sequential processing stages.
(c) AlignTac: real image and known geometry, initial/aligned mesh depths,
    training support from a few registered touches, and additional real RGB /
    aligned-depth examples. All same-contact assets match the supplied draft.

Authoritative paper figure:
  AlignTac_Figure1_Updated.pdf
  motivation_figure.tex (editable TikZ)
  motivation_standalone.tex (standalone build wrapper)

Rebuild the PDFs and paper from the project root:
  python build.py --figures

To regenerate the TikZ/JSON layout from the layout authoring script:
  python figures/intro_figure/make_motivation_figure.py

Editable PowerPoint:
  AlignTac_Figure1_Updated.pptx
  Most labels, boxes, and connectors are native editable elements.
  The q-star label is a LaTeX-rendered SVG with PNG fallback; its editable
  LaTeX source is in motivation_math/. No font files are included.
  Regenerate with python-pptx, PyMuPDF and pdfLaTeX installed:
    python figures/intro_figure/export_motivation_pptx.py

Image provenance is in figure_provenance.json. No generated images, numerical
curves, or experimental results were introduced. Source images in
motivation_assets/ are unmodified copies. The original support-contour paths
in assets/ document earlier artwork and are not drawn in this figure.
