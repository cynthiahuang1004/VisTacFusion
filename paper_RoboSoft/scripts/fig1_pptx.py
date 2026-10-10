"""Editable PowerPoint of the revised Fig. 1 mock (same layout as figures/intro_figure/mock_fig1_v2.tex).
Every image, label, box and arrow is a separate native object.  usage (cwd = paper_RoboSoft, python with python-pptx):
    python scripts/fig1_pptx.py   ->  figures/intro_figure/AlignTac_Figure1_v2.pptx"""
import math
from pptx import Presentation
from pptx.util import Cm, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.xmlchemy import OxmlElement

SC = 3.0                      # 1 cm of the printed figure -> 3 cm on the slide
TOP = 9.15                    # top of the figure in the TikZ coordinates
T, V, A = "figures/teaser", "figures/intro_figure/v2_assets", "figures/intro_figure/assets/images"
C = dict(afill="FDEEE6", aedge="D2691E", bfill="EEF4FA", bedge="3A6EA5", cfill="F3EFFA", cedge="7A52C7",
         bad="D8382E", good="1E9E55", ink="2B2B2B", gray="666666")
FONT = "Times New Roman"; TINY, TITLE = 15, 21      # pt on the slide (5 pt and 7 pt in print)
prs = Presentation(); prs.slide_width = Cm(8.6 * SC); prs.slide_height = Cm((TOP - 0.55) * SC)
sl = prs.slides.add_slide(prs.slide_layouts[6])
X = lambda x: Cm(x * SC); Y = lambda y: Cm((TOP - y) * SC); L = lambda d: Cm(d * SC)
rgb = lambda k: RGBColor.from_string(C.get(k, k))

def no_shadow(sh):
    sp = sh._element.spPr
    for e in list(sp):
        if e.tag.endswith("effectLst"): sp.remove(e)
    sp.append(OxmlElement("a:effectLst"))
def panel(x0, y0, x1, y1, fill, edge, w=1.6):
    sh = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, X(x0), Y(y1), L(x1 - x0), L(y1 - y0)); sh.adjustments[0] = 0.04
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill); sh.line.color.rgb = rgb(edge); sh.line.width = Pt(w); no_shadow(sh); return sh
def img(path, cx, cy, w, frame=None):
    p = sl.shapes.add_picture(path, 0, 0, width=L(w)); p.left = int(X(cx) - p.width / 2); p.top = int(Y(cy) - p.height / 2)
    if frame: p.line.color.rgb = rgb(frame); p.line.width = Pt(3)
    return p
def text(s, cx, cy, w, color="ink", size=TINY, bold=False, italic=False, align="c", h=0.3):
    left = {"c": cx - w / 2, "l": cx, "r": cx - w}[align]
    tb = sl.shapes.add_textbox(X(left), Y(cy + h / 2), L(w), L(h)); tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"c": PP_ALIGN.CENTER, "l": PP_ALIGN.LEFT, "r": PP_ALIGN.RIGHT}[align]
        r = p.add_run(); r.text = line; f = r.font; f.name = FONT; f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = rgb(color)
    return tb
def head(ln, kind="triangle"):
    e = OxmlElement("a:tailEnd"); e.set("type", kind); e.set("w", "med"); e.set("len", "med"); ln.append(e)
def arrow(x1, y1, x2, y2, color="gray", w=1.6, dash=False):
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, X(x1), Y(y1), X(x2), Y(y2)); c.line.color.rgb = rgb(color); c.line.width = Pt(w)
    if dash: c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    head(c.line._get_or_add_ln()); return c
def line(x1, y1, x2, y2, color="BBBBBB", w=1.0):
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, X(x1), Y(y1), X(x2), Y(y2)); c.line.color.rgb = rgb(color); c.line.width = Pt(w); return c
def arc(x1, y1, x2, y2, rise, color, w=1.8, n=40):
    pts = [(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t + rise * math.sin(math.pi * t)) for t in (i / n for i in range(n + 1))]
    fb = sl.shapes.build_freeform(int(X(pts[0][0])), int(Y(pts[0][1])), scale=1.0)
    fb.add_line_segments([(int(X(x)), int(Y(y))) for x, y in pts[1:]], close=False)
    sh = fb.convert_to_shape(); sh.fill.background(); sh.line.color.rgb = rgb(color); sh.line.width = Pt(w); no_shadow(sh)
    head(sh.line._get_or_add_ln()); return sh

# ---------------- (a) register real contacts ----------------
panel(0, 6.55, 8.6, 9.15, "afill", "aedge")
text("(a) Register real contacts", 0.15, 8.93, 4.2, "aedge", TITLE, bold=True, align="l")
text("Labels need calibration and verification", 8.45, 8.93, 4.3, "aedge", TINY, italic=True, align="r")
th = 0.55
img(f"{T}/real_contact_crop.png", 0.78, 7.62, 1.1); img(f"{T}/real.png", 2.55, 7.62, 1.1); img(f"{T}/gt.png", 4.05, 7.62, 1.1)
arrow(0.78 + th, 7.62, 2.55 - th, 7.62)
arc(0.9, 7.62 + th + 0.03, 4.0, 7.62 + th + 0.03, 0.22, "aedge"); text("q* from the robot pose", 2.45, 8.55, 3.0, "aedge")
text("Registered press", 0.78, 6.9, 1.6); text("Real image", 2.55, 6.9, 1.4); text("GT label", 4.05, 6.9, 1.4, "aedge", bold=True)
line(4.95, 6.72, 4.95, 8.6)
img(f"{V}/a_overlay_rawpose.png", 5.95, 7.62, 1.1); img(f"{V}/a_overlay_verified.png", 7.55, 7.62, 1.1); arrow(5.95 + th, 7.62, 7.55 - th, 7.62)
text("after re-placing the object", 6.75, 8.36, 3.2)
text("old calibration", 5.95, 6.9, 1.5, "bad"); text("re-calibrated, verified", 7.55, 6.9, 1.9, "good")
# ---------------- (b) simulate contacts ----------------
panel(0, 4.2, 8.6, 6.4, "bfill", "bedge")
text("(b) Simulate contacts", 0.15, 6.17, 4.2, "bedge", TITLE, bold=True, align="l")
text("Synthetic RGB needs sensor-response modeling", 8.45, 6.17, 4.6, "bedge", TINY, italic=True, align="r")
for p, x in ((f"{T}/sim_contact.png", 0.78), (f"{T}/gt.png", 2.55), (f"{T}/pbr.png", 4.55), (f"{T}/gan.png", 5.95), (f"{T}/real.png", 7.75)):
    img(p, x, 5.3, 1.1)
arrow(0.78 + th, 5.3, 2.55 - th, 5.3); arrow(2.55 + th, 5.3, 4.55 - th, 5.3); text("render RGB", 3.55, 5.46, 1.0)
text("≠", 6.85, 5.3, 0.5, "gray", 26)
text("Simulated contact", 0.78, 4.6, 1.6); text("GT label (free)", 2.55, 4.6, 1.5, "bedge", bold=True); text("PBR", 4.55, 4.6, 1.0)
text("Learned\n(fitted on real pairs)", 5.95, 4.52, 1.9, h=0.42); text("Real image", 7.75, 4.6, 1.3)
# ---------------- (c) AlignTac ----------------
panel(0, 0.55, 8.6, 4.05, "cfill", "cedge", 2.2)
text("(c) AlignTac: real appearance + aligned geometry", 0.15, 3.83, 7.0, "cedge", TITLE, bold=True, align="l")
t2 = 0.65
img(f"{V}/c_overlay_teacher.png", 0.95, 2.6, 1.3); img(f"{V}/c_overlay_aligned.png", 5.6, 2.6, 1.3); img(f"{V}/c_aligned_depth.png", 7.6, 2.6, 1.3, frame="cedge")
box = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, X(2.3), Y(2.98), L(1.5), L(0.76)); box.adjustments[0] = 0.12
box.fill.solid(); box.fill.fore_color.rgb = RGBColor(255, 255, 255); box.line.color.rgb = rgb("cedge"); box.line.width = Pt(2); no_shadow(box)
tf = box.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
for i, s in enumerate(("Learned", "aligner 𝒜")):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = s; r.font.name = FONT; r.font.size = Pt(TITLE); r.font.bold = True; r.font.color.rgb = rgb("cedge")
arrow(0.95 + t2, 2.6, 2.3, 2.6); arrow(3.8, 2.6, 5.6 - t2, 2.6); arrow(5.6 + t2, 2.6, 7.6 - t2 - 0.03, 2.6)
text("q ← q + Δq", 4.375, 2.8, 1.1, "cedge", italic=True)
text("unlabeled real image + known mesh", 0.3, 3.4, 3.6, align="l")
text("mesh at teacher estimate", 0.95, 1.82, 2.2, "bad"); text("mesh after alignment", 5.6, 1.82, 2.0, "good"); text("Pseudo-label", 7.6, 1.8, 1.6, "cedge", bold=True)
img(f"{A}/registered_pair_rgb.png", 2.75, 1.37, 0.5); img(f"{A}/registered_pair_geometry.png", 3.35, 1.37, 0.5)
arrow(3.05, 1.66, 3.05, 2.2, "cedge", 1.6, dash=True); text("train", 3.38, 1.93, 0.6, "cedge")
text("K registered contacts", 3.05, 0.98, 2.2, "cedge", bold=True)
for i, x in (("01", 5.0), ("02", 6.35), ("03", 7.7)):
    img(f"{A}/contact_{i}_rgb.png", x - 0.29, 1.3, 0.52); img(f"{A}/contact_{i}_aligned_depth.png", x + 0.29, 1.3, 0.52)
arrow(7.6, 1.7, 7.6, 1.6, "cedge")
text("pseudo-labels for the remaining real images", 6.35, 0.88, 4.2, "cedge")
tag = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, X(0.15), Y(1.58), L(2.05), L(0.76)); tag.adjustments[0] = 0.12
tag.fill.solid(); tag.fill.fore_color.rgb = RGBColor(255, 255, 255); tag.line.color.rgb = RGBColor(160, 160, 160); tag.line.width = Pt(1); no_shadow(tag)
tf = tag.text_frame; tf.margin_left = Cm(0.2); tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
for i, s in enumerate(("K=25 with AlignTac ≈", "real-only K=50", "(contact metrics)")):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = s; r.font.name = FONT; r.font.size = Pt(TINY); r.font.color.rgb = rgb("ink")
out = "figures/intro_figure/AlignTac_Figure1_v2.pptx"; prs.save(out); print("saved", out, "shapes:", len(sl.shapes))
