"""Editable PowerPoint proposal for the pipeline figure (Fig. 2): three numbered training steps + inference.
usage: <python with python-pptx> paper_RoboSoft/scripts/fig2_pptx.py   (cwd = VisTacFusion; assets from fig2_assets.py)
Every element is a native shape, text box or picture, so it can be moved and re-worded in PowerPoint.
Slide is 16 in wide; at \\textwidth (7.16 in) an 18 pt label prints at about 8 pt."""
import os
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

FIG = "paper_RoboSoft/figures"; A = f"{FIG}/intro_figure/fig2_assets"; PA = f"{FIG}/pipeline_assets"
OUT = f"{FIG}/intro_figure/AlignTac_Fig2_proposal.pptx"
W, H = 16.0, 7.3
INK, GRAY, LIGHT = RGBColor(0x22, 0x22, 0x22), RGBColor(0x6B, 0x6B, 0x6B), RGBColor(0xB5, 0xB5, 0xB5)
RED, GREEN, ORANGE = RGBColor(0xD6, 0x36, 0x2B), RGBColor(0x1E, 0x8E, 0x4E), RGBColor(0xD9, 0x7A, 0x0B)
TEACHER = (RGBColor(0xEE, 0xF1, 0xF5), RGBColor(0x4A, 0x55, 0x68))     # one colour per network, wherever it appears
ALIGNER = (RGBColor(0xEF, 0xE9, 0xFB), RGBColor(0x6B, 0x46, 0xC1))
STUDENT = (RGBColor(0xE3, 0xF0, 0xFB), RGBColor(0x2B, 0x6C, 0xB0))
PANEL = RGBColor(0xF6, 0xF6, 0xF8)
SERIF, MATH = "Times New Roman", "Cambria Math"

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H)
sl = prs.slides.add_slide(prs.slide_layouts[6]); sh = sl.shapes


def text(x, y, w, h, segs, size=16, align="c", anchor="m", color=INK, name=None):
    """segs: str, or list of (text, flags) with flags from: b i sup sub math, or an RGBColor; '\\n' in a list starts a new line."""
    tb = sh.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, m, 0)
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    fill_text(tf, segs, size, align, color)
    if name: tb.name = name
    return tb


def fill_text(tf, segs, size, align, color):
    if isinstance(segs, str): segs = [(segs, "")]
    p = tf.paragraphs[0]; p.alignment = {"c": PP_ALIGN.CENTER, "l": PP_ALIGN.LEFT, "r": PP_ALIGN.RIGHT}[align]
    for s in segs:
        if s == "\n":
            p = tf.add_paragraph(); p.alignment = {"c": PP_ALIGN.CENTER, "l": PP_ALIGN.LEFT, "r": PP_ALIGN.RIGHT}[align]; continue
        t, fl = s if isinstance(s, tuple) else (s, "")
        col = color
        if isinstance(fl, tuple): fl, col = fl
        r = p.add_run(); r.text = t; f = r.font; f.size = Pt(size); f.name = MATH if "math" in fl else SERIF
        f.bold = "b" in fl.split(); f.italic = "i" in fl.split(); f.color.rgb = col
        if "sup" in fl: r._r.get_or_add_rPr().set("baseline", "30000")
        if "sub" in fl: r._r.get_or_add_rPr().set("baseline", "-25000")


def box(x, y, w, h, segs, style, size=18, name=None, dash=False):
    s = sh.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); s.adjustments[0] = 0.12
    s.fill.solid(); s.fill.fore_color.rgb = style[0]; s.line.color.rgb = style[1]; s.line.width = Pt(1.75); s.shadow.inherit = False
    if dash: s.line.dash_style = 4
    tf = s.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, m, Inches(0.03))
    fill_text(tf, segs, size, "c", INK)
    if name: s.name = name
    return s


def panel(x, y, w, h, name, fill=PANEL, line=None, dash=False):
    s = sh.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); s.adjustments[0] = 0.035
    s.shadow.inherit = False; s.name = name
    if fill is None: s.fill.background()
    else: s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None: s.line.fill.background()
    else: s.line.color.rgb = line; s.line.width = Pt(1.25)
    if dash: s.line.dash_style = 4
    return s


def arrow(x0, y0, x1, y1, color=INK, width=1.75, head=True, dash=False, name=None):
    c = sh.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0), Inches(x1), Inches(y1))
    c.line.color.rgb = color; c.line.width = Pt(width)
    if dash: c.line.dash_style = 4
    if head:
        ln = c.line._get_or_add_ln(); e = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}); ln.append(e)
    if name: c.name = name
    return c


def pic(path, x, y, s, name, border=None, stack=0):
    for k in range(stack, 0, -1):                                   # offset cards behind the picture: "many images"
        r = sh.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + 0.07 * k), Inches(y - 0.07 * k), Inches(s), Inches(s))
        r.fill.solid(); r.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF); r.line.color.rgb = LIGHT; r.line.width = Pt(1); r.shadow.inherit = False
        r.name = f"{name} / stack card {k}"
    p = sh.add_picture(path, Inches(x), Inches(y), Inches(s), Inches(s)); p.name = name
    if border is not None: p.line.color.rgb = border; p.line.width = Pt(2.25)
    return p


def pose_glyph(x, y, s, name, color=ORANGE, rot=-28):
    b = sh.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(s), Inches(s)); b.fill.solid(); b.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    b.line.color.rgb = LIGHT; b.line.width = Pt(1); b.shadow.inherit = False; b.name = f"{name} / background"
    ox, oy = x + 0.2 * s, y + 0.8 * s
    arrow(ox, oy, x + 0.88 * s, oy, GRAY, 1.25, name=f"{name} / x axis"); arrow(ox, oy, ox, y + 0.12 * s, GRAY, 1.25, name=f"{name} / y axis")
    r = sh.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + 0.33 * s), Inches(y + 0.36 * s), Inches(0.48 * s), Inches(0.2 * s))
    r.fill.background(); r.line.color.rgb = color; r.line.width = Pt(2); r.rotation = rot; r.shadow.inherit = False; r.name = f"{name} / object frame"


def step(x, y, n, title, sub, w):
    c = sh.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.16), Inches(y + 0.13), Inches(0.44), Inches(0.44)); c.fill.solid(); c.fill.fore_color.rgb = INK
    c.line.fill.background(); c.shadow.inherit = False; c.name = f"Step {n} badge"
    tf = c.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, m, 0)
    fill_text(tf, [(str(n), "b")], 18, "c", RGBColor(0xFF, 0xFF, 0xFF))
    text(x + 0.72, y + 0.1, w - 0.8, 0.5, [(title, "b")] + ([("   " + sub, ("", GRAY))] if sub else []), 20, "l", name=f"Step {n} title")


def fmath(sym, mark, kind="sup"):                                    # f^t, q_0 ... as italic runs with a raised / lowered mark
    return [(sym, "i"), (mark, f"i {kind}")]
A_SYM = ("\U0001D49C", "math")                                       # script A
CAP = 16


# ───────────────────────── (a) Training ─────────────────────────
text(0.2, 0.06, 4, 0.5, [("(a) Training", "b")], 22, "l", name="Panel a heading")
Y0, PH = 0.62, 4.36
P1, P2, P3 = (0.2, 4.2), (4.62, 7.5), (12.34, 3.46)
for i, (px, pw) in enumerate((P1, P2, P3), 1): panel(px, Y0, pw, PH, f"Step {i} panel")
arrow(P1[0] + P1[1] + 0.02, Y0 + 0.36, P2[0] - 0.02, Y0 + 0.36, GRAY, 2.25, name="Step 1 to step 2")
arrow(P2[0] + P2[1] + 0.02, Y0 + 0.36, P3[0] - 0.02, Y0 + 0.36, GRAY, 2.25, name="Step 2 to step 3")

# ① teacher and aligner from the K registered contacts
step(P1[0], Y0, 1, "Train teacher and aligner", "", P1[1])
text(P1[0] + 0.2, Y0 + 0.62, P1[1] - 0.4, 0.3, [("on ", ("", GRAY)), ("K", ("i", GRAY)), (" labeled contacts per object", ("", GRAY))], CAP, "l", name="Step 1 data note")
ts, bw, bh = 1.02, 1.18, 0.8
xa = P1[0] + 0.24; xb = xa + ts + 0.4; xc = xb + bw + 0.4
ya = 1.64                                                             # teacher row
pic(f"{A}/reg_real.png", xa, ya, ts, "Registered real image")
text(xa - 0.2, ya + ts + 0.03, ts + 0.4, 0.3, [("image ", ""), ("\U0001D4D8" if False else "I", "i")], CAP, name="Registered image caption")
arrow(xa + ts + 0.04, ya + ts / 2, xb - 0.04, ya + ts / 2, name="Registered image to teacher")
box(xb, ya + (ts - bh) / 2, bw, bh, [("Teacher", "b"), "\n"] + fmath("f", "t"), TEACHER, name="Teacher (training)")
arrow(xb + bw + 0.04, ya + ts / 2, xc - 0.04, ya + ts / 2, name="Teacher to registered label")
pic(f"{A}/reg_depth.png", xc, ya + 0.1, ts - 0.2, "Registered label (mesh at q*)", border=GREEN)
text(xc - 0.25, ya + ts + 0.03, ts + 0.3, 0.3, [("label at ", ("", GREEN)), ("q", ("i", GREEN)), ("*", ("i sup", GREEN))], CAP, name="Registered label caption")
yb = 3.4                                                             # aligner row
pic(f"{A}/reg_overlay_perturbed.png", xa, yb, ts, "Registered image with mesh outline at q*+delta")
text(xa - 0.2, yb + ts + 0.03, ts + 0.5, 0.3, [("mesh at ", ("", ORANGE)), ("q", ("i", ORANGE)), ("*", ("i sup", ORANGE)), (" + δ", ("i", ORANGE))], CAP, name="Perturbed state caption")
arrow(xa + ts + 0.04, yb + ts / 2, xb - 0.04, yb + ts / 2, name="Perturbed pair to aligner")
box(xb, yb + (ts - bh) / 2, bw, bh, [("Aligner", "b"), "\n", A_SYM], ALIGNER, name="Aligner (training)")
arrow(xb + bw + 0.04, yb + ts / 2, xc - 0.04, yb + ts / 2, name="Aligner to correction target")
t = box(xc, yb + (ts - 0.62) / 2, ts - 0.2, 0.62, [("−δ", "i")], (RGBColor(0xFF, 0xFF, 0xFF), ORANGE), 22, name="Correction target")
text(xc - 0.25, yb + ts + 0.03, ts + 0.3, 0.3, [("correction", ("", GRAY))], CAP, name="Correction target caption")
text(xa, yb - 0.36, P1[1] - 0.4, 0.3, [("same image, mesh moved by a known ", ("i", GRAY)), ("δ", ("i", GRAY)), (":", ("i", GRAY))], 15, "l", name="Aligner training note")

# ② pseudo-label every other real image
step(P2[0], Y0, 2, "Align the mesh to each unlabeled image", "", P2[1])
ts2, bw2 = 1.2, 1.12; g = 0.3
x = P2[0] + 0.2; y2 = 1.95; cy = y2 + ts2 / 2
x_img = x; x_t = x_img + ts2 + g; x_q0 = x_t + bw2 + g; x_a = x_q0 + ts2 + g; x_qt = x_a + bw2 + g
pic(f"{A}/unl_real.png", x_img, y2, ts2, "Unlabeled real image", stack=2)
text(x_img - 0.2, y2 + ts2 + 0.03, ts2 + 0.5, 0.3, "unlabeled image", CAP, name="Unlabeled image caption")
arrow(x_img + ts2 + 0.04, cy, x_t - 0.04, cy, name="Unlabeled image to teacher")
box(x_t, cy - bh / 2, bw2, bh, [("Teacher", "b"), "\n"] + fmath("f", "t"), TEACHER, name="Teacher (labeling)")
arrow(x_t + bw2 + 0.04, cy, x_q0 - 0.04, cy, name="Teacher to initial state")
pic(f"{A}/unl_overlay_teacher.png", x_q0, y2, ts2, "Mesh outline at the teacher state q0")
text(x_q0 - 0.2, y2 + ts2 + 0.03, ts2 + 0.4, 0.3, [("initial state ", ("", RED))] + [(a, (b, RED)) for a, b in fmath("q", "0", "sub")], CAP, name="Initial state caption")
arrow(x_q0 + ts2 + 0.04, cy, x_a - 0.04, cy, name="Initial state to aligner")
box(x_a, cy - bh / 2, bw2, bh, [("Aligner", "b"), "\n", A_SYM], ALIGNER, name="Aligner (labeling)")
text(x_a - 0.2, cy + bh / 2 + 0.04, bw2 + 0.4, 0.3, [("↻ 6 updates", ("", ALIGNER[1]))], CAP, name="Iteration note")
pic(f"{PA}/asset_02.png", x_a + bw2 / 2 - 0.3, 1.24, 0.6, "Object mesh")
text(x_a + bw2 / 2 + 0.36, 1.34, 1.2, 0.4, [("mesh ", ("", GRAY)), ("ℳ", ("math", GRAY))], CAP, "l", name="Mesh caption")
arrow(x_a + bw2 / 2, 1.86, x_a + bw2 / 2, cy - bh / 2 - 0.03, GRAY, 1.5, name="Mesh to aligner")
arrow(x_a + bw2 + 0.04, cy, x_qt - 0.04, cy, name="Aligner to aligned state")
pic(f"{A}/unl_overlay_aligned.png", x_qt, y2, ts2, "Mesh outline at the aligned state qT")
text(x_qt - 0.25, y2 + ts2 + 0.03, ts2 + 0.4, 0.3, [("aligned state ", ("", GREEN))] + [(a, (b, GREEN)) for a, b in fmath("q", "T", "sub")], CAP, name="Aligned state caption")
# pseudo-label row: the mesh projected at qT
tl = 0.74; yl = 3.86; xl = x_qt + ts2 - (3 * tl + 0.2)
arrow(x_qt + ts2 / 2, y2 + ts2 + 0.36, x_qt + ts2 / 2, yl - 0.04, name="Aligned state to pseudo-label")
pic(f"{A}/unl_depth.png", xl, yl, tl, "Pseudo-label depth", border=GREEN)
pic(f"{A}/unl_normal.png", xl + tl + 0.1, yl, tl, "Pseudo-label normals", border=GREEN)
pose_glyph(xl + 2 * (tl + 0.1), yl, tl, "Pseudo-label pose", GREEN)
for k, nm in enumerate(("depth", "normals", "pose")):
    text(xl + k * (tl + 0.1) - 0.1, yl + tl + 0.01, tl + 0.2, 0.28, nm, 15, name=f"Pseudo-label {nm} caption")
text(xl - 3.3, yl + 0.02, 3.15, 0.36, [("Pseudo-label", "b")], 18, "r", name="Pseudo-label heading")
text(xl - 3.6, yl + 0.38, 3.45, 0.3, [("= mesh projected at ", ("", GRAY))] + [(a, (b, GRAY)) for a, b in fmath("q", "T", "sub")], CAP, "r", name="Pseudo-label definition")
text(P2[0] + 0.2, Y0 + PH - 0.4, 3.6, 0.3, [("outlines: mesh contact region at the state", ("i", GRAY))], 14, "l", name="Outline legend")

# ③ student on all real images
step(P3[0], Y0, 3, "Train student", "", P3[1])
text(P3[0] + 0.2, Y0 + 0.62, P3[1] - 0.3, 0.3, [("on all real images", ("", GRAY))], CAP, "l", name="Step 3 data note")
tg = 0.74; xg = P3[0] + 0.22; yg1 = 2.12; yg2 = yl
text(xg - 0.05, yg1 - 0.4, 2 * tg + 0.3, 0.3, [("K", "i b"), (" labeled", "b")], CAP, "l", name="Registered group caption")
pic(f"{A}/reg_real.png", xg, yg1, tg, "Registered image (student data)")
pic(f"{A}/reg_depth.png", xg + tg + 0.08, yg1, tg, "Registered label (student data)", border=GREEN)
text(xg - 0.05, yg2 - 0.44, 2 * tg + 0.6, 0.3, [("pseudo-labeled", "b")], CAP, "l", name="Pseudo-labeled group caption")
pic(f"{A}/unl_real.png", xg, yg2, tg, "Pseudo-labeled image (student data)", stack=2)
pic(f"{A}/unl_depth.png", xg + tg + 0.08, yg2, tg, "Pseudo-label (student data)", border=GREEN)
arrow(xl + 3 * tl + 0.24, yl + tl / 2, xg - 0.04, yl + tl / 2, name="Pseudo-labels to student data")
sw, sx = 1.12, P3[0] + P3[1] - 1.12 - 0.16; sy = (yg1 + tg / 2 + yg2 + tg / 2) / 2 - bh / 2
box(sx, sy, sw, bh, [("Student", "b"), "\n"] + fmath("f", "s"), STUDENT, name="Student (training)")
xr = xg + 2 * tg + 0.12
arrow(xr, yg1 + tg / 2, sx + sw / 2, yg1 + tg / 2, head=False, name="Registered data to student / 1")
arrow(sx + sw / 2, yg1 + tg / 2, sx + sw / 2, sy - 0.03, name="Registered data to student / 2")
arrow(xr, yg2 + tg / 2, sx + sw / 2, yg2 + tg / 2, head=False, name="Pseudo-labeled data to student / 1")
arrow(sx + sw / 2, yg2 + tg / 2, sx + sw / 2, sy + bh + 0.03, name="Pseudo-labeled data to student / 2")

# ───────────────────────── (b) Inference ─────────────────────────
YB = Y0 + PH + 0.16
text(0.2, YB, 4, 0.5, [("(b) Inference", "b")], 22, "l", name="Panel b heading")
ti = 0.95; yi = YB + 0.62; ci = yi + ti / 2; xi = 0.42
pic(f"{PA}/asset_01.png", xi, yi, ti, "Inference real image")
text(xi, yi + ti + 0.02, ti + 1.5, 0.3, [("image ", ""), ("I", "i"), (" + object ID ", ""), ("o", "i")], CAP, "l", name="Inference input caption")
xs = xi + ti + 0.75
arrow(xi + ti + 0.04, ci, xs - 0.04, ci, name="Inference input to student")
box(xs, ci - bh / 2, 1.25, bh, [("Student", "b"), "\n"] + fmath("f", "s"), STUDENT, name="Student (inference)")
xo = xs + 1.25 + 0.5
arrow(xs + 1.29, ci, xo - 0.04, ci, name="Student to predictions")
pic(f"{PA}/asset_03.png", xo, yi, ti, "Predicted depth"); pic(f"{PA}/asset_04.png", xo + ti + 0.12, yi, ti, "Predicted normals")
pose_glyph(xo + 2 * (ti + 0.12), yi, ti, "Predicted pose")
for k, nm in enumerate(("depth", "normals", "pose")):
    text(xo + k * (ti + 0.12), yi + ti + 0.02, ti, 0.3, nm, CAP, name=f"Predicted {nm} caption")
xopt = xo + 3 * ti + 0.24 + 0.75; wopt = W - 0.2 - xopt
panel(xopt, YB + 0.12, wopt, H - YB - 0.24, "Optional refinement boundary", fill=None, line=LIGHT, dash=True)
text(xopt + 0.2, YB + 0.18, wopt - 0.4, 0.34, [("Optional: pose refinement when the mesh is available", ("i", GRAY))], CAP, "l", name="Optional refinement heading")
xal = xopt + 2.3
arrow(xo + 3 * ti + 0.28, ci, xal - 0.04, ci, name="Predicted pose to aligner")
text(xopt + 0.25, ci - 0.4, 1.9, 0.3, [("pose as ", ("", GRAY))] + [(a, (b, GRAY)) for a, b in fmath("q", "0", "sub")], CAP, "l", name="Refinement initial state")
text(xopt + 0.25, ci + 0.1, 2.0, 0.3, [("+ image ", ("", GRAY)), ("I", ("i", GRAY)), (", mesh ", ("", GRAY)), ("\u2133", ("math", GRAY))], CAP, "l", name="Refinement extra inputs")
box(xal, ci - bh / 2, 1.25, bh, [("Aligner", "b"), "\n", A_SYM], ALIGNER, name="Aligner (inference)")
xrp = xal + 1.25 + 0.5
arrow(xal + 1.29, ci, xrp - 0.04, ci, name="Aligner to refined pose")
pose_glyph(xrp, yi, ti, "Refined pose", GREEN, rot=-20)
text(xrp + ti + 0.12, ci - 0.2, 1.4, 0.4, "refined pose", CAP, "l", name="Refined pose caption")

for s_ in sh:
    st = s_._element.find(qn("p:style"))
    if st is not None: s_._element.remove(st)
prs.save(OUT); print("saved", OUT, "| shapes:", len(sh))
