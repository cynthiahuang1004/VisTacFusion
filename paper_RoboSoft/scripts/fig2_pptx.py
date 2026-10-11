"""Editable PowerPoint proposal for the pipeline figure (Fig. 2): three numbered training steps + inference (2x2 layout).
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
W, H = 16.0, 7.4
INK, GRAY, LIGHT = RGBColor(0x22, 0x22, 0x22), RGBColor(0x6B, 0x6B, 0x6B), RGBColor(0xB5, 0xB5, 0xB5)
RED, GREEN, ORANGE = RGBColor(0xD6, 0x36, 0x2B), RGBColor(0x1E, 0x8E, 0x4E), RGBColor(0xD9, 0x7A, 0x0B)
TEACHER = (RGBColor(0xEE, 0xF1, 0xF5), RGBColor(0x4A, 0x55, 0x68))     # one colour per network, wherever it appears
ALIGNER = (RGBColor(0xEF, 0xE9, 0xFB), RGBColor(0x6B, 0x46, 0xC1))
STUDENT = (RGBColor(0xE3, 0xF0, 0xFB), RGBColor(0x2B, 0x6C, 0xB0))
PANEL = RGBColor(0xF6, 0xF6, 0xF8)
SERIF, MATH = "Times New Roman", "Cambria Math"

HEX = lambda c: "%02X%02X%02X" % (c[0], c[1], c[2])
COL = lambda c, t: r"\textcolor[HTML]{%s}{%s}" % (HEX(c), t)
PURPLE_ = RGBColor(0x6B, 0x46, 0xC1)
LABELS = {                                                             # key: LaTeX (macros from notation.tex)
    "ft": r"$\Teacher$", "fs": r"$\Student$", "A": r"$\Aligner$",
    "dq": r"$\Delta q$", "mdelta": r"$-\delta$",
    "gt_state": COL(GREEN, r"GT state $q^\ast$"),
    "k_labeled": r"$K$ \textbf{labeled}",
    "perturbed": r"image + " + COL(ORANGE, r"mesh at $q^\ast+\delta$"),
    "q0": COL(RED, r"initial state $q_0$"), "qT": COL(GREEN, r"aligned state $q_T$"),
    "mesh": COL(GRAY, r"mesh $\Mesh$"),
    "pseudo_def": COL(PURPLE_, r"\textbf{Pseudo-label}") + r" = mesh $\Mesh$ projected at $q_T$",
    "optional": COL(GRAY, r"\textit{optional, with mesh} $\Mesh$"),
}
DPI = 1200


def build_labels():
    """Typeset LABELS with pdflatex (10 pt, one cropped page each) and rasterise to transparent PNGs."""
    import subprocess, tempfile
    from PIL import Image
    out = os.path.abspath(f"{A}/math"); os.makedirs(out, exist_ok=True); note = os.path.abspath("paper_RoboSoft/notation.tex")
    body = "\n".join(r"\begin{lbl}\strut %s\end{lbl}" % t for t in LABELS.values())
    tex = (r"\documentclass[multi=lbl,border=0.6pt]{standalone}" "\n" r"\usepackage{fix-cm}\usepackage{amsmath,amssymb,xcolor}" "\n"
           r"\renewcommand{\rmdefault}{ptm}" "\n" r"\input{%s}" "\n" r"\newenvironment{lbl}{}{}" "\n" r"\begin{document}" "\n%s\n" r"\end{document}" "\n") % (note, body)
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/labels.tex", "w").write(tex)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "labels.tex"], cwd=d, capture_output=True, text=True)
        if r.returncode: raise SystemExit(r.stdout[-2000:])
        subprocess.run(["pdftocairo", "-png", "-transp", "-r", str(DPI), "labels.pdf", "pg"], cwd=d, check=True)
        pages = sorted(f for f in os.listdir(d) if f.startswith("pg-")); assert len(pages) == len(LABELS), pages
        res = {}
        for key, f in zip(LABELS, pages):
            im = Image.open(f"{d}/{f}"); im.save(f"{out}/{key}.png"); res[key] = (f"{out}/{key}.png", im.size[0] / DPI, im.size[1] / DPI)
    return res


prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H)
sl = prs.slides.add_slide(prs.slide_layouts[6]); sh = sl.shapes
LAB = build_labels()


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


def mlabel(key, x, y, pt=16, align="c", name=None):
    """Place a LaTeX label; (x, y) is its centre (align c), or its left / right edge at mid-height (l / r)."""
    path, w10, h10 = LAB[key]; w, h = w10 * pt / 10, h10 * pt / 10
    left = x - w / 2 if align == "c" else x if align == "l" else x - w
    p = sh.add_picture(path, Inches(left), Inches(y - h / 2 + 0.02), Inches(w), Inches(h)); p.name = name or f"{key} [LaTeX]"   # +0.02: match text-box baselines
    return p


MODEL = lambda word, key: ("MODEL", word, key)                         # box label: editable bold word + LaTeX symbol underneath


def box(x, y, w, h, segs, style, size=18, name=None, dash=False):
    model = segs if isinstance(segs, tuple) and segs[0] == "MODEL" else None
    if model: segs = [(model[1], "b"), "\n", (" ", "")]
    s = sh.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); s.adjustments[0] = 0.12
    s.fill.solid(); s.fill.fore_color.rgb = style[0]; s.line.color.rgb = style[1]; s.line.width = Pt(1.75); s.shadow.inherit = False
    if dash: s.line.dash_style = 4
    tf = s.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, m, Inches(0.03))
    fill_text(tf, segs, size, "c", INK)
    if name: s.name = name
    if model: mlabel(model[2], x + w / 2, y + h / 2 + 0.16, size, name=f"{name} / symbol [LaTeX]")
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


def arrow(x0, y0, x1, y1, color=INK, width=1.75, head=True, dash=False, name=None, both=False):
    c = sh.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0), Inches(x1), Inches(y1))
    c.line.color.rgb = color; c.line.width = Pt(width)
    if dash: c.line.dash_style = 4
    if head:
        ln = c.line._get_or_add_ln(); e = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}); ln.append(e)
        if both: ln.insert(list(ln).index(e), ln.makeelement(qn("a:headEnd"), {"type": "triangle", "w": "med", "len": "med"}))
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
WHITE = RGBColor(0xFF, 0xFF, 0xFF); PURPLE = ALIGNER[1]
Q_STAR = lambda c: [("q", ("i", c)), ("*", ("i sup", c))]
Q_SUB = lambda m, c: [("q", ("i", c)), (m, ("i sub", c))]
MESH = lambda c: ("ℳ", ("math", c))
PRED = ["depth", "\n", "normals", "\n", "pose"]


def loss_link(x0, x1, y, name):
    arrow(x0 + 0.04, y, x1 - 0.04, y, GRAY, 1.5, dash=True, both=True, name=f"{name} / link")
    text(x0 - 0.05, y - 0.31, x1 - x0 + 0.1, 0.26, [("loss", ("i", GRAY))], 14, name=f"{name} / label")


# ───────────────────────── (a) Training ─────────────────────────
text(0.2, 0.06, 4, 0.5, [("(a) Training", "b")], 22, "l", name="Panel a heading")
Y0, PH = 0.62, 4.12
P1, P2 = (0.2, 6.3), (6.72, 9.08)
Y1, PH2 = Y0 + PH + 0.2, 2.3
P3, PB = (0.2, 7.6), (8.0, 7.8)
panel(P1[0], Y0, P1[1], PH, "Step 1 panel"); panel(P2[0], Y0, P2[1], PH, "Step 2 panel"); panel(P3[0], Y1, P3[1], PH2, "Step 3 panel")
arrow(P1[0] + P1[1] + 0.02, Y0 + 0.36, P2[0] - 0.02, Y0 + 0.36, GRAY, 2.25, name="Step 1 to step 2")
bh = 0.8

# ① K labeled real images -> teacher (image only) and aligner (image + mesh at the perturbed GT state)
step(P1[0], Y0, 1, "Train teacher and aligner", "", P1[1])
ya, yb = 2.02, 3.68                                                    # row centres: teacher, aligner
src = 1.1; xs = P1[0] + 0.2; ysrc = (ya + yb) / 2 - src / 2
pic(f"{A}/reg_overlay_gt.png", xs, ysrc, src, "Labeled real image with mesh outline at the GT state", stack=2)
mlabel("gt_state", xs + src / 2 + 0.07, ysrc - 0.32, CAP, name="GT state caption [LaTeX]")
mlabel("k_labeled", xs + src / 2, ysrc + src + 0.18, CAP, name="Labeled data caption / 1 [LaTeX]")
text(xs - 0.15, ysrc + src + 0.32, src + 0.3, 0.28, [("real images", "b")], CAP, name="Labeled data caption / 2")
xj = xs + src + 0.34
arrow(xs + src + 0.18, (ya + yb) / 2, xj, (ya + yb) / 2, head=False, name="Labeled data / stem")
arrow(xj, ya, xj, yb, head=False, name="Labeled data / split")
ov = 0.92; xo = xj + 0.24; bw = 1.05; xb = xo + ov + 0.22; tp = 0.78; xp = xb + bw + 0.22; tt = 0.66; xt = xp + tp + 0.42
# teacher row
arrow(xj, ya, xb - 0.04, ya, name="Image to teacher")
text(xj + 0.05, ya - 0.33, xb - xj - 0.1, 0.28, [("image", ("i", GRAY))], 14, name="Teacher input note")
box(xb, ya - bh / 2, bw, bh, MODEL("Teacher", "ft"), TEACHER, name="Teacher (training)")
arrow(xb + bw + 0.04, ya, xp - 0.04, ya, name="Teacher to prediction")
box(xp, ya - 0.43, tp, 0.86, PRED, (WHITE, TEACHER[1]), 14, name="Teacher prediction")
text(xp - 0.15, ya + 0.46, tp + 0.3, 0.3, [("prediction", ("", GRAY))], 15, name="Teacher prediction caption")
loss_link(xp + tp, xt, ya, "Teacher loss")
pic(f"{A}/reg_depth.png", xt, ya - tt / 2, tt, "GT label (mesh projected at q*)", border=GREEN)
text(xt - 0.2, ya + 0.46, tt + 0.4, 0.3, [("GT label", ("", GREEN))], 15, name="GT label caption")
# aligner row
arrow(xj, yb, xo - 0.04, yb, name="Labeled data to perturbed pair")
text(xj + 0.1, yb - ov / 2 - 0.34, 1.6, 0.28, [("perturb GT state", ("i", GRAY))], 14, "l", name="Perturbation note")
pic(f"{A}/reg_overlay_perturbed.png", xo, yb - ov / 2, ov, "Same image with mesh outline at q*+delta")
mlabel("perturbed", xo + ov / 2, yb + ov / 2 + 0.18, 15, name="Perturbed pair caption [LaTeX]")
arrow(xo + ov + 0.04, yb, xb - 0.04, yb, name="Perturbed pair to aligner")
box(xb, yb - bh / 2, bw, bh, MODEL("Aligner", "A"), ALIGNER, name="Aligner (training)")
arrow(xb + bw + 0.04, yb, xp - 0.04, yb, name="Aligner to prediction")
box(xp, yb - 0.3, tp, 0.6, " ", (WHITE, PURPLE), 20, name="Aligner prediction"); mlabel("dq", xp + tp / 2, yb, 20, name="Aligner prediction / symbol [LaTeX]")
text(xp - 0.15, yb + 0.33, tp + 0.3, 0.3, [("prediction", ("", GRAY))], 15, name="Aligner prediction caption")
loss_link(xp + tp, xt, yb, "Aligner loss")
box(xt, yb - 0.3, tt, 0.6, " ", (WHITE, ORANGE), 20, name="Correction target"); mlabel("mdelta", xt + tt / 2, yb, 20, name="Correction target / symbol [LaTeX]")
text(xt - 0.2, yb + 0.33, tt + 0.4, 0.3, [("target", ("", ORANGE))], 15, name="Correction target caption")

# ② unlabeled real images -> teacher estimate q0 -> aligner (6 updates) -> qT -> mesh projection = pseudo-label
step(P2[0], Y0, 2, "Pseudo-label the unlabeled real images", "", P2[1])
ts2, bw2 = 1.2, 1.08; g = 0.73; cy = 2.45; y2 = cy - ts2 / 2
x_img = P2[0] + 0.2; x_t = x_img + ts2 + g; x_q0 = x_t + bw2 + g; x_a = x_q0 + ts2 + g; x_qt = x_a + bw2 + g
pic(f"{A}/unl_real.png", x_img, y2, ts2, "Unlabeled real image", stack=2)
text(x_img - 0.05, y2 + ts2 + 0.03, ts2 + 1.0, 0.3, [("unlabeled real images", "b")], CAP, "l", name="Unlabeled data caption")
arrow(x_img + ts2 + 0.18, cy, x_t - 0.04, cy, name="Unlabeled image to teacher")
box(x_t, cy - bh / 2, bw2, bh, MODEL("Teacher", "ft"), TEACHER, name="Teacher (labeling)")
arrow(x_t + bw2 + 0.04, cy, x_q0 - 0.04, cy, name="Teacher to initial state")
text(x_t + bw2, cy - 0.33, g, 0.28, [("predict", ("i", GRAY))], 14, name="Teacher output note")
pic(f"{A}/unl_overlay_teacher.png", x_q0, y2, ts2, "Mesh outline at the teacher state q0")
mlabel("q0", x_q0 + ts2 / 2, y2 + ts2 + 0.18, CAP, name="Initial state caption [LaTeX]")
arrow(x_q0 + ts2 + 0.04, cy, x_a - 0.04, cy, name="Initial state to aligner")
box(x_a, cy - bh / 2, bw2, bh, MODEL("Aligner", "A"), ALIGNER, name="Aligner (labeling)")
text(x_a - 0.3, cy + bh / 2 + 0.04, bw2 + 0.6, 0.3, [("↻ 6 updates", ("", PURPLE))], CAP, name="Iteration note")
pic(f"{PA}/asset_02.png", x_a + bw2 / 2 - 0.29, 1.22, 0.58, "Object mesh")
mlabel("mesh", x_a + bw2 / 2 + 0.36, 1.51, CAP, "l", name="Mesh caption [LaTeX]")
arrow(x_a + bw2 / 2, 1.82, x_a + bw2 / 2, cy - bh / 2 - 0.03, GRAY, 1.5, name="Mesh to aligner")
arrow(x_a + bw2 + 0.04, cy, x_qt - 0.04, cy, name="Aligner to aligned state")
pic(f"{A}/unl_overlay_aligned.png", x_qt, y2, ts2, "Mesh outline at the aligned state qT")
mlabel("qT", x_qt + ts2 / 2, y2 + ts2 + 0.18, CAP, name="Aligned state caption [LaTeX]")
# pseudo-label row (right to left): the mesh projected at qT
tl = 0.74; yl = 3.6; xl = x_img; yc = yl + tl / 2; xg1 = xl + 3 * tl + 0.2
panel(xl - 0.08, yl - 0.08, 3 * tl + 0.36, tl + 0.42, "Pseudo-label group", fill=WHITE, line=PURPLE)
pic(f"{A}/unl_depth.png", xl, yl, tl, "Pseudo-label depth")
pic(f"{A}/unl_normal.png", xl + tl + 0.1, yl, tl, "Pseudo-label normals")
pose_glyph(xl + 2 * (tl + 0.1), yl, tl, "Pseudo-label pose", GREEN)
for k, nm in enumerate(("depth", "normals", "pose")):
    text(xl + k * (tl + 0.1) - 0.1, yl + tl + 0.01, tl + 0.2, 0.28, nm, 15, name=f"Pseudo-label {nm} caption")
xv = x_qt + ts2 / 2
arrow(xv, y2 + ts2 + 0.37, xv, yc, head=False, name="Aligned state to pseudo-label / 1")
arrow(xv, yc, xg1 + 0.14, yc, name="Aligned state to pseudo-label / 2")
xm = (xg1 + xv) / 2
mlabel("pseudo_def", xm, yc - 0.23, 17, name="Pseudo-label definition [LaTeX]")
text(xm - 2.6, yc + 0.08, 5.2, 0.3, [("outlines: mesh contact region at each state", ("i", GRAY))], 14, name="Outline legend")

# ③ student on all real images; GT labels for the K labeled, pseudo-labels for the rest
step(P3[0], Y1, 3, "Train student", "", P3[1])
y3 = Y1 + 1.17; tg = 0.85
xk = P3[0] + 0.28; xu = xk + tg + 0.42
pic(f"{A}/reg_real.png", xk, y3 - tg / 2, tg, "Labeled real images (student input)", stack=1)
pic(f"{A}/unl_real.png", xu, y3 - tg / 2, tg, "Pseudo-labeled real images (student input)", stack=2)
text(xk + tg + 0.07, y3 - 0.2, 0.3, 0.4, "+", 18, name="Student inputs plus")
mlabel("k_labeled", xk + tg / 2, y3 + tg / 2 + 0.18, 15, name="Student labeled caption [LaTeX]")
text(xu - 0.3, y3 + tg / 2 + 0.03, tg + 0.6, 0.3, [("pseudo-labeled", "b")], 15, name="Student pseudo-labeled caption")
sw = 1.15; xst = xu + tg + 0.45
arrow(xu + tg + 0.18, y3, xst - 0.04, y3, name="Real images to student")
box(xst, y3 - bh / 2, sw, bh, MODEL("Student", "fs"), STUDENT, name="Student (training)")
xsp = xst + sw + 0.25
arrow(xst + sw + 0.04, y3, xsp - 0.04, y3, name="Student to prediction")
box(xsp, y3 - 0.43, tp, 0.86, PRED, (WHITE, STUDENT[1]), 14, name="Student prediction")
text(xsp - 0.15, y3 + 0.46, tp + 0.3, 0.3, [("prediction", ("", GRAY))], 15, name="Student prediction caption")
tt3 = 0.72; xpl = P3[0] + P3[1] - 0.22 - tt3; xgt = xpl - 0.4 - tt3
panel(xgt - 0.1, y3 - tt3 / 2 - 0.1, xpl + tt3 + 0.1 - (xgt - 0.1), tt3 + 0.2, "Student targets group", fill=WHITE, line=LIGHT)
loss_link(xsp + tp, xgt - 0.1, y3, "Student loss")
pic(f"{A}/reg_depth.png", xgt, y3 - tt3 / 2, tt3, "GT label (student target)", border=GREEN)
pic(f"{A}/unl_depth.png", xpl, y3 - tt3 / 2, tt3, "Pseudo-label (student target)", border=PURPLE)
text(xgt - 0.2, y3 + tt3 / 2 + 0.13, tt3 + 0.4, 0.3, [("GT label", ("", GREEN))], 14, name="Student GT label caption")
text(xpl - 0.3, y3 + tt3 / 2 + 0.13, tt3 + 0.6, 0.3, [("pseudo-label", ("", PURPLE))], 14, name="Student pseudo-label caption")
arrow(xpl + tt3 / 2, yl + tl + 0.36, xpl + tt3 / 2, y3 - tt3 / 2 - 0.14, PURPLE, 1.75, name="Pseudo-labels to student targets")

# ───────────────────────── (b) Inference ─────────────────────────
panel(PB[0], Y1, PB[1], PH2, "Inference panel", fill=WHITE, line=LIGHT)
text(PB[0] + 0.2, Y1 + 0.1, 4, 0.5, [("(b) Inference", "b")], 22, "l", name="Panel b heading")
ti = 0.85; xi = PB[0] + 0.25
pic(f"{PA}/asset_01.png", xi, y3 - ti / 2, ti, "New real image")
text(xi - 0.25, y3 + ti / 2 + 0.03, ti + 0.5, 0.56, ["real image", "\n", "+ object ID"], 15, name="Inference input caption")
xs2 = xi + ti + 0.32
arrow(xi + ti + 0.04, y3, xs2 - 0.04, y3, name="Inference input to student")
box(xs2, y3 - bh / 2, 1.1, bh, MODEL("Student", "fs"), STUDENT, name="Student (inference)")
to = 0.7; xo2 = xs2 + 1.1 + 0.28
arrow(xs2 + 1.14, y3, xo2 - 0.04, y3, name="Student to predictions")
pic(f"{PA}/asset_03.png", xo2, y3 - to / 2, to, "Predicted depth"); pic(f"{PA}/asset_04.png", xo2 + to + 0.1, y3 - to / 2, to, "Predicted normals")
pose_glyph(xo2 + 2 * (to + 0.1), y3 - to / 2, to, "Predicted pose")
for k, nm in enumerate(("depth", "normals", "pose")):
    text(xo2 + k * (to + 0.1) - 0.1, y3 + to / 2 + 0.02, to + 0.2, 0.3, nm, 15, name=f"Predicted {nm} caption")
xe = xo2 + 3 * to + 0.2; xopt = xe + 0.2; wopt = PB[0] + PB[1] - 0.12 - xopt
panel(xopt, Y1 + 0.3, wopt, PH2 - 0.42, "Optional refinement boundary", fill=None, line=LIGHT, dash=True)
mlabel("optional", xopt + wopt / 2, Y1 + 0.49, 14, name="Optional refinement heading [LaTeX]")
xal = xopt + 0.24; wal = 1.0
arrow(xe + 0.04, y3, xal - 0.04, y3, name="Predicted pose to aligner")
box(xal, y3 - bh / 2, wal, bh, MODEL("Aligner", "A"), ALIGNER, name="Aligner (inference)")
xrp = xal + wal + 0.26
arrow(xal + wal + 0.04, y3, xrp - 0.04, y3, name="Aligner to refined pose")
pose_glyph(xrp, y3 - to / 2, to, "Refined pose", GREEN, rot=-20)
text(xrp - 0.25, y3 + to / 2 + 0.02, to + 0.5, 0.3, "refined pose", 14, name="Refined pose caption")

for s_ in sh:
    st = s_._element.find(qn("p:style"))
    if st is not None: s_._element.remove(st)
prs.save(OUT); print("saved", OUT, "| shapes:", len(sh))
