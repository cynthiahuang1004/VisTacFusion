"""Three-minute silent supplementary-video deck for AlignTac (RoboSoft 2027): 16:9, white background, Arial.
Each slide's notes hold "<seconds> | narration" (the narration doubles as the script in video/SCRIPT.md).
Assets: video/png/fig*.png (pdftoppm -r 300 of figures/fig*.pdf), video/plots/ (video_plots.py), video/streams/ (video_stream_clips.py,
video_newobj_scan.py), video/align_anim/ (video_align_anim.py), video/perturb/ (video_perturb_clip.py), video/robot/robot_press.mp4.
Clips that also exist as GIFs are embedded as animated GIFs in the pptx; the rendered video (video_render.py) overlays the mp4s.
usage (cwd = VisTacFusion): python paper_RoboSoft/scripts/video_deck.py   -> paper_RoboSoft/video/AlignTac_video.pptx, SCRIPT.md
       DECK_POSTER=1 ... -> AlignTac_video_poster.pptx (clips replaced by their first frame, for rendering the backgrounds)"""
import json, os, os.path as osp, subprocess
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

P = "paper_RoboSoft"; V = f"{P}/video"; PNG = f"{V}/png"; ANIM = f"{V}/align_anim"; CROP = f"{V}/crops"; PLOTS = f"{V}/plots"; os.makedirs(CROP, exist_ok=True)
BLACK, WHITE, ACC, BLUE, GREY = RGBColor(0, 0, 0), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0x6A, 0x2B, 0xB8), RGBColor(0x1F, 0x5F, 0xBF), RGBColor(0x6B, 0x6A, 0x66)
FONT = "Arial"; W, H = 13.333, 7.5

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]
script = []
POSTER = bool(os.environ.get("DECK_POSTER"))      # poster deck: clips replaced by their first frame (for rendering backgrounds)
GIF_DECK = bool(os.environ.get("DECK_GIF"))        # every clip embedded as an animated GIF (no mp4 objects) -> AlignTac_video_gif.pptx
PLACEMENTS = []                                    # (slide index, clip path, x, y, w, h in inches) of every embedded clip
NARR = [
    'AlignTac: aligning real tactile images with simulated geometry for label-efficient contact learning.',
    'Tactile sensors turn contact into detailed images, but learning geometry from them needs dense labels. Each label is the object mesh projected at a calibrated, verified robot pose, redone whenever the object moves. Images are cheap; labels are not.',
    "Simulation gives geometry for free, but its renderer must be fitted to the sensor with labeled real pairs. Self-training reuses the predictor's own outputs and inherits its errors.",
    'Our insight: the real image already holds the appearance, and the known mesh already holds the geometry. Only a four-dimensional contact state is missing. AlignTac learns to align the mesh with each real image.',
    'From K labeled contacts per object, twenty-five here, we train a teacher for depth, normals and pose, and an aligner: perturb the known contact state, project the mesh there, and learn to recover the perturbation from the image and the candidate depth.',
    'For each unlabeled image, the teacher gives an initial state and the aligner refines it in six updates. The mesh at the final state is the pseudo-label. A student trains on both sets and predicts from the image and object identity alone.',
    "On held-out frames, red is the mesh at the teacher's state, several degrees off the ridges. Each update moves it; after six, the green outline matches the ground-truth region in white. Over all held-out frames, the mean rotation error falls from 5.7 to 1.2 degrees in three updates.",
    "Alignment is what makes geometry useful. With the same teacher, the mesh at the teacher's pose gives labels with IoU 0.47, worse than teacher depth. Aligned, the IoU is 0.72 and rotation error falls from six degrees to one.",
    'The student trained on aligned geometry has the lowest contact error at every budget on both variants. With 25 labels per object, error drops twenty and thirty-six percent on the two sensors, matching real-only training with twice the labels.',
    'The real-only model blurs or breaks thin ridges and misplaces junctions; the AlignTac student recovers them, also on everyday shapes: a button, a hex key, an edge and a marble. Where depth errors are similar, the surface normals still show the difference.',
    'On a second session with re-placed objects, error stays fourteen to twenty-one percent lower. Pseudo-labeling the new images with the same aligner, with no poses, lowers it a further five percent; on four new objects the gain is thirteen percent.',
    "The same aligner also refines pose at inference: fifteen-degree perturbations, twice the training range, shrink to half a degree, and refining each predictor's own pose lowers its error on both sessions.",
    'In short: real images provide appearance, aligned geometry provides supervision. A few labeled contacts learn the alignment, which labels many more images, transfers to a new session, and refines pose at test time. Code will be released. Thank you.',
]


def crop(src, box, name):
    """box = (x0, y0, x1, y1) as fractions of the source image."""
    im = Image.open(src); w, h = im.size
    out = osp.join(CROP, name); im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))).save(out); return out


def new_slide(seconds, narration, title=None, title_color=BLACK):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = WHITE
    if title:
        text(s, title, 0.3, 0.15, W - 0.6, 0.85, 32, title_color, bold=True)
    s.notes_slide.notes_text_frame.text = f"{seconds} | {narration}"
    script.append((len(prs.slides), seconds, title or "(title slide)", narration))
    return s


def text(s, txt, x, y, w, h, size, color=BLACK, bold=False, align=PP_ALIGN.LEFT, italic=False, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for i, line in enumerate(txt.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        r = p.add_run(); r.text = line; f = r.font; f.name = FONT; f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = color
    return tb


def table(s, rows, x, y, widths, size=15, row_h=0.42, header=True, bold_last_col=False):
    """Plain grid table (white cells, thin borders): rows = list of lists of strings; widths in inches per column."""
    shp = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(sum(widths)), Inches(row_h * len(rows))); tbl = shp.table
    pr = tbl._tbl.tblPr; pr.set("firstRow", "1" if header else "0"); pr.set("bandRow", "0")
    sid = pr.find(qn("a:tableStyleId"))
    if sid is None:
        sid = pr.makeelement(qn("a:tableStyleId"), {}); pr.append(sid)
    sid.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"            # "No Style, Table Grid"
    for j, wdt in enumerate(widths): tbl.columns[j].width = Inches(wdt)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j); c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xF2, 0xF0, 0xEC) if (header and i == 0) else WHITE
            c.margin_left = c.margin_right = Inches(0.08); c.margin_top = c.margin_bottom = Inches(0.03); c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame; tf.word_wrap = True; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run(); r.text = val; f = r.font; f.name = FONT; f.size = Pt(size); f.color.rgb = BLACK
            f.bold = (header and i == 0) or (bold_last_col and j == len(row) - 1 and i > 0)
    return shp


def pic(s, path, x, y, w, h):
    """Fit the image into the box (x, y, w, h) inches, centred, keeping its aspect ratio."""
    iw, ih = Image.open(path).size; sc = min(w / iw, h / ih); pw, ph = iw * sc, ih * sc
    return s.shapes.add_picture(path, Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def movie(s, path, x, y, w, h, poster=None, gif=None):
    """Embed the clip fitted into the box: an animated GIF when `gif` is given (plays in the slideshow), else the mp4 with its first
    frame as poster. The rendered video always overlays the mp4 (PLACEMENTS)."""
    if poster is None:
        poster = osp.join(CROP, osp.basename(path).replace(".mp4", "_poster.png"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-frames:v", "1", poster], check=True)
    iw, ih = Image.open(poster).size; sc = min(w / iw, h / ih); pw, ph = iw * sc, ih * sc; px, py = x + (w - pw) / 2, y + (h - ph) / 2
    PLACEMENTS.append(dict(slide=len(prs.slides), clip=path, x=px, y=py, w=pw, h=ph))
    if POSTER:
        return s.shapes.add_picture(poster, Inches(px), Inches(py), Inches(pw), Inches(ph))
    if GIF_DECK and not gif:                                   # all-GIF deck: convert the mp4 once (cached next to it)
        gif = path.replace(".mp4", "_deck.gif")
        if not osp.exists(gif):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vf", "fps=8,scale=760:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse", gif], check=True)
    if gif:
        return s.shapes.add_picture(gif, Inches(px), Inches(py), Inches(pw), Inches(ph))
    return s.shapes.add_movie(path, Inches(px), Inches(py), Inches(pw), Inches(ph), poster_frame_image=poster, mime_type="video/mp4")


def caption(s, txt, y=6.55, color=ACC, size=22):
    text(s, txt, 0.6, y, W - 1.2, 0.7, size, color, bold=True, align=PP_ALIGN.CENTER)


# ---------------------------------------------------------------- assets
f1 = f"{PNG}/fig1_teaser.png"
f1b = crop(f1, (0, 848 / 2563, 1, 1388 / 2563), "fig1b.png"); f1c = crop(f1, (0, 1407 / 2563, 1, 2389 / 2563), "fig1c.png")
f2 = f"{PNG}/fig2_pipeline.png"
f2_s1 = crop(f2, (0.005, 0.07, 0.412, 0.635), "fig2_step1.png")
f2_s3 = crop(f2, (0.005, 0.645, 0.495, 0.965), "fig2_step3.png"); f2_inf = crop(f2, (0.498, 0.645, 0.995, 0.965), "fig2_inference.png")
STREAM = f"{V}/streams"; PERTURB = f"{V}/perturb/perturb.mp4"
clip = lambda name: (f"{STREAM}/{name}.mp4", f"{STREAM}/{name}.gif")

# ---------------------------------------------------------------- slides
s = new_slide(8, NARR[0])
text(s, "AlignTac", 0.7, 1.9, W - 1.4, 1.1, 54, ACC, bold=True, align=PP_ALIGN.CENTER)
text(s, "Aligning Real Tactile Images with Simulated Geometry\nfor Label-Efficient Contact Learning", 0.7, 3.1, W - 1.4, 2.0, 36, BLACK, bold=True, align=PP_ALIGN.CENTER)
text(s, "Anonymous submission to RoboSoft 2027   ·   no audio", 0.7, 5.6, W - 1.4, 0.6, 22, GREY, align=PP_ALIGN.CENTER)

s = new_slide(14, NARR[1], "Tactile images are easy to collect; their labels are not")
movie(s, f"{V}/robot/robot_press.mp4", 0.35, 1.1, 3.3, 5.3)
m, g = clip("pred_pattern_32_run1"); movie(s, m, 3.9, 1.1, 9.2, 3.25, gif=g)
text(s, "Each press yields a tactile image. Its label, the mesh projected at the robot pose, requires calibration and verification, repeated whenever the object is re-placed.", 3.9, 4.45, 9.2, 0.9, 17, BLACK, align=PP_ALIGN.CENTER)
text(s, "Left: data collection (1.5× speed).  Right: second-session stream with the contact-depth predictions of two K = 25 students.", 3.9, 5.6, 9.2, 0.7, 15, GREY, italic=True, align=PP_ALIGN.CENTER)
caption(s, "Images are cheap; labels are not.", 6.5)

s = new_slide(12, NARR[2], "Simulation and self-training each pay a different price")
pic(s, f1b, 1.2, 1.1, 10.9, 4.8)
text(s, "Simulation: a sensor-specific renderer,\nfitted on labeled real pairs", 0.6, 6.0, 6.2, 1.1, 22, BLACK, bold=True, align=PP_ALIGN.CENTER)
text(s, "Self-training: labels from the teacher's own predictions,\nerrors included", 6.6, 6.0, 6.2, 1.1, 22, BLACK, bold=True, align=PP_ALIGN.CENTER)

s = new_slide(14, NARR[3], "AlignTac: keep the real appearance, align the known geometry", ACC)
pic(s, f1c, 1.0, 1.1, 11.3, 5.2)
caption(s, "Only a 4-D contact state (θ, x, y, z) separates a real image from its dense label", 6.5, BLACK)

s = new_slide(16, NARR[4], "Step 1: a teacher and an aligner from K labeled contacts")
pic(s, f2_s1, 0.3, 1.1, 5.2, 4.3); movie(s, PERTURB, 5.7, 1.1, 7.4, 4.3, gif=f"{V}/perturb/perturb.gif")
text(s, "Teacher: K labeled contacts → depth, normals, pose", 0.3, 5.5, 5.2, 0.9, 19, BLACK, bold=True, align=PP_ALIGN.CENTER)
text(s, "Aligner: perturb the known state, project the mesh there,\nlearn to undo the perturbation", 5.7, 5.5, 7.4, 0.9, 19, BLACK, bold=True, align=PP_ALIGN.CENTER)
caption(s, "One labeled contact → many training pairs:  (image, mesh depth at q* + δ) → −δ", 6.5, ACC, 20)

s = new_slide(16, NARR[5], "Steps 2 and 3: pseudo-label the unlabeled images, train the student")
m, g = clip("label_pattern_32_run1"); movie(s, m, 0.3, 1.1, 8.6, 3.05, gif=g); pic(s, f2_s3, 9.1, 1.1, 4.0, 3.05)
text(s, "Unlabeled second-session stream: teacher state (red) → aligner updates (green) → mesh depth as the pseudo-label", 0.3, 4.2, 8.6, 0.8, 17, BLACK, bold=True, align=PP_ALIGN.CENTER)
text(s, "Student: labeled + pseudo-labeled images", 9.1, 4.2, 4.0, 0.5, 17, BLACK, bold=True, align=PP_ALIGN.CENTER)
pic(s, f2_inf, 2.9, 5.0, 7.5, 1.55)
caption(s, "Inference uses only the image and the object identity;  mesh-based pose refinement is optional", 6.6, ACC, 20)

s = new_slide(16, NARR[6], "Alignment on held-out real contacts", ACC)
movie(s, f"{ANIM}/align_anim.mp4", 0.3, 1.05, 12.73, 3.2, poster=f"{ANIM}/strip_it0.png")
text(s, "red: mesh at the teacher state     green: after the aligner updates     white: ground-truth contact region", 0.3, 4.3, W - 0.6, 0.45, 17, BLACK, bold=True, align=PP_ALIGN.CENTER)
movie(s, f"{PLOTS}/conv_live.mp4", 0.3, 4.8, 3.9, 2.6, poster=f"{PLOTS}/conv_live_poster.png")
CV = json.load(open("ablation/pilot_robosoft/aligner_convergence_video.json"))                 # mesh-label IoU / MAE per update (same 472 frames)
tr = json.load(open("ablation/pilot_robosoft/aligner_convergence.json"))["teacher"]             # rotation / translation: the paper's numbers (Fig. 7a)
rows = [["Aligner updates", "Rotation [deg]", "Translation [mm]", "Label IoU", "Contact MAE [mm]"]]
for it in (0, 1, 3, 6):
    rows.append([f"n = {it}" + ("  (teacher state)" if it == 0 else ""), f"{tr['rot']['mean'][it]:.1f}", f"{tr['trans']['mean'][it]:.2f}", f"{CV['label'][str(it)]['iou']:.2f}", f"{CV['label'][str(it)]['cmae']:.2f}"])
table(s, rows, 4.6, 4.8, [2.7, 1.4, 1.6, 1.2, 1.6], size=15, row_h=0.4)
text(s, f"Means over all {CV['n']} held-out frames of sensor A (K = 25); label IoU and contact MAE compare the mesh at the current state with the ground truth.", 4.6, 7.0, 8.5, 0.45, 12.5, GREY, italic=True)

s = new_slide(14, NARR[7], "Alignment turns known geometry into accurate labels")
movie(s, f"{PLOTS}/align_toggle.mp4", 0.3, 1.1, 6.0, 5.0, poster=f"{PLOTS}/align_toggle_0.png")
movie(s, f"{PLOTS}/labels_anim.mp4", 6.5, 1.1, 6.55, 3.0, poster=f"{PLOTS}/labels_final.png")
text(s, "Label quality on the unlabeled training images (both GelSlim 4.0 variants)", 6.5, 4.15, 6.55, 0.5, 16, GREY, align=PP_ALIGN.CENTER)
table(s, [["Same teacher, K = 25", "Label IoU", "Rotation error"], ["Mesh at the teacher pose", "0.47", "6.0°"], ["Aligned mesh", "0.72", "1.1°"]],
      6.6, 4.8, [3.1, 1.5, 1.9], size=17, row_h=0.42)
text(s, "Left: three unlabeled frames, mesh at the teacher pose vs. at the aligned pose", 0.3, 6.2, 6.0, 0.6, 16, GREY, align=PP_ALIGN.CENTER)

s = new_slide(14, NARR[8], "Label efficiency on two GelSlim 4.0 variants")
pic(s, f"{PLOTS}/kshot6.png", 0.3, 1.05, 12.73, 3.95)
table(s, [["K = 25", "Real only", "AlignTac", "Change"],
          ["Contact MAE [mm], sensor A", "0.291", "0.233", "−20%"], ["Rotation [deg], sensor A", "5.7", "3.2", "−44%"], ["Translation [mm], sensor A", "1.09", "0.88", "−19%"],
          ["Contact MAE [mm], sensor B", "0.127", "0.081", "−36%"], ["Rotation [deg], sensor B", "8.6", "8.3", "−4%"], ["Translation [mm], sensor B", "1.18", "1.11", "−6%"]],
      0.5, 5.05, [3.4, 1.4, 1.4, 1.2], size=14, row_h=0.33, bold_last_col=True)
text(s, "AlignTac matches real-only accuracy obtained with 2× the labels.\n\nPose gains are large on sensor A and small on sensor B.\n\nLearned renderer (K) and self-training use the same K labels and the same unlabeled images.", 8.2, 5.05, 4.9, 2.35, 15, BLACK, anchor=MSO_ANCHOR.MIDDLE)

s = new_slide(14, NARR[9], "Contact depth and normals on held-out frames")
movie(s, f"{PLOTS}/qual_cycle.mp4", 0.3, 1.05, 12.73, 2.95, poster=f"{PLOTS}/qual_0.png")
m, g = clip("pred_pattern_36_run1"); movie(s, m, 0.3, 4.2, 7.7, 2.7, gif=g)
text(s, "Top: held-out frames of both sensor variants and of the four additional objects (K = 25). Numbers: contact MAE and mean normal angle error over the contact region. Where depth errors are similar, the normals still differ.\n\nBottom: live predictions on a second-session stream.", 8.2, 4.2, 4.9, 2.8, 16, BLACK, anchor=MSO_ANCHOR.MIDDLE)

s = new_slide(16, NARR[10], "Robust across sessions, sensor variants and objects")
movie(s, f"{PLOTS}/qual_p4.mp4", 0.3, 1.05, 5.5, 2.95, poster=f"{PLOTS}/qual_p4_poster.png")
table(s, [["Setting", "Contact MAE", "Pose / IoU"],
          ["Second session, zero-shot (objects re-placed)", "−14 to −21%", "—"],
          ["Label-free adaptation (unlabeled session-2 images)", "further −5%", "10.4° → 7.9°"],
          ["Dot-marker variant, retrained", "−36% at K = 25", "—"],
          ["Four additional objects, K = 25", "0.235 → 0.204 mm (−13%)", "IoU 0.70 → 0.77"]],
      5.75, 1.05, [3.7, 2.0, 1.8], size=13.5, row_h=0.5)
text(s, "changes vs. real-only training;  pose column: rotation error", 5.75, 3.65, 7.5, 0.35, 13, GREY, italic=True)
m, g = clip("scan_hex_key_v2"); movie(s, m, 0.3, 4.3, 6.3, 1.55, gif=g); m, g = clip("scan_button_v2"); movie(s, m, 6.8, 4.3, 6.3, 1.55, gif=g)
m, g = clip("scan_marble_v2"); movie(s, m, 0.3, 5.9, 6.3, 1.55, gif=g); m, g = clip("scan_edge_v2"); movie(s, m, 6.8, 5.9, 6.3, 1.55, gif=g)

s = new_slide(12, NARR[11], "The aligner also refines pose at test time")
pic(s, f"{PLOTS}/conv_b.png", 0.3, 1.05, 6.2, 4.4); pic(s, f"{PLOTS}/poseref.png", 6.7, 1.05, 6.3, 4.4)
text(s, "Left: from a rotated GT pose, n updates bring the error below 1° up to 15°, twice the training range", 0.3, 5.55, 6.2, 0.9, 16, BLACK, align=PP_ALIGN.CENTER)
text(s, "Right: each predictor's own pose refined at inference (hollow → filled), session 1 and session 2", 6.7, 5.55, 6.3, 0.9, 16, BLACK, align=PP_ALIGN.CENTER)
caption(s, "15° perturbations → 0.5° after six updates", 6.5)

s = new_slide(12, NARR[12], "Takeaways", ACC)
text(s, "•  Real images supply appearance; the known mesh supplies geometry; AlignTac learns the contact state that connects them\n\n"
        "•  K = 25 labeled contacts per object: contact MAE −20% / −36% on two GelSlim 4.0 variants, matching real-only training with 2× the labels\n\n"
        "•  Alignment, not geometry alone, makes the labels accurate (label IoU 0.47 → 0.72 with the same teacher)\n\n"
        "•  The same aligner pseudo-labels a new session without re-calibration and refines pose at inference\n\n"
        "•  Code will be released",
     0.9, 1.3, W - 1.8, 5.2, 24, BLACK, bold=True, anchor=MSO_ANCHOR.MIDDLE)

prs.save(f"{V}/AlignTac_video_poster.pptx" if POSTER else f"{V}/AlignTac_video_gif.pptx" if GIF_DECK else f"{V}/AlignTac_video.pptx")
json.dump(dict(placements=PLACEMENTS, seconds=[x[1] for x in script]), open(f"{V}/placements.json", "w"), indent=1)
total = sum(x[1] for x in script)
with open(f"{V}/SCRIPT.md", "w") as f:
    f.write(f"# AlignTac video script  (planned {total} s ≈ {total // 60}:{total % 60:02d}; narration ~{sum(len(x[3].split()) for x in script)} words)\n\n")
    f.write("| # | s | slide | narration |\n|---|---|---|---|\n")
    for n, sec, title, nar in script:
        f.write(f"| {n} | {sec} | {title} | {nar} |\n")
print(f"{V}/AlignTac_video.pptx  slides={len(prs.slides)}  planned={total}s  words={sum(len(x[3].split()) for x in script)}")
