"""Three-minute supplementary-video deck for AlignTac (RoboSoft 2027), in the style of the ICRA GelSlim 5.0 video deck
(16:9, black background, Times New Roman, gold title / white headings). Each slide's notes hold "<seconds> | narration".
Assets: paper_RoboSoft/video/png/fig*.png (pdftoppm -r 300 of figures/fig*.pdf), video/align_anim/ (video_align_anim.py),
figures_archive/intro_figure/assets/images/acquisition_photo_crop.png.
usage (cwd = VisTacFusion): python paper_RoboSoft/scripts/video_deck.py   -> paper_RoboSoft/video/AlignTac_video.pptx, SCRIPT.md"""
import json, os, os.path as osp, subprocess
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

P = "paper_RoboSoft"; V = f"{P}/video"; PNG = f"{V}/png"; ANIM = f"{V}/align_anim"; CROP = f"{V}/crops"; os.makedirs(CROP, exist_ok=True)
GOLD, WHITE, BLUE, RED, GREY = RGBColor(0xFF, 0xC0, 0x00), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0x00, 0xB0, 0xF0), RGBColor(0xFF, 0x5A, 0x60), RGBColor(0xBB, 0xBB, 0xBB)
FONT = "Times New Roman"; W, H = 13.333, 7.5

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]
script = []
POSTER = bool(os.environ.get("DECK_POSTER"))      # poster deck: clips replaced by their first frame (for rendering backgrounds)
PLACEMENTS = []                                    # (slide index, clip path, x, y, w, h in inches) of every embedded clip
# narration per slide (~2.4 words/s)
NARR = [
    'AlignTac: aligning real tactile images with simulated geometry for label-efficient contact learning.',
    'Tactile sensors turn contact into detailed images, but learning geometry from them needs dense labels. Each label is the object mesh projected at a calibrated, verified robot pose, redone whenever the object moves. Images are cheap; labels are not.',
    "Simulation gives geometry for free, but its renderer must be fitted to the sensor with labeled real pairs. Self-training reuses the predictor's own outputs and inherits its errors.",
    'Our insight: the real image already holds the appearance, and the known mesh already holds the geometry. Only a four-dimensional contact state is missing. AlignTac learns to align the mesh with each real image.',
    'From K labeled contacts per object, twenty-five here, we train a teacher for depth, normals and pose, and an aligner: perturb the known contact state, project the mesh there, and learn to recover the perturbation from the image and the candidate depth.',
    'For each unlabeled image, the teacher gives an initial state and the aligner refines it in six updates. The mesh at the final state is the pseudo-label. A student trains on both sets and predicts from the image and object identity alone.',
    "On held-out frames, red is the mesh at the teacher's state, several degrees off the ridges. Each update moves it; after six, the green outline matches the ground-truth region in white, with errors below half a degree.",
    "Alignment is what makes geometry useful. With the same teacher, the mesh at the teacher's pose gives labels with IoU 0.47, worse than teacher depth. Aligned, the IoU is 0.72 and rotation error falls from six degrees to one.",
    'The student trained on aligned geometry has the lowest contact error at every budget on both variants. With 25 labels per object, error drops twenty and thirty-six percent on the two sensors, matching real-only training with twice the labels.',
    'The real-only model blurs or breaks thin ridges and misplaces junctions; the AlignTac student recovers them, also on everyday shapes: a button, a hex key, an edge and a marble.',
    'On a second session with re-placed objects, error stays fourteen to twenty-one percent lower. Pseudo-labeling the new images with the same aligner, with no poses, lowers it a further five percent; on four new objects the gain is thirteen percent.',
    "The same aligner also refines pose at inference: from the teacher's pose, rotation error drops from 5.7 to 1.2 degrees in three updates, and fifteen-degree perturbations shrink to half a degree.",
    'In short: real images provide appearance, aligned geometry provides supervision. A few labeled contacts learn the alignment, which labels many more images, transfers to a new session, and refines pose at test time. Code will be released. Thank you.',
]


def crop(src, box, name):
    """box = (x0, y0, x1, y1) as fractions of the source image."""
    im = Image.open(src); w, h = im.size
    out = osp.join(CROP, name); im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))).save(out); return out


def new_slide(seconds, narration, title=None, title_color=WHITE):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = RGBColor(0, 0, 0)
    if title:
        text(s, title, 0.25, 0.12, W - 0.5, 0.8, 28, title_color, bold=True)
    s.notes_slide.notes_text_frame.text = f"{seconds} | {narration}"
    script.append((len(prs.slides), seconds, title or "(title slide)", narration))
    return s


def text(s, txt, x, y, w, h, size, color=WHITE, bold=False, align=PP_ALIGN.LEFT, italic=False, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for i, line in enumerate(txt.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        r = p.add_run(); r.text = line; f = r.font; f.name = FONT; f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = color
    return tb


def pic(s, path, x, y, w, h):
    """Fit the image into the box (x, y, w, h) inches, centred, keeping its aspect ratio."""
    iw, ih = Image.open(path).size; sc = min(w / iw, h / ih); pw, ph = iw * sc, ih * sc
    return s.shapes.add_picture(path, Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def movie(s, path, x, y, w, h, poster=None):
    """Embed an mp4 fitted into the box; poster = first frame of the clip (extracted with ffmpeg) unless given."""
    if poster is None:
        poster = osp.join(CROP, osp.basename(path).replace(".mp4", "_poster.png"))
        if not osp.exists(poster):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-frames:v", "1", poster], check=True)
    iw, ih = Image.open(poster).size; sc = min(w / iw, h / ih); pw, ph = iw * sc, ih * sc; px, py = x + (w - pw) / 2, y + (h - ph) / 2
    PLACEMENTS.append(dict(slide=len(prs.slides), clip=path, x=px, y=py, w=pw, h=ph))
    if POSTER:
        return s.shapes.add_picture(poster, Inches(px), Inches(py), Inches(pw), Inches(ph))
    return s.shapes.add_movie(path, Inches(px), Inches(py), Inches(pw), Inches(ph), poster_frame_image=poster, mime_type="video/mp4")


def caption(s, txt, y=6.55, color=GOLD, size=20):
    text(s, txt, 0.6, y, W - 1.2, 0.7, size, color, bold=True, align=PP_ALIGN.CENTER)


# ---------------------------------------------------------------- assets
f1 = f"{PNG}/fig1_teaser.png"
f1a = crop(f1, (0, 0, 1, 829 / 2563), "fig1a.png"); f1b = crop(f1, (0, 848 / 2563, 1, 1388 / 2563), "fig1b.png"); f1c = crop(f1, (0, 1407 / 2563, 1, 2389 / 2563), "fig1c.png")
f2 = f"{PNG}/fig2_pipeline.png"
f2_s1 = crop(f2, (0.005, 0.07, 0.412, 0.635), "fig2_step1.png"); f2_s2 = crop(f2, (0.415, 0.07, 0.995, 0.635), "fig2_step2.png")
f2_s3 = crop(f2, (0.005, 0.645, 0.495, 0.965), "fig2_step3.png"); f2_inf = crop(f2, (0.498, 0.645, 0.995, 0.965), "fig2_inference.png")
f4abc = crop(f"{PNG}/fig4_kshot.png", (0, 0, 0.5, 1), "fig4_sensorA.png"); f4def = crop(f"{PNG}/fig4_kshot.png", (0.5, 0, 1, 1), "fig4_sensorB.png")
f6 = f"{PNG}/fig6_qual.png"; f6_new = crop(f6, (0.335, 0, 0.665, 1), "fig6_new_objects.png")
photo = f"{P}/figures_archive/intro_figure/assets/images/acquisition_photo_crop.png"
anim_mp4, anim_poster, anim_gif = f"{ANIM}/align_anim.mp4", f"{ANIM}/strip_it0.png", f"{ANIM}/align_anim_small.gif"
STREAM = f"{V}/streams"; PERTURB = f"{V}/perturb/perturb.mp4"
if not osp.exists(anim_gif):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", anim_mp4, "-vf", "fps=8,scale=1600:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse", anim_gif], check=True)

# ---------------------------------------------------------------- slides
s = new_slide(8, NARR[0])
text(s, "AlignTac:", 0.7, 2.0, W - 1.4, 1.1, 48, GOLD, bold=True, align=PP_ALIGN.CENTER)
text(s, "Aligning Real Tactile Images with Simulated Geometry\nfor Label-Efficient Contact Learning", 0.7, 3.1, W - 1.4, 2.0, 36, WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, "Anonymous submission to RoboSoft 2027   ·   no audio", 0.7, 5.6, W - 1.4, 0.6, 20, GREY, align=PP_ALIGN.CENTER)

s = new_slide(14, NARR[1],
              "Tactile images are easy to collect, their geometric labels are not")
pic(s, photo, 0.4, 1.05, 3.6, 5.3); pic(s, f1a, 4.2, 1.05, 8.9, 5.3)
text(s, "[ robot video clip here ]", 0.4, 6.4, 3.6, 0.4, 14, GREY, italic=True, align=PP_ALIGN.CENTER)
caption(s, "Every labeled contact requires calibration and verification of the robot pose", 6.6)

s = new_slide(12, NARR[2],
              "Simulation and self-training each pay a different price")
pic(s, f1b, 1.2, 1.05, 10.9, 4.9)
text(s, "Synthetic images need sensor-specific modeling, fitted on labeled real pairs", 0.6, 6.05, 6.2, 1.0, 20, WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, "Self-training labels with the teacher's own predictions and inherits its errors", 6.6, 6.05, 6.2, 1.0, 20, WHITE, bold=True, align=PP_ALIGN.CENTER)

s = new_slide(14, NARR[3],
              "AlignTac: keep the real appearance, align the known geometry", GOLD)
pic(s, f1c, 1.0, 1.05, 11.3, 5.3)
caption(s, "Only a 4-D contact state (θ, x, y, z) separates a real image from its dense label", 6.55, WHITE)

s = new_slide(16, NARR[4],
              "Step 1: a teacher and an aligner from K labeled contacts")
pic(s, f2_s1, 0.3, 1.05, 5.3, 4.6); movie(s, PERTURB, 5.7, 1.05, 7.4, 4.6)
text(s, "Teacher: K labeled contacts → depth, normals, pose", 0.3, 5.75, 5.3, 0.5, 18, WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, "Aligner: perturb the known state, project the mesh, learn to undo the perturbation", 5.7, 5.75, 7.4, 0.5, 18, WHITE, bold=True, align=PP_ALIGN.CENTER)
caption(s, "One labeled contact gives many (image, perturbed mesh depth → −δ) training pairs", 6.45, GOLD)

s = new_slide(16, NARR[5],
              "Steps 2 and 3: pseudo-label the unlabeled images, train the student")
movie(s, f"{STREAM}/label_pattern_32_run1.mp4", 0.3, 1.0, 8.6, 3.05); pic(s, f2_s3, 9.1, 1.0, 4.0, 3.05)
text(s, "Unlabeled stream from a second session: teacher state (red) → six aligner updates (green) → mesh depth as the pseudo-label", 0.3, 4.1, 8.6, 0.8, 16, WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, "Student: labeled + pseudo-labeled images", 9.1, 4.1, 4.0, 0.5, 16, WHITE, bold=True, align=PP_ALIGN.CENTER)
pic(s, f2_inf, 2.9, 4.95, 7.5, 1.6)
caption(s, "Inference needs only the image and the object identity;  mesh-assisted pose refinement is optional", 6.6, GOLD, 18)

s = new_slide(16, NARR[6],
              "Alignment in action: six updates on held-out real contacts", GOLD)
movie(s, anim_mp4, 0.4, 1.15, 12.53, 3.13, poster=anim_poster)
text(s, "red: teacher initial state q0     green: after six aligner updates     white: ground-truth contact region", 0.6, 4.45, W - 1.2, 0.5, 18, WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, "Rotation error on these four frames: 6–8° → 0.0–0.4°;  translation: 0.2–1.3 mm → 0.1–0.3 mm", 0.6, 5.0, W - 1.2, 0.5, 18, GREY, align=PP_ALIGN.CENTER)
text(s, "(video embedded; align_anim_small.gif is the same sequence as an animated GIF)", 0.6, 5.5, W - 1.2, 0.4, 12, GREY, italic=True, align=PP_ALIGN.CENTER)

s = new_slide(14, NARR[7],
              "Accurate alignment turns known geometry into accurate labels")
pic(s, f"{PNG}/fig3_align.png", 0.4, 1.05, 7.2, 4.4); pic(s, f"{PNG}/fig5_labels.png", 7.8, 1.05, 5.2, 4.4)
text(s, "Same teacher, K = 25:  label IoU 0.47 (mesh at teacher pose)  →  0.72 (aligned mesh);  label rotation error 6.0° → 1.1°", 0.6, 5.65, W - 1.2, 0.6, 19, GOLD, bold=True, align=PP_ALIGN.CENTER)
text(s, "Right: teacher vs. aligned-mesh label quality across budgets on both GelSlim 4.0 variants", 0.6, 6.3, W - 1.2, 0.5, 17, WHITE, align=PP_ALIGN.CENTER)

s = new_slide(16, NARR[8],
              "Label efficiency on two GelSlim 4.0 variants")
pic(s, f"{PNG}/fig4_kshot.png", 0.3, 1.0, 12.7, 3.7)
text(s, "K = 25:  contact MAE −20% (marker-free)  and  −36% (dot-marker)  vs. real-only training", 0.6, 4.85, W - 1.2, 0.6, 22, GOLD, bold=True, align=PP_ALIGN.CENTER)
text(s, "Matches real-only accuracy obtained with 2× the labels;  pose error also falls (3.2° vs. 5.7° at K = 25)", 0.6, 5.5, W - 1.2, 0.6, 20, WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, "Learned renderer (K) and self-training use the same K labels / the same unlabeled images", 0.6, 6.15, W - 1.2, 0.5, 16, GREY, align=PP_ALIGN.CENTER)

s = new_slide(12, NARR[9],
              "Contact-depth predictions on held-out frames")
movie(s, f"{STREAM}/pred_pattern_32_run1.mp4", 0.3, 1.0, 6.3, 2.3); movie(s, f"{STREAM}/pred_pattern_36_run1.mp4", 6.8, 1.0, 6.3, 2.3)
pic(s, f6, 0.3, 3.5, 12.7, 2.9)
text(s, "top: live predictions on a second-session stream (approach, press, lift);  arrow: AlignTac planar pose", 0.6, 6.4, W - 1.2, 0.45, 16, WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, "bottom: held-out frames — real image | real only | AlignTac | ground truth;  numbers: contact MAE (mm)", 0.6, 6.85, W - 1.2, 0.45, 15, GREY, align=PP_ALIGN.CENTER)

s = new_slide(16, NARR[10],
              "Robust across sessions, sensor variants and objects")
for i, (hd, body) in enumerate([("Second session, zero-shot", "objects re-placed, no retraining\n\ncontact MAE −14 to −21%\nvs. real-only training"),
                                 ("Label-free adaptation", "unlabeled session-2 images,\npseudo-labeled by the same aligner\n\na further −5% contact MAE\nrotation 10.4° → 7.9°"),
                                 ("Dot-marker variant", "retrained, same aligner settings\n\ncontact MAE −36% at K = 25\n−15 to −33% at other budgets")]):
    x = 0.4 + i * 4.2
    text(s, hd, x, 1.05, 4.1, 0.5, 21, BLUE, bold=True, align=PP_ALIGN.CENTER)
    text(s, body, x, 1.55, 4.1, 1.9, 17, WHITE, bold=True, align=PP_ALIGN.CENTER)
movie(s, f"{STREAM}/scan_hex_key_v2.mp4", 0.3, 3.95, 6.3, 1.65); movie(s, f"{STREAM}/scan_button_v2.mp4", 6.8, 3.95, 6.3, 1.65)
movie(s, f"{STREAM}/scan_marble_v2.mp4", 0.3, 5.65, 6.3, 1.65); movie(s, f"{STREAM}/scan_edge_v2.mp4", 6.8, 5.65, 6.3, 1.65)
text(s, "Four additional objects, consecutive captures:  real image | real only | AlignTac | ground truth   —   contact MAE 0.235 → 0.204 mm (−13%, three seeds)", 0.3, 3.5, W - 0.6, 0.4, 14, GOLD, bold=True, align=PP_ALIGN.CENTER)

s = new_slide(12, NARR[11],
              "The aligner also refines pose at test time")
pic(s, f"{PNG}/fig7_conv.png", 0.4, 1.05, 6.3, 4.3); pic(s, f"{PNG}/fig8_poseref.png", 6.9, 1.05, 6.1, 4.3)
text(s, "From the teacher pose: 5.7° → 1.2° after three updates;   15° perturbations → 0.5°", 0.6, 5.55, W - 1.2, 0.6, 22, GOLD, bold=True, align=PP_ALIGN.CENTER)
text(s, "Optional: standard inference needs only the image and the object identity", 0.6, 6.2, W - 1.2, 0.5, 18, WHITE, align=PP_ALIGN.CENTER)

s = new_slide(14, NARR[12],
              "Takeaways", GOLD)
text(s, "•  Real images supply appearance; a known mesh supplies geometry; AlignTac learns the contact state that connects them\n\n"
        "•  K = 25 labeled contacts per object: contact MAE −20% / −36% on two GelSlim 4.0 variants, matching real-only with 2× the labels\n\n"
        "•  Alignment, not geometry alone, makes the labels accurate (label IoU 0.47 → 0.72 with the same teacher)\n\n"
        "•  The same aligner pseudo-labels a new session without re-calibration and refines pose at inference\n\n"
        "•  Code will be released",
     0.9, 1.3, W - 1.8, 5.2, 24, WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)

prs.save(f"{V}/AlignTac_video_poster.pptx" if POSTER else f"{V}/AlignTac_video.pptx")
json.dump(dict(placements=PLACEMENTS, seconds=[x[1] for x in script]), open(f"{V}/placements.json", "w"), indent=1)
total = sum(x[1] for x in script)
with open(f"{V}/SCRIPT.md", "w") as f:
    f.write(f"# AlignTac video script  (planned {total} s ≈ {total // 60}:{total % 60:02d}; narration ~{sum(len(x[3].split()) for x in script)} words)\n\n")
    f.write("| # | s | slide | narration |\n|---|---|---|---|\n")
    for n, sec, title, nar in script:
        f.write(f"| {n} | {sec} | {title} | {nar} |\n")
print(f"{V}/AlignTac_video.pptx  slides={len(prs.slides)}  planned={total}s  words={sum(len(x[3].split()) for x in script)}")
