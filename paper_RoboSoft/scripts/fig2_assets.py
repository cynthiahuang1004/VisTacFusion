"""Image assets for the revised pipeline figure (Fig. 2), all from one object (pattern_06, sensor A, session 1).
usage (cwd = VisTacFusion): conda run -n vistacfusion python paper_RoboSoft/scripts/fig2_assets.py
Writes paper_RoboSoft/figures/intro_figure/fig2_assets/:
  reg_*   a registered contact (one of the K=25 labeled frames): image, mesh outline at q* and at q*+delta, depth, normals
  unl_*   an unlabeled contact (frame 0479): image, mesh outline at the teacher state q0 and the aligned state qT, depth, normals"""
import json, math, os, sys
import cv2, numpy as np
from PIL import Image
from matplotlib import cm
sys.path.insert(0, "scripts")
import build_real_filtered_new as B
B.VIEW = 0.017502
G = "/media/hdd2/ihsuan/gs_blender"; OUT = "paper_RoboSoft/figures/intro_figure/fig2_assets"; os.makedirs(OUT, exist_ok=True)
RED, GREEN, ORANGE = (235, 60, 50), (40, 220, 90), (255, 150, 30); UP = 3
obj = "pattern_06_5_lines_angle_1"

def overlay(rgb, depths_colors, name):
    im = cv2.resize(rgb, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
    for d, color in depths_colors:
        m = cv2.resize((d > 0.05).astype(np.uint8), None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cs, -1, (255, 255, 255), 13, cv2.LINE_AA); cv2.drawContours(im, cs, -1, color, 8, cv2.LINE_AA)
    Image.fromarray(im).save(f"{OUT}/{name}.png")
def depth_png(d, name):
    Image.fromarray((cm.viridis(np.clip(d / 1.2, 0, 1))[..., :3] * 255).astype(np.uint8)).resize((448, 448), Image.BILINEAR).save(f"{OUT}/{name}.png")
def normal_png(d, name):
    px = B.VIEW * 1000 / B.S; d = cv2.GaussianBlur(d, (0, 0), 1.0)
    gy, gx = np.gradient(d, px); n = np.dstack([-gx, gy, np.ones_like(d)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    Image.fromarray(((n + 1) / 2 * 255).astype(np.uint8)).resize((448, 448), Image.BILINEAR).save(f"{OUT}/{name}.png")
def save_rgb(rgb, name): Image.fromarray(rgb).resize((448, 448), Image.BICUBIC).save(f"{OUT}/{name}.png")

# registered contact: K=25 labeled frame; delta is an illustrative perturbation
fr = sys.argv[1] if len(sys.argv) > 1 else "0064"
t = f"{G}/real_filtered/{obj}/session_000/sensor_0000/samples/{fr}.png"
P = json.load(open(t.replace("samples", "raw_data").replace(".png", "_pose.json")))
q = (P["rotation_euler"][2], P["sample_x"] * 1000, P["sample_y"] * 1000); p = P["sample_z"] * 1000
dl = (math.radians(7.0), 1.3, -1.1)                       # inside the training range (+-8 deg, +-1.5 mm)
rgb = np.array(Image.open(t).convert("RGB")); d0 = B.gt_depth_mm(obj, *q, p); d1 = B.gt_depth_mm(obj, q[0] + dl[0], q[1] + dl[1], q[2] + dl[2], p)
save_rgb(rgb, "reg_real"); overlay(rgb, [(d0, GREEN)], "reg_overlay_gt"); overlay(rgb, [(d1, ORANGE)], "reg_overlay_perturbed")
depth_png(d0, "reg_depth"); normal_png(d0, "reg_normal"); depth_png(d1, "reg_depth_perturbed")
print("registered", fr, "q* = (%.1f deg, %.2f, %.2f mm), press %.2f mm" % (math.degrees(q[0]), q[1], q[2], p))

# unlabeled contact: frame 0479 with the teacher state and the aligned state of the K=25 aligner
t = f"{G}/real_filtered/{obj}/session_000/sensor_0000/samples/0479.png"
f = json.load(open(f"{G}/real_pseudo_k25_ro_al/rc_fits.json"))[t]; rgb = np.array(Image.open(t).convert("RGB"))
m0 = B.gt_depth_mm(obj, *f["tp"][:3], f["p"]); m1 = B.gt_depth_mm(obj, *f["pose"], f["p"])
save_rgb(rgb, "unl_real"); overlay(rgb, [(m0, RED)], "unl_overlay_teacher"); overlay(rgb, [(m1, GREEN)], "unl_overlay_aligned")
depth_png(m1, "unl_depth"); normal_png(m1, "unl_normal")
print("written to", OUT, sorted(os.listdir(OUT)))
