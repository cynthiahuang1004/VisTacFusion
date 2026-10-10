"""Overlay assets for the revised intro figure (reference mock + PNGs to paste into the PPT).
usage (cwd = VisTacFusion): conda run -n vistacfusion python paper_RoboSoft/scripts/fig1_assets.py
Writes paper_RoboSoft/figures/intro_figure/v2_assets/:
  c_overlay_teacher.png / c_overlay_aligned.png   real image 0479 (pattern_06) + mesh contact outline at the teacher / aligned state
  a_overlay_rawpose.png / a_overlay_verified.png  a second-session image + mesh outline at the first-session calibration / verified pose
  *_depth.png                                     the corresponding viridis depth maps"""
import glob, json, os, sys
import cv2, numpy as np
from PIL import Image
from matplotlib import cm
sys.path.insert(0, "scripts")
import build_real_filtered_new as B
B.VIEW = 0.017502
G = "/media/hdd2/ihsuan/gs_blender"; OUT = "paper_RoboSoft/figures/intro_figure/v2_assets"; os.makedirs(OUT, exist_ok=True)
RED, GREEN = (235, 60, 50), (40, 220, 90); UP = 3

def overlay(rgb, depth_mm, color, name):
    im = cv2.resize(rgb, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
    m = cv2.resize((depth_mm > 0.05).astype(np.uint8), None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, cs, -1, (255, 255, 255), 13, cv2.LINE_AA); cv2.drawContours(im, cs, -1, color, 8, cv2.LINE_AA)
    Image.fromarray(im).save(f"{OUT}/{name}.png")
def depth_png(d, name):
    Image.fromarray((cm.viridis(np.clip(d / 1.2, 0, 1))[..., :3] * 255).astype(np.uint8)).resize((448, 448), Image.BILINEAR).save(f"{OUT}/{name}.png")

# (c) the figure's main contact: sensor A, session 1, pattern_06 frame 0479
obj = "pattern_06_5_lines_angle_1"; t = f"{G}/real_filtered/{obj}/session_000/sensor_0000/samples/0479.png"
f = json.load(open(f"{G}/real_pseudo_k25_ro_al/rc_fits.json"))[t]; rgb = np.array(Image.open(t).convert("RGB"))
m0 = B.gt_depth_mm(obj, *f["tp"][:3], f["p"]); m1 = B.gt_depth_mm(obj, *f["pose"], f["p"])
overlay(rgb, m0, RED, "c_overlay_teacher"); overlay(rgb, m1, GREEN, "c_overlay_aligned"); depth_png(m0, "c_teacher_depth"); depth_png(m1, "c_aligned_depth")
Image.fromarray(rgb).resize((448, 448), Image.BICUBIC).save(f"{OUT}/c_real.png")

# (a) second-session example of the same object: mesh at the first-session calibration vs. the verified pose
gt = {}
for pj in glob.glob(f"{G}/real_s2_poolcheck/{obj}/session_*/sensor_0000/raw_data/*_pose.json"):
    gt[(pj.split("/")[-4], int(json.load(open(pj))["source"]["frame"]))] = pj
best = None
for pj in sorted(glob.glob(f"{G}/real_s2_pool/{obj}/session_*/sensor_0000/raw_data/*_pose.json")):
    key = (pj.split("/")[-4], int(json.load(open(pj))["source"]["frame"]))
    if key not in gt: continue
    raw = np.load(pj.replace("_pose.json", "_gt.npy")) * 1000; ver = np.load(gt[key].replace("_pose.json", "_gt.npy")) * 1000
    a, b = raw > 0.05, ver > 0.05; iou = (a & b).sum() / max(1, (a | b).sum())
    if b.sum() > 3000 and (best is None or abs(iou - 0.25) < abs(best[0] - 0.25)):      # a typical, clearly visible offset
        best = (iou, pj, gt[key], raw, ver)
iou, pj, gpj, raw, ver = best
rgb2 = np.array(Image.open(pj.replace("raw_data", "samples").replace("_pose.json", ".png")).convert("RGB"))
overlay(rgb2, raw, RED, "a_overlay_rawpose"); overlay(rgb2, ver, GREEN, "a_overlay_verified"); depth_png(ver, "a_verified_depth")
print("session-2 example:", pj, "IoU(raw pose, verified) = %.2f" % iou)
print("written to", OUT, sorted(os.listdir(OUT)))
