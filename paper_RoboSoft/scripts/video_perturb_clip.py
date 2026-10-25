"""Aligner training-data clip: one labeled real frame with the mesh outline at its GT state (green) and at randomly
perturbed states q* + delta (orange), delta drawn from the aligner's training ranges (+-8 deg, +-1.5 mm, +-0.15 mm),
with the correction target -delta printed. No GPU.  usage (cwd = VisTacFusion): python paper_RoboSoft/scripts/video_perturb_clip.py
writes paper_RoboSoft/video/perturb/perturb.{mp4,gif}"""
import json, math, os, os.path as osp, subprocess, sys
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))); sys.path.insert(0, osp.join(ROOT, "scripts"))
import build_real_filtered_new as B  # noqa: E402
B.VIEW = 0.017502
G = "/media/hdd2/ihsuan/gs_blender/real_filtered"; OBJ = "pattern_06_5_lines_angle_1"; FR = "0064"; UP = 3
VID = osp.join(ROOT, "paper_RoboSoft", "video", "perturb"); FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def outline(im, d, color, width):
    cs, _ = cv2.findContours((d > 0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE); cs = [(c * UP).astype(np.int32) for c in cs]
    cv2.drawContours(im, cs, -1, (0, 0, 0), width + 2); cv2.drawContours(im, cs, -1, color, width); return im


def main():
    os.makedirs(VID, exist_ok=True); u = f"{G}/{OBJ}/session_000/sensor_0000"
    rgb = np.array(Image.open(f"{u}/samples/{FR}.png").convert("RGB")); P = json.load(open(f"{u}/raw_data/{FR}_pose.json"))
    q = [P["rotation_euler"][2], P["sample_x"] * 1e3, P["sample_y"] * 1e3, float(np.clip(np.load(f"{u}/raw_data/{FR}_gt.npy").max() * 1e3, 0.3, 1.6))]
    dg = B.gt_depth_mm(OBJ, *q); rng = np.random.RandomState(3); frames = []; f = ImageFont.truetype(FONT, 30); f2 = ImageFont.truetype(FONT, 26)
    base = cv2.resize(rgb, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
    for k in range(14):
        d = rng.uniform(-1, 1, 4) * np.array([math.radians(8), 1.5, 1.5, 0.15]) * (0.3 if k % 2 else 1.0)
        im = outline(base.copy(), dg, (60, 230, 90), 3); im = outline(im, B.gt_depth_mm(OBJ, q[0] + d[0], q[1] + d[1], q[2] + d[2], q[3] + d[3]), (255, 150, 0), 3)
        pil = Image.new("RGB", (im.shape[1] + 800, im.shape[0]), (0, 0, 0)); pil.paste(Image.fromarray(im), (0, 0)); dr = ImageDraw.Draw(pil); x0 = im.shape[1] + 24
        dr.text((x0, 30), "labeled real image", fill=(255, 255, 255), font=f)
        dr.text((x0, 90), "mesh at GT state q*", fill=(60, 230, 90), font=f2)
        dr.text((x0, 130), "mesh at q* + δ", fill=(255, 150, 0), font=f2)
        dr.text((x0, 220), "sampled perturbation δ", fill=(255, 255, 255), font=f2)
        dr.text((x0, 260), f"rotation  {math.degrees(d[0]):+5.1f}°", fill=(255, 150, 0), font=f2)
        dr.text((x0, 296), f"x, y      {d[1]:+4.2f}, {d[2]:+4.2f} mm", fill=(255, 150, 0), font=f2)
        dr.text((x0, 332), f"indent.   {d[3]:+5.2f} mm", fill=(255, 150, 0), font=f2)
        dr.text((x0, 420), "aligner input: image + mesh depth at q* + δ", fill=(255, 255, 255), font=f2)
        dr.text((x0, 456), "training target:  −δ", fill=(60, 230, 90), font=f2)
        dr.text((x0, 540), "one labeled contact → many training pairs", fill=(180, 180, 180), font=f2)
        frames.append(np.array(pil))
    for i, fr in enumerate(frames): Image.fromarray(fr).save(osp.join(VID, f"{i:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "2", "-i", osp.join(VID, "%04d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-r", "30", osp.join(VID, "perturb.mp4")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "2", "-i", osp.join(VID, "%04d.png"), "-vf", "fps=2,scale=1100:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse", osp.join(VID, "perturb.gif")], check=True)
    print("wrote", VID)


if __name__ == "__main__":
    main()
