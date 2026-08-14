"""Rotate-then-render: pre-rotated G tactile images with physically correct lighting.

For every sim depth GT, sample a deterministic target angle theta* inside the
object's real rotation window, rotate the DEPTH by phi = theta_session - theta*
(same cv2 convention as transforms.rotate_gel_spin), then render with G. The
illumination therefore always stays in G's learned (= real sensor's) fixed
orientation — unlike rotating pre-rendered images at train time, which spins
the lighting along with the geometry.

Outputs per sensor unit:
    samples_gr/{idx}.png       rendered tactile at angle theta*
    samples_gr/rot_meta.json   {"0000": phi_deg, ...} — the dataset applies the
                               same phi to rgb/depth/normal GT + pose label.

Usage:
    python scripts/tactile_render/generate_tactile_rot.py \
        --ckpt outputs/tactile_gan/G_final.pt --device cuda:2
"""
import argparse
import glob
import hashlib
import json
import math
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from train_tactile_gan import DEPTH_SCALE, UNetG, with_coords

SIM_ROOT = "/media/hdd2/ihsuan/gs_blender/renders_v3"
WINDOWS = "/media/hdd2/ihsuan/VisTacFusion/ablation/simqty_filtered/real_rotation_windows.json"


def unit_seed(path, tag):
    return int(hashlib.md5(f"{path}:{tag}".encode()).hexdigest()[:8], 16)


def rotate_depth(depth, angle_deg):
    """Identical convention to transforms.rotate_gel_spin (positive = CCW)."""
    H, W = depth.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle_deg, 1.0)
    return cv2.warpAffine(depth, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/tactile_gan/G_final.pt")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--depth-jitter", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    with open(WINDOWS) as f:
        windows = json.load(f)

    dev = args.device
    G = UNetG().to(dev).eval()
    G.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))

    units = sorted(glob.glob(f"{SIM_ROOT}/*/session_*/sensor_*"))
    units = [u for u in units
             if u.split(SIM_ROOT + "/")[1].split("/")[0] in windows]
    print(f"{len(units)} sensor units across {len(windows)} shared objects", flush=True)

    todo = []  # (depth_path, out_png, phi_deg)
    for unit in units:
        obj = unit.split(SIM_ROOT + "/")[1].split("/")[0]
        lo, hi = windows[obj]
        with open(osp.join(osp.dirname(unit), "session.json")) as f:
            theta_sess = math.degrees(json.load(f)["base_rotation"][2])
        outdir = osp.join(unit, "samples_gr")
        os.makedirs(outdir, exist_ok=True)
        meta_path = osp.join(outdir, "rot_meta.json")
        meta = {}
        if osp.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        deps = sorted(glob.glob(osp.join(unit, "raw_data", "*_gt.npy")))
        for dp in deps:
            idx = osp.basename(dp).replace("_gt.npy", "")
            u = unit_seed(dp, "theta") / 0xFFFFFFFF
            theta_star = lo + u * (hi - lo)
            phi_deg = theta_sess - theta_star
            meta[idx] = phi_deg
            out = osp.join(outdir, f"{idx}.png")
            if not osp.exists(out):
                todo.append((dp, out, phi_deg))
        with open(meta_path, "w") as f:
            json.dump(meta, f)
    print(f"{len(todo)} images to generate (rest exist)", flush=True)

    from concurrent.futures import ThreadPoolExecutor

    def prep(item):
        dp, _, phi = item
        d = np.load(dp).astype(np.float32)
        d = rotate_depth(d, phi)
        if args.depth_jitter > 0:
            rng = np.random.RandomState(unit_seed(dp, "jitter"))
            d = d * (1 + rng.uniform(-args.depth_jitter, args.depth_jitter))
        return with_coords(torch.from_numpy(d / DEPTH_SCALE * 2 - 1)[None])

    def save(arg):
        out, img = arg
        Image.fromarray(img.transpose(1, 2, 0)).save(out)

    prep_pool = ThreadPoolExecutor(max_workers=8)
    save_pool = ThreadPoolExecutor(max_workers=6)
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    # prefetch: prep next batch on CPU while current renders on GPU
    pending = prep_pool.map(prep, batches[0]) if batches else None
    with torch.no_grad():
        for bi, chunk in enumerate(batches):
            xs = list(pending)
            if bi + 1 < len(batches):
                pending = prep_pool.map(prep, batches[bi + 1])
            x = torch.stack(xs).to(dev)
            y = G(x)
            y = ((y.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
            list(save_pool.map(save, [(out, img) for (_, out, _), img in zip(chunk, y)]))
            if bi % 20 == 0:
                print(f"  {(bi + 1) * args.batch}/{len(todo)}", flush=True)
    prep_pool.shutdown()
    save_pool.shutdown()
    print("done", flush=True)


if __name__ == "__main__":
    main()
