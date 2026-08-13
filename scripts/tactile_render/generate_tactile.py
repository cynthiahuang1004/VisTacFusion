"""Re-render sim tactile images with the real-sensor renderer G.

Sweeps sim depth GT (raw_data/*_gt.npy) and writes G(depth) into a sibling
subdir samples_g/ next to the original Blender renders (samples/), mirroring
the rgb_tuned convention. Random per-image depth scaling (default ±15%)
mimics the real press-depth variability.

Usage:
    python scripts/tactile_render/generate_tactile.py \
        --ckpt outputs/tactile_gan/G_final.pt --device cuda:1
    # preview mode: only N images per object, into samples_g_preview/
    python scripts/tactile_render/generate_tactile.py --preview 3
"""
import argparse
import glob
import hashlib
import os
import os.path as osp
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from train_tactile_gan import DEPTH_SCALE, UNetG, with_coords

SIM_ROOT = "/media/hdd2/ihsuan/gs_blender/renders_v3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/tactile_gan/G_final.pt")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--depth-jitter", type=float, default=0.15,
                    help="uniform random depth scale ±this (0 disables)")
    ap.add_argument("--preview", type=int, default=0,
                    help="if >0: only N imgs/object into samples_g_preview/")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--objects", nargs="*", default=None,
                    help="restrict to these object names")
    args = ap.parse_args()

    dev = args.device
    G = UNetG().to(dev).eval()
    G.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    outdir_name = "samples_g_preview" if args.preview else "samples_g"

    deps = sorted(glob.glob(f"{SIM_ROOT}/*/session_*/sensor_0000/raw_data/*_gt.npy"))
    if args.objects:
        keep = set(args.objects)
        deps = [d for d in deps if d.split(SIM_ROOT + "/")[1].split("/")[0] in keep]
    if args.preview:
        per_obj = {}
        keep = []
        for d in deps:
            obj = d.split(SIM_ROOT + "/")[1].split("/")[0]
            if per_obj.get(obj, 0) < args.preview:
                per_obj[obj] = per_obj.get(obj, 0) + 1
                keep.append(d)
        deps = keep
    print(f"{len(deps)} depth maps -> {outdir_name}/", flush=True)

    todo = []
    for d in deps:
        unit = osp.dirname(osp.dirname(d))
        idx = osp.basename(d).replace("_gt.npy", "")
        out = osp.join(unit, outdir_name, f"{idx}.png")
        if not osp.exists(out):
            todo.append((d, out))
    print(f"{len(todo)} to generate (rest exist)", flush=True)

    with torch.no_grad():
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            xs = []
            for dp, _ in chunk:
                d = np.load(dp).astype(np.float32)
                if args.depth_jitter > 0:
                    seed = int(hashlib.md5(dp.encode()).hexdigest()[:8], 16)
                    rng = np.random.RandomState(seed)
                    d = d * (1 + rng.uniform(-args.depth_jitter, args.depth_jitter))
                xs.append(with_coords(torch.from_numpy(d / DEPTH_SCALE * 2 - 1)[None]))
            x = torch.stack(xs).to(dev)
            y = G(x)
            y = ((y.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
            for (dp, out), img in zip(chunk, y):
                os.makedirs(osp.dirname(out), exist_ok=True)
                Image.fromarray(img.transpose(1, 2, 0)).save(out)
            if (i // args.batch) % 20 == 0:
                print(f"  {i + len(chunk)}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
