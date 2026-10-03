"""Fit the planar pose of an object mesh to a depth map by global search (no pose convention needed).

Used for sensor B, whose public depth labels come without a mesh-registered pose and are a slightly smoothed version of
the mesh depth: `label_depth` = Gaussian blur (sigma 2 px) of the mesh depth clipped at the 1 mm indentation.
  fit(obj, target_mm)  ->  (theta, x_mm, y_mm, p_mm), residual      target = a labeled depth map or a teacher prediction

usage: python scripts/tactile_render/fit_mesh_pose.py --frames labeled|pseudo --k 25 --out fits.json [--src-root pseudo root] [--limit N]
"""
import argparse
import glob
import json
import math
import os
import os.path as osp
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.join(ROOT, "scripts")); sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import build_real_filtered_new as B  # noqa: E402

REAL_B = "/media/hdd2/ihsuan/gs_blender/real_gelslim40_train"
VIEW_B = 0.012
_zt = {}


def mesh_depth(obj, pose, dev, stride=1):
    """pose [N, 4] = (theta, x_mm, y_mm, p_mm) -> mesh depth (mm) [N, 224/stride, 224/stride]."""
    if obj not in _zt:
        zmap, LO, BB, PPM, RS = B.get_zmap(obj)
        _zt[obj] = (torch.from_numpy(zmap).to(dev), LO, BB, PPM, RS, float(zmap.min()))
    z, LO, BB, PPM, RS, zmin = _zt[obj]
    u = torch.arange(0, 224, stride, device=dev) + 0.5 * stride
    vv, uu = torch.meshgrid(u, u, indexing="ij")
    dx = (-(uu - 112) * B.VIEW / 224)[None] - (pose[:, 1] * 1e-3)[:, None, None]
    dy = (-(vv - 112) * B.VIEW / 224)[None] + (pose[:, 2] * 1e-3)[:, None, None]
    c, s = torch.cos(pose[:, 0])[:, None, None], torch.sin(pose[:, 0])[:, None, None]
    jx = (((c * dx + s * dy) * 1000 + BB[0] - LO[0]) * PPM).long().clamp(0, RS - 1)
    jy = (((-s * dx + c * dy) * 1000 + BB[1] - LO[1]) * PPM).long().clamp(0, RS - 1)
    return (zmin + pose[:, 3][:, None, None] - z[jy, jx]).clamp(min=0)


def label_depth(d_mm, stride=1):
    """Mesh depth -> the dataset's label style (clip at 1 mm, Gaussian blur of sigma 2 px at full resolution)."""
    sg = 2.0 / stride
    return TF.gaussian_blur(d_mm.clamp(max=1.0)[:, None], kernel_size=max(3, int(6 * sg) | 1), sigma=sg)[:, 0]


@torch.no_grad()
def fit(obj, target_mm, dev):
    g = torch.from_numpy(target_mm).to(dev); g4 = g[2::4, 2::4]

    def search(cands, stride):
        tgt = g4 if stride == 4 else g
        best = (1e9, None)
        for i in range(0, len(cands), 4096 if stride == 4 else 256):
            c = cands[i:i + (4096 if stride == 4 else 256)]
            e = (label_depth(mesh_depth(obj, c, dev, stride), stride) - tgt[None]).abs().mean((1, 2)); j = int(e.argmin())
            if float(e[j]) < best[0]:
                best = (float(e[j]), c[j].clone())
        return best
    c = torch.cartesian_prod(torch.deg2rad(torch.arange(0, 360, 3.0)), torch.arange(-7, 7.01, 0.5), torch.arange(-7, 7.01, 0.5), torch.tensor([1.0])).to(dev)
    e, b = search(c, 4)
    for st, sx, ps in ((1.0, 0.2, (-0.1, 0.0, 0.1)), (0.25, 0.05, (-0.05, 0.0, 0.05))):
        b = b.cpu()
        c = torch.cartesian_prod(b[0] + torch.deg2rad(torch.arange(-3, 3.01, 1.0) * st), b[1] + torch.arange(-3, 3.01, 1.0) * sx,
                                 b[2] + torch.arange(-3, 3.01, 1.0) * sx, b[3] + torch.tensor(ps)).to(dev)
        e, b = search(c, 1)
    return b.tolist(), e


def load_depth_mm(path):
    d = np.abs(np.load(path).astype(np.float32))
    return d * (1000 if d.max() < 0.1 else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", choices=["labeled", "pseudo", "all"], required=True)
    ap.add_argument("--root", default=REAL_B, help="root whose frames are fitted (--frames all: every frame with a _gt.npy)")
    ap.add_argument("--objects", nargs="*", default=None, help="--frames all: restrict to these objects")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--src-root", default=None, help="pseudo root with the teacher depth (for --frames pseudo)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    B.VIEW = VIEW_B
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    from train_tactile_gan import RealPairs
    if args.frames == "labeled":
        items = [(tac, dep) for dep, tac in RealPairs(root=REAL_B, split="train", per_object=args.k).items]
    elif args.frames == "all":
        items = []
        for dep in sorted(glob.glob(osp.join(args.root, "*", "session_*", "sensor_0000", "raw_data", "*_gt.npy"))):
            if args.objects and dep.split("/")[-5] not in args.objects:
                continue
            items.append((osp.join(osp.dirname(osp.dirname(dep)), "samples", osp.basename(dep)[:4] + ".png"), dep))
    else:
        man = json.load(open(osp.join(args.src_root, "pseudo_manifest.json")))["manifest"]
        fr = [t for t, k, _ in man if k == "pseudo"]
        if args.limit:
            fr = fr[::max(1, len(fr) // args.limit)]
        items = [(t, osp.join(osp.dirname(osp.dirname(t)).replace(REAL_B, args.src_root), "raw_data", osp.basename(t)[:4] + "_gt.npy")) for t in fr]
    out = {}
    for n, (tac, dep) in enumerate(items):
        pose, e = fit(tac.split("/")[-5], load_depth_mm(dep), args.device)
        out[tac] = pose + [e]
        if n % 50 == 0:
            print(f"{n}/{len(items)} residual {e:.4f}", flush=True)
    json.dump(out, open(args.out, "w"))
    print("mean residual", float(np.mean([v[4] for v in out.values()])), "->", args.out)


if __name__ == "__main__":
    main()
