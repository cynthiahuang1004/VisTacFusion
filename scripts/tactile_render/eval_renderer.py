"""Renderer fidelity on REAL val pairs: G(depth_real) vs the real tactile image.

Reports, per generator: L1 and gradient-L1 over the full image and inside the GT
contact mask (images in [0, 1]). --objects restricts to given real objects (e.g. the
leave-object-out held-out set).

usage: python scripts/tactile_render/eval_renderer.py --device cuda:0 [--objects ...] tag=dir ...
"""
import argparse
import os.path as osp
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from train_tactile_gan import REAL_ROOT, RealPairs, UNetG  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gens", nargs="+", help="tag=dir_with_G_final.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--objects", nargs="*", default=None)
    args = ap.parse_args()
    va = RealPairs(split="val")
    if args.objects:
        keep = set(args.objects)
        va.items = [it for it in va.items if it[1].split("real_filtered/")[1].split("/")[0] in keep]
    dl = DataLoader(va, batch_size=64, num_workers=4)
    print(f"{len(va)} real val pairs" + (f" ({args.objects})" if args.objects else ""))
    print("(diff_* generators are scored on the difference image = same absolute-image error once the session bg is added back)")
    print(f"{'generator':16s} {'L1':>7s} {'gradL1':>7s} {'L1_contact':>10s} {'gradL1_contact':>14s}")
    for g in args.gens:
        tag, d = g.split("=")
        diff = tag.startswith("diff")          # diff generators predict tactile - session bg
        RealPairs.diff = diff
        RealPairs.bg_cond = "bgcond" in tag
        RealPairs.phys_cond = "phys" in tag        # physics-guided: +3 ch Blender render at the same pose
        UNetG.residual = "physres" in tag
        va.items = [it for it in va.items if not RealPairs.phys_cond or osp.exists(it[1].replace(REAL_ROOT, RealPairs.phys_root))]
        dl = DataLoader(va, batch_size=64, num_workers=4)
        G = UNetG(in_ch=3 + 3 * RealPairs.bg_cond + 3 * RealPairs.phys_cond).to(args.device).eval()
        G.load_state_dict(torch.load(osp.join(d, "G_final.pt"), map_location=args.device))
        s = dict(l1=0.0, g=0.0, n=0, cl1=0.0, cg=0.0, nc=0)
        for x, t in dl:
            x, t = x.to(args.device), t.to(args.device)
            e = (G(x) - t).abs().mean(1, keepdim=True) / 2          # [-1,1] -> [0,1] units
            p, q = G(x), t
            gy = ((p[..., 1:, :] - p[..., :-1, :]) - (q[..., 1:, :] - q[..., :-1, :])).abs().mean(1, keepdim=True) / 2
            m = x[:, :1] > -1 + 1e-6                                 # depth > 0 (contact)
            s["l1"] += e.sum().item(); s["n"] += e.numel()
            s["g"] += gy.sum().item()
            s["cl1"] += e[m].sum().item(); s["nc"] += int(m.sum())
            s["cg"] += gy[m[..., 1:, :]].sum().item()
        print(f"{tag:16s} {s['l1']/s['n']:7.4f} {s['g']/s['n']:7.4f} {s['cl1']/s['nc']:10.4f} {s['cg']/s['nc']:14.4f}", flush=True)


if __name__ == "__main__":
    main()
