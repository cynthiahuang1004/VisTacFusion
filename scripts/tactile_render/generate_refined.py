"""Refine every Blender render of the pattern objects with a trained refiner (train_refiner.py) and write the
result into a sibling subdir (e.g. samples_r_loo/) so the downstream data configs can point at it via
sim.tactile_subdir.  Depth / normal / pose labels are untouched (geometry is preserved by construction).

usage: python scripts/tactile_render/generate_refined.py --ckpt outputs/tactile_refiner_loo/G_final.pt \
           --out-subdir samples_r_loo --device cuda:0 [--sim-root .../renders_v3_dr]
"""
import argparse
import glob
import os
import os.path as osp
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from train_refiner import OBJECTS, RefinerG, contact_mask, load_img, refine, session_bg  # noqa: E402
from train_tactile_gan import SIM_ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-subdir", required=True)
    ap.add_argument("--sim-root", default=SIM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dilate", type=int, default=4)
    ap.add_argument("--feather", type=int, default=2)
    ap.add_argument("--objects", nargs="*", default=None)
    args = ap.parse_args()
    dev = args.device
    G = RefinerG().to(dev).eval()
    G.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=True))

    todo = []
    for obj in (args.objects or OBJECTS):
        for tac in sorted(glob.glob(f"{args.sim_root}/{obj}/session_*/sensor_0000/samples/*.png")):
            unit = osp.dirname(osp.dirname(tac))
            idx = osp.splitext(osp.basename(tac))[0]
            dep = osp.join(unit, "raw_data", f"{idx}_gt.npy")
            out = osp.join(unit, args.out_subdir, f"{idx}.png")
            if osp.exists(dep) and not osp.exists(out):
                todo.append((obj, tac, dep, out))
    print(f"{len(todo)} renders to refine -> {args.out_subdir}/", flush=True)

    from concurrent.futures import ThreadPoolExecutor

    def prep(item):
        obj, tac, dep, _ = item
        d = torch.from_numpy(np.load(dep).astype(np.float32))[None]
        return load_img(tac), session_bg(obj), contact_mask(d, args.dilate, args.feather)

    def save(a):
        out, img = a
        os.makedirs(osp.dirname(out), exist_ok=True)
        Image.fromarray(img.transpose(1, 2, 0)).save(out)

    pool, spool = ThreadPoolExecutor(8), ThreadPoolExecutor(6)
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    pending = pool.map(prep, batches[0]) if batches else None
    with torch.no_grad():
        for bi, chunk in enumerate(batches):
            xs = list(pending)
            if bi + 1 < len(batches):
                pending = pool.map(prep, batches[bi + 1])
            b, bg, m = (torch.stack([x[k] for x in xs]).to(dev) for k in range(3))
            y, _ = refine(G, b, bg, m)
            y = ((y.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
            list(spool.map(save, [(out, img) for (_, _, _, out), img in zip(chunk, y)]))
            if bi % 20 == 0:
                print(f"  {(bi + 1) * args.batch}/{len(todo)}", flush=True)
    pool.shutdown(); spool.shutdown()
    print("done", flush=True)


if __name__ == "__main__":
    main()
