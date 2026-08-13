"""Precompute CORAL statistics for the DPT multiscale tap layers.

Same data/protocol as compute_coral_stats.py, but statistics are computed at
each encoder tap layer used by the DPT head (forward_multiscale), for both
branches (tactile/T3 taps [3,7,10,14]; rgb/MAE taps [5,11,17,23] — rgb taps
feed DPT in the rgb-only modality config). Patch tokens only (taps have no CLS).

Output: {branch: {"tap{i}": {patch_mu_s, patch_mu_r, patch_A,
                             obj_patch_mu_s, obj_patch_mu_r},
                  "tap_layers": [...]},
         "objects": [...], "obj_has_real": ...}

Usage:
    python scripts/compute_coral_tap_stats.py --out pretrained_encoders/coral_stats_dpt.pt
"""
import argparse
import json
import os.path as osp
import sys

import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from compute_coral_stats import (WINDOWS, StatAcc, collect_paths, mat_pow,
                                 preprocess)
from vistacfusion.models.encoders import MAEEncoder, T3Encoder


@torch.no_grad()
def tap_stats(enc, paths_by_obj, subdir, device, batch=32):
    D = enc.embed_dim
    layers = enc.multiscale_layers
    n_taps = len(layers)
    acc = {(t, dom): StatAcc(D) for t in range(n_taps) for dom in ("sim", "real")}
    obj_mean = {}   # (obj, dom, tap) -> [sum, n]

    for obj, entry in paths_by_obj.items():
        for dom in ("sim", "real"):
            if entry[dom] is None:
                continue
            paths = entry[dom][subdir]
            for i in range(0, len(paths), batch):
                x = torch.stack([preprocess(p) for p in paths[i:i + batch]]).to(device)
                taps = enc.forward_multiscale(x)          # n_taps × [B, N, D]
                for t, tap in enumerate(taps):
                    pt = tap.reshape(-1, D).float()
                    acc[(t, dom)].add(pt)
                    key = (obj, dom, t)
                    if key not in obj_mean:
                        obj_mean[key] = [torch.zeros(D, dtype=torch.float64), 0]
                    obj_mean[key][0] += pt.double().cpu().sum(0)
                    obj_mean[key][1] += pt.shape[0]
        print(f"    {obj}: sim={'y' if entry['sim'] else '-'} "
              f"real={'y' if entry['real'] else '-'}", flush=True)

    objects = sorted(paths_by_obj.keys())
    out = {"tap_layers": list(layers)}
    for t in range(n_taps):
        mu_s = acc[(t, "sim")].mean()
        mu_r = acc[(t, "real")].mean()
        A = mat_pow(acc[(t, "sim")].cov(), -0.5) @ mat_pow(acc[(t, "real")].cov(), 0.5)
        oms = torch.zeros(len(objects), D)
        omr = torch.zeros(len(objects), D)
        for i, obj in enumerate(objects):
            ks, kr = (obj, "sim", t), (obj, "real", t)
            oms[i] = (obj_mean[ks][0] / obj_mean[ks][1]).float() if ks in obj_mean else mu_s.float()
            omr[i] = (obj_mean[kr][0] / obj_mean[kr][1]).float() if kr in obj_mean else mu_r.float()
        out[f"tap{t}"] = {
            "patch_mu_s": mu_s.float(), "patch_mu_r": mu_r.float(),
            "patch_A": A.float(),
            "obj_patch_mu_s": oms, "obj_patch_mu_r": omr,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pretrained_encoders/coral_stats_dpt.pt")
    ap.add_argument("--device", default="cuda:3")
    args = ap.parse_args()

    cache = "/tmp/coral_paths_cache.json"
    if osp.exists(cache):
        paths = json.load(open(cache))
        print(f"loaded path cache: {cache}")
    else:
        windows = json.load(open(WINDOWS))
        paths = collect_paths(windows)
        with open(cache, "w") as f:
            json.dump(paths, f)
    objects = sorted(paths.keys())
    stats = {"objects": objects,
             "obj_has_real": torch.tensor(
                 [paths[o]["real"] is not None for o in objects])}

    print("  [tactile / T3 taps]")
    t3 = T3Encoder("pretrained_encoders/t3_large").to(args.device).eval()
    stats["tactile"] = tap_stats(t3, paths, "samples", args.device)
    del t3
    torch.cuda.empty_cache()

    print("  [rgb / MAE taps]")
    mae = MAEEncoder("pretrained_encoders/mae_vitl16.pth").to(args.device).eval()
    stats["rgb"] = tap_stats(mae, paths, "rgb", args.device)

    torch.save(stats, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
