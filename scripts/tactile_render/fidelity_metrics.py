"""Which renderer-fidelity metric predicts downstream error?

For every learned renderer (G_final.pt) render the real val depth maps and compare with the real
images using: full-image L1, contact-region L1, PSNR, SSIM, LPIPS (AlexNet), and the same restricted to
the contact region (contact-masked SSIM / LPIPS via a dilated bounding box crop).  Then Spearman-
correlate each metric with the downstream depth MSE of the run trained with that renderer.

usage: python scripts/tactile_render/fidelity_metrics.py --device cuda:2
"""
import argparse
import json
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from train_tactile_gan import REAL_ROOT, RealPairs, UNetG  # noqa: E402

OUT = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"
# tag -> (generator dir, downstream run whose synthetic data came from it)
GENS = {
    "k10": ("tactile_gan_k10", "B_k10_gank"), "k25": ("tactile_gan_k25", "B_k25_gank"),
    "k50": ("tactile_gan_k50", "B_k50_gank"), "k100": ("tactile_gan_k100", "B_k100_gank"),
    "hyb_k10": ("tactile_gan_hyb_k10", "B_k10_hyb"), "hyb_k25": ("tactile_gan_hyb_k25", "B_k25_hyb"),
    "hyb_k100": ("tactile_gan_hyb_k100", "B_k100_hyb"),
    "bgcond_k25": ("tactile_gan_bgcond_k25", "B_k25_bgcond"), "bgcond_k100": ("tactile_gan_bgcond_k100", "B_k100_bgcond"),
    "full_k10": ("tactile_gan", "B_k10_ganfull"), "full_k25": ("tactile_gan", "B_k25_ganfull"),
    "full_k100": ("tactile_gan", "B_k100_ganfull"),
}


def downstream(run):
    for pre in ("pilot150_", "pilot_"):
        p = f"{OUT}/{pre}{run}/history.json"
        if osp.exists(p):
            h = json.load(open(p))
            if len(h) >= 50 or pre == "pilot_":
                g = lambda e: next(iter(e["val"].values()))["depth_mse"]
                return min(g(e) for e in h)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", default="ablation/pilot_robosoft/fidelity_metrics.json")
    a = ap.parse_args()
    import lpips
    lp = lpips.LPIPS(net="alex").to(a.device).eval()
    res = {}
    for tag, (gdir, run) in GENS.items():
        RealPairs.bg_cond = "bgcond" in tag
        va = RealPairs(split="val")
        G = UNetG(in_ch=6 if RealPairs.bg_cond else 3).to(a.device).eval()
        G.load_state_dict(torch.load(f"outputs/{gdir}/G_final.pt", map_location=a.device))
        acc = dict(l1=[], cl1=[], psnr=[], ssim=[], lpips=[], c_ssim=[], c_lpips=[], c_psnr=[])
        with torch.no_grad():
            for x, t in DataLoader(va, batch_size=32, num_workers=4):
                x, t = x.to(a.device), t.to(a.device)
                p = G(x)
                m = (x[:, :1] > -1 + 1e-6)                                   # contact mask [B,1,H,W]
                e = (p - t).abs().mean(1, keepdim=True) / 2
                acc["l1"] += e.mean((1, 2, 3)).tolist()
                acc["cl1"] += [(e[i][m[i]].mean().item() if m[i].any() else np.nan) for i in range(len(e))]
                mse = ((p - t) ** 2).mean((1, 2, 3)) / 4
                acc["psnr"] += (10 * torch.log10(1 / mse.clamp(min=1e-10))).tolist()
                acc["lpips"] += lp(p, t).flatten().tolist()
                pn, tn = ((p + 1) / 2).cpu().numpy(), ((t + 1) / 2).cpu().numpy()
                for i in range(len(p)):
                    acc["ssim"].append(ssim_fn(pn[i].transpose(1, 2, 0), tn[i].transpose(1, 2, 0), channel_axis=2, data_range=1.0))
                    ys, xs = torch.where(m[i, 0])
                    if len(ys) == 0:
                        acc["c_ssim"].append(np.nan); acc["c_lpips"].append(np.nan); acc["c_psnr"].append(np.nan); continue
                    y0, y1 = max(0, ys.min().item() - 8), min(224, ys.max().item() + 9)
                    x0, x1 = max(0, xs.min().item() - 8), min(224, xs.max().item() + 9)
                    if y1 - y0 < 16 or x1 - x0 < 16:
                        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2; y0, y1 = max(0, cy - 8), min(224, cy + 8); x0, x1 = max(0, cx - 8), min(224, cx + 8)
                    pc, tc = pn[i][:, y0:y1, x0:x1], tn[i][:, y0:y1, x0:x1]
                    acc["c_ssim"].append(ssim_fn(pc.transpose(1, 2, 0), tc.transpose(1, 2, 0), channel_axis=2, data_range=1.0, win_size=7))
                    cm = ((pc - tc) ** 2).mean(); acc["c_psnr"].append(float(10 * np.log10(1 / max(cm, 1e-10))))
                    pcl = F.interpolate(p[i:i + 1, :, y0:y1, x0:x1], size=(64, 64), mode="bilinear", align_corners=False)
                    tcl = F.interpolate(t[i:i + 1, :, y0:y1, x0:x1], size=(64, 64), mode="bilinear", align_corners=False)
                    acc["c_lpips"].append(lp(pcl, tcl).item())
        r = {k: float(np.nanmean(v)) for k, v in acc.items()}
        r["downstream_depth_mse"] = downstream(run); r["run"] = run
        res[tag] = r
        print(f"{tag:12s} " + " ".join(f"{k}={r[k]:.4f}" for k in ("l1", "cl1", "psnr", "ssim", "lpips", "c_psnr", "c_ssim", "c_lpips")) + f" -> depth {r['downstream_depth_mse']}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("\nSpearman correlation with downstream depth MSE (n=%d):" % len(res))
    y = [r["downstream_depth_mse"] for r in res.values()]
    for k in ("l1", "cl1", "psnr", "ssim", "lpips", "c_psnr", "c_ssim", "c_lpips"):
        x = [r[k] for r in res.values()]
        print(f"  {k:8s} rho = {spearmanr(x, y).correlation:+.3f}")


if __name__ == "__main__":
    main()
