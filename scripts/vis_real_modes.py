"""Per-sample three-mode predictions on the REAL val split (same split/preproc/object
ids as training-time eval), for qualitative figures.

Pass 1: run all real-val samples through both/tactile/rgb, write metrics CSV.
Pass 2: re-run the selected candidates and dump inputs, GT, and predictions (npz)
        plus a quick composite PNG per candidate.

Usage:
  python scripts/vis_real_modes.py \
      --train-dir /media/hdd/ihsuan/VisTacFusion_outputs/ratio_g3s_sim348_transfilt_zoom115_crop816 \
      --model ablation/encoder/tac_t3_rgb_mae.yaml --train configs/train_bs32.yaml \
      --data ablation/simqty_gtac/data_ratio_g3s_sim348_transfilt_zoom115_crop816.yaml \
      --device cuda:1 --out eval_results/vis_real_modes_sim348 [--topk 8] [--indices 3,17,42]
"""
from __future__ import annotations

import argparse
import csv
import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from vistacfusion.data.dataset import build_datasets
from vistacfusion.engine.inference import load_model
from vistacfusion.utils.config import merge_configs

CONFIGS = ("both", "tactile", "rgb")
MEAN = np.array([123.675, 116.28, 103.53]) / 255.0
STD = np.array([58.395, 57.12, 57.375]) / 255.0


def denorm(img_t):
    """(3,H,W) normalized tensor -> (H,W,3) uint8."""
    x = img_t.permute(1, 2, 0).cpu().numpy() * STD + MEAN
    return (x.clip(0, 1) * 255).astype(np.uint8)


def rot_err_deg(p, g):
    c = (p[:, 0] * g[:, 0] + p[:, 1] * g[:, 1]).clamp(-1, 1)
    return torch.rad2deg(torch.acos(c))


def half_of(ds, i):
    """Object half-size (m) used to normalise (tx, ty) for sample i."""
    unit = ds.samples[i][0]
    obj_name = osp.basename(osp.dirname(osp.dirname(unit)))
    return float(ds._obj_pose_info[obj_name]["half"])


def sample_name(ds, i):
    for attr in ("samples", "items", "entries", "records", "_samples", "_items"):
        lst = getattr(ds, attr, None)
        if lst is not None and len(lst) == len(ds):
            s = lst[i]
            return str(s if not isinstance(s, (tuple, list, dict)) else s)
    return str(i)


@torch.no_grad()
def run(models, batch, device, cfg_name):
    depth_m, pose_m = models
    rgb, tac = batch["rgb"].to(device), batch["tactile"].to(device)
    obj = batch.get("object")
    obj = obj.to(device) if obj is not None else None
    with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
        od = depth_m(rgb, tac, config=cfg_name, object_ids=obj)
        op = pose_m(rgb, tac, config=cfg_name, object_ids=obj)
    depth = od["depth"].float()
    normal = od["normal"].float()
    normal = normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-8)
    se2 = op["se2"].float()
    return depth, normal, se2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--topk", type=int, default=8,
                    help="auto-pick this many samples by (opaque-only rot err - both rot err)")
    ap.add_argument("--indices", default=None,
                    help="comma-separated val indices to dump instead of auto-pick")
    ap.add_argument("--depth-ckpt", default="best_depth.pt",
                    help="checkpoint (in train-dir) for depth/normal")
    ap.add_argument("--pose-ckpt", default="best_pose.pt",
                    help="checkpoint (in train-dir) for pose; latest.pt reproduces the "
                         "ladder's pose numbers (best_pose.pt is chosen by 1-cos, not deg)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    cfg = merge_configs(args.model, args.train, args.data)

    _, val_ds = build_datasets(cfg)
    print(f"real val: {len(val_ds)} samples")

    depth_m = load_model(cfg, osp.join(args.train_dir, args.depth_ckpt), device)
    pose_m = load_model(cfg, osp.join(args.train_dir, args.pose_ckpt), device)
    models = (depth_m, pose_m)

    # ---------------- pass 1: metrics for every sample ----------------
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    rows = []
    gi = 0
    for batch in loader:
        B = batch["rgb"].shape[0]
        gt_d = batch["depth"].to(device).float()
        gt_n = batch["normal"].to(device).float()
        gt_p = batch["pose"].to(device).float()
        per_cfg = {}
        for c in CONFIGS:
            d, n, p = run(models, batch, device, c)
            gt_nn = gt_n / gt_n.norm(dim=1, keepdim=True).clamp_min(1e-8)
            cosang = (n * gt_nn).sum(1).clamp(-1, 1)            # [B,H,W]
            per_cfg[c] = dict(
                depth_mse=((d - gt_d) ** 2).flatten(1).mean(1).cpu().numpy(),
                normal_mse=((n - gt_n) ** 2).flatten(1).mean(1).cpu().numpy(),
                normal_deg=torch.rad2deg(torch.acos(cosang)).flatten(1).mean(1).cpu().numpy(),
                rot_deg=rot_err_deg(p, gt_p).cpu().numpy(),
                trans_l1=(p[:, 2:] - gt_p[:, 2:]).abs().mean(1).cpu().numpy(),
            )
        objs = batch["object"].numpy()
        for b in range(B):
            name = sample_name(val_ds, gi + b)
            r = {"idx": gi + b, "object": int(objs[b]), "name": name}
            half = half_of(val_ds, gi + b)
            r["half_m"] = half
            for c in CONFIGS:
                for k, v in per_cfg[c].items():
                    r[f"{c}_{k}"] = float(v[b])
                r[f"{c}_trans_mm"] = float(per_cfg[c]["trans_l1"][b]) * half * 1000.0
            rows.append(r)
        gi += B
        print(f"  {gi}/{len(val_ds)}", end="\r")
    print()

    csv_path = osp.join(args.out, "metrics_per_sample.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    # summary per config (sanity check vs Table I)
    for c in CONFIGS:
        dm = np.mean([r[f'{c}_depth_mse'] for r in rows])
        print(f"  [{c:7s}] depth_mse={dm:.4f} depth_rmse_mm={np.sqrt(dm):.3f} "
              f"normal_mse={np.mean([r[f'{c}_normal_mse'] for r in rows]):.4f} "
              f"normal_deg={np.mean([r[f'{c}_normal_deg'] for r in rows]):.2f} "
              f"rot={np.mean([r[f'{c}_rot_deg'] for r in rows]):.2f} "
              f"trans_l1={np.mean([r[f'{c}_trans_l1'] for r in rows]):.4f} "
              f"trans_mm={np.mean([r[f'{c}_trans_mm'] for r in rows]):.3f}")

    # ---------------- pick candidates ----------------
    if args.indices:
        picks = [int(x) for x in args.indices.split(",")]
    else:
        # want: opaque-only pose clearly off, both correct, transparent depth clearly off
        def score(r):
            return (r["tactile_rot_deg"] - r["both_rot_deg"]) \
                - 5.0 * max(0.0, r["both_rot_deg"] - 1.0)
        ranked = sorted(rows, key=score, reverse=True)
        picks = [r["idx"] for r in ranked[: args.topk]]
        print("auto-picked (idx, object, name, both_rot, tac_rot, rgb_dmse/both_dmse):")
        for r in ranked[: args.topk]:
            print(f"  {r['idx']:4d} obj={r['object']:2d} {r['name'][:60]} "
                  f"both={r['both_rot_deg']:.2f} tac={r['tactile_rot_deg']:.2f} "
                  f"dmse rgb/both={r['rgb_depth_mse']/max(r['both_depth_mse'],1e-9):.1f}x")

    # ---------------- pass 2: dump picked samples ----------------
    sub = DataLoader(Subset(val_ds, picks), batch_size=1, shuffle=False)
    for k, batch in zip(picks, sub):
        preds = {c: run(models, batch, device, c) for c in CONFIGS}
        gt_pose = batch["pose"][0].numpy()
        gt_theta = np.degrees(np.arctan2(gt_pose[1], gt_pose[0]))
        out = dict(
            rgb=denorm(batch["rgb"][0]), tactile=denorm(batch["tactile"][0]),
            gt_depth=batch["depth"][0, 0].numpy(),
            gt_normal=batch["normal"][0].permute(1, 2, 0).numpy(),
            gt_pose=gt_pose, object=int(batch["object"][0]),
        )
        for c in CONFIGS:
            d, n, p = preds[c]
            out[f"{c}_depth"] = d[0, 0].cpu().numpy()
            out[f"{c}_normal"] = n[0].permute(1, 2, 0).cpu().numpy()
            out[f"{c}_pose"] = p[0].cpu().numpy()
        np.savez_compressed(osp.join(args.out, f"sample_{k:04d}.npz"), **out)

        # quick composite: rows = transparent / opaque / both / GT ; cols = input, depth, normal, pose
        fig, ax = plt.subplots(4, 4, figsize=(10, 10))
        order = [("rgb", "Transparent only", out["rgb"]),
                 ("tactile", "Opaque only", out["tactile"]),
                 ("both", "Both", out["tactile"])]
        vmax = float(out["gt_depth"].max()) if out["gt_depth"].max() > 0 else None
        for r_, (c, label, inp) in enumerate(order):
            ax[r_, 0].imshow(inp); ax[r_, 0].set_ylabel(label, fontweight="bold")
            ax[r_, 1].imshow(out[f"{c}_depth"], cmap="viridis", vmin=0, vmax=vmax)
            ax[r_, 2].imshow((out[f"{c}_normal"] * 0.5 + 0.5).clip(0, 1))
            p = out[f"{c}_pose"]; th = np.degrees(np.arctan2(p[1], p[0]))
            ax[r_, 3].text(0.5, 0.5, f"θ={th:.1f}°\ntx={p[2]:.3f}\nty={p[3]:.3f}",
                           ha="center", va="center", fontsize=14, family="monospace",
                           transform=ax[r_, 3].transAxes)
        ax[3, 0].imshow(out["tactile"]); ax[3, 0].set_ylabel("GT", fontweight="bold")
        ax[3, 1].imshow(out["gt_depth"], cmap="viridis", vmin=0, vmax=vmax)
        ax[3, 2].imshow((out["gt_normal"] * 0.5 + 0.5).clip(0, 1))
        ax[3, 3].text(0.5, 0.5, f"θ={gt_theta:.1f}°\ntx={gt_pose[2]:.3f}\nty={gt_pose[3]:.3f}",
                      ha="center", va="center", fontsize=14, family="monospace", color="green",
                      transform=ax[3, 3].transAxes)
        for a in ax.flat:
            a.set_xticks([]); a.set_yticks([])
        for a in ax[:, 3]:
            a.axis("off")
        fig.suptitle(f"val idx {k} | object {out['object']} | {sample_name(val_ds, k)[:80]}", fontsize=9)
        fig.tight_layout()
        fig.savefig(osp.join(args.out, f"sample_{k:04d}.png"), dpi=110)
        plt.close(fig)
        print(f"  dumped sample {k}")

    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
