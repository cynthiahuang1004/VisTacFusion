"""Contact-masked depth evaluation of pilot checkpoints (best_depth.pt, selected on val).

Full-image depth MSE is dominated by background (depth == 0) pixels. This re-evaluates
each run on its real val split (and the held-out real test objects for the A runs) with:
  full_mse   all-pixel MSE [mm^2]            (the number history.json reports)
  c_mae      MAE inside the GT contact mask  [mm]
  c_rmse     RMSE inside the GT contact mask [mm]
  bg_mae     MAE outside the contact mask    [mm]
  iou        contact IoU, pred > IOU_THR mm vs GT > 0
  peak_err   |max(pred) - max(gt)| per image [mm]  (indentation-depth error)

usage: python ablation/pilot_robosoft/eval_masked.py [--prefix pilot_] [--device cuda:0] run ...
"""
import argparse
import contextlib
import io
import json
import os.path as osp

import torch
from torch.utils.data import DataLoader

from vistacfusion.data.dataset import build_datasets
from vistacfusion.engine.train import load_checkpoint
from vistacfusion.models.model import build_model
from vistacfusion.utils.config import merge_configs

OUT = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"
MODEL = "ablation/encoder/tac_sitr_single.yaml"
TRAIN = "ablation/pilot_robosoft/train_pilot_e50.yaml"
IOU_THR = 0.05  # mm


@torch.no_grad()
def masked_metrics(model, loader, device):
    s = dict(full_se=0.0, npix=0, c_ae=0.0, c_se=0.0, nc=0, bg_ae=0.0, nbg=0,
             iou=0.0, peak=0.0, n=0)
    for b in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(b["rgb"], b["tactile"], config="tactile", object_ids=b.get("object"))
        pred, gt = out["depth"].float(), b["depth"].float()
        m = gt > 0
        err = pred - gt
        s["full_se"] += (err ** 2).sum().item(); s["npix"] += err.numel()
        s["c_ae"] += err[m].abs().sum().item(); s["c_se"] += (err[m] ** 2).sum().item()
        s["nc"] += int(m.sum())
        s["bg_ae"] += err[~m].abs().sum().item(); s["nbg"] += int((~m).sum())
        pm = pred > IOU_THR
        inter = (pm & m).flatten(1).sum(1).float()
        union = (pm | m).flatten(1).sum(1).float().clamp(min=1)
        s["iou"] += (inter / union).sum().item()
        s["peak"] += (pred.flatten(1).amax(1) - gt.flatten(1).amax(1)).abs().sum().item()
        s["n"] += pred.shape[0]
    return {
        "full_mse": s["full_se"] / s["npix"],
        "c_mae": s["c_ae"] / max(1, s["nc"]),
        "c_rmse": (s["c_se"] / max(1, s["nc"])) ** 0.5,
        "bg_mae": s["bg_ae"] / max(1, s["nbg"]),
        "iou": s["iou"] / s["n"],
        "peak_err": s["peak"] / s["n"],
        "contact_frac": s["nc"] / s["npix"],
        "n": s["n"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--prefix", default="pilot_")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default="best_depth.pt")
    ap.add_argument("--json", default="ablation/pilot_robosoft/masked_results.json")
    args = ap.parse_args()

    results = json.load(open(args.json)) if osp.exists(args.json) else {}
    hdr = f"{'run':24s} {'split':5s} {'full_mse':>8s} {'c_mae':>7s} {'c_rmse':>7s} {'bg_mae':>7s} {'iou':>6s} {'peak':>6s}"
    print(hdr, flush=True)
    for run in args.runs:
        ckpt = osp.join(OUT, args.prefix + run, args.ckpt)
        if not osp.exists(ckpt):
            print(f"{run:24s} (no {args.ckpt})")
            continue
        cfg = merge_configs(MODEL, TRAIN, f"ablation/pilot_robosoft/data_{run}.yaml")
        with contextlib.redirect_stdout(io.StringIO()):
            train_ds, val_ds = build_datasets(cfg)
            model = build_model(cfg).to(args.device).eval()
            load_checkpoint(ckpt, model, device=args.device)
        splits = {"val": val_ds}
        if getattr(train_ds, "real_test", None) is not None:
            splits["TEST"] = train_ds.real_test
        for name, ds in splits.items():
            dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
            r = masked_metrics(model, dl, args.device)
            results[f"{args.prefix}{run}/{name}"] = r
            print(f"{run:24s} {name:5s} {r['full_mse']:8.4f} {r['c_mae']:7.4f} {r['c_rmse']:7.4f} "
                  f"{r['bg_mae']:7.4f} {r['iou']:6.3f} {r['peak_err']:6.3f}", flush=True)
        json.dump(results, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
