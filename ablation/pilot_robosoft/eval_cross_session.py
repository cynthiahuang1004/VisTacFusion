"""Cross-session evaluation: run old-session checkpoints on the new-session real data.

No training — just loads the checkpoint and evaluates on real_filtered_new as a test set.
Reports both full-image and contact-masked metrics.

usage: python ablation/pilot_robosoft/eval_cross_session.py [--device cuda:0] run:prefix ...
  e.g.  A_realonly:pilot_  A_blender:pilot_  A_ganloo:pilot_  A_bgcondloo:pilot_  A_realonly:pilot150_
"""
import argparse
import contextlib
import io
import json
import os.path as osp

import torch
from torch.utils.data import DataLoader

from vistacfusion.data.dataset import SimVisuoTactileDataset
from vistacfusion.engine.train import load_checkpoint
from vistacfusion.models.model import build_model
from vistacfusion.utils.config import merge_configs

OUT = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"
MODEL = "ablation/encoder/tac_sitr_single.yaml"
TRAIN = "ablation/pilot_robosoft/train_pilot_e50.yaml"
NEW_ROOT = "/media/hdd2/ihsuan/gs_blender/real_filtered_new_curated"
OLD_ROOT = "/media/hdd2/ihsuan/gs_blender/real_filtered"
IOU_THR = 0.05


@torch.no_grad()
def evaluate(model, loader, device):
    s = dict(full_se=0.0, npix=0, c_ae=0.0, c_se=0.0, nc=0, bg_ae=0.0, nbg=0,
             iou=0.0, peak=0.0, rot_sum=0.0, rot_n=0, trans_sum=0.0, n=0, nang=0.0, nang_n=0)
    import torch.nn.functional as F
    import math
    for b in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(b["rgb"], b["tactile"], config="tactile", object_ids=b.get("object"))
        pred, gt = out["depth"].float(), b["depth"].float()
        m = gt > 0; err = pred - gt
        s["full_se"] += (err ** 2).sum().item(); s["npix"] += err.numel()
        s["c_ae"] += err[m].abs().sum().item(); s["c_se"] += (err[m] ** 2).sum().item()
        s["nc"] += int(m.sum())
        s["bg_ae"] += err[~m].abs().sum().item(); s["nbg"] += int((~m).sum())
        if "normal" in out and "normal" in b:
            pn = F.normalize(out["normal"].float(), dim=1); gn = F.normalize(b["normal"].float(), dim=1)
            ang = torch.acos((pn * gn).sum(1).clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / math.pi   # [B,H,W]
            s["nang"] += ang[m[:, 0]].sum().item(); s["nang_n"] += int(m.sum())              # contact region
        pm = pred > IOU_THR
        inter = (pm & m).flatten(1).sum(1).float()
        union = (pm | m).flatten(1).sum(1).float().clamp(min=1)
        s["iou"] += (inter / union).sum().item()
        s["peak"] += (pred.flatten(1).amax(1) - gt.flatten(1).amax(1)).abs().sum().item()
        if "se2" in out:
            se2 = out["se2"]; gt_p = b["pose"]
            cos_p, sin_p = se2[:, 0], se2[:, 1]; cos_g, sin_g = gt_p[:, 0], gt_p[:, 1]
            dcos = (cos_p * cos_g + sin_p * sin_g).clamp(-1 + 1e-6, 1 - 1e-6)
            s["rot_sum"] += (torch.acos(dcos) * 180 / math.pi).sum().item()
            s["trans_sum"] += F.l1_loss(se2[:, 2:], gt_p[:, 2:], reduction="sum").item()
            s["rot_n"] += se2.shape[0]
        s["n"] += pred.shape[0]
    return {
        "full_mse": s["full_se"] / s["npix"],
        "c_mae": s["c_ae"] / max(1, s["nc"]),
        "c_rmse": (s["c_se"] / max(1, s["nc"])) ** 0.5,
        "iou": s["iou"] / s["n"],
        "peak_err": s["peak"] / s["n"],
        "normal_deg": s["nang"] / max(1, s["nang_n"]),
        "rot_deg": s["rot_sum"] / max(1, s["rot_n"]),
        "trans_l1": s["trans_sum"] / max(1, s["rot_n"]),
        "n": s["n"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run:prefix e.g. A_realonly:pilot_")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default="best_depth.pt")
    ap.add_argument("--json", default="ablation/pilot_robosoft/cross_session_results.json")
    ap.add_argument("--root", default=NEW_ROOT)
    args = ap.parse_args()

    results = json.load(open(args.json)) if osp.exists(args.json) else {}
    hdr = f"{'run':28s} {'prefix':10s} {'mse':>8s} {'c_mae':>7s} {'iou':>6s} {'peak':>6s} {'nrm':>6s} {'rot':>6s} {'trans':>6s} {'n':>4s}"
    print(hdr, flush=True)

    for spec in args.runs:
        run, prefix = spec.split(":")
        ckpt = osp.join(OUT, prefix + run, args.ckpt)
        if not osp.exists(ckpt):
            print(f"{run:28s} {prefix:10s} (no {args.ckpt})"); continue
        data_cfg = f"ablation/pilot_robosoft/data_{run}.yaml"
        if not osp.exists(data_cfg):
            data_cfg = f"ablation/pilot_robosoft/data_A_realonly.yaml"
        cfg = merge_configs(MODEL, TRAIN, data_cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            # Build the old-session obj map so object IDs match the checkpoint
            old_ds = SimVisuoTactileDataset(cfg, cfg.image_size, augment=False, split="val",
                                            data_section="real")
            shared_obj_map = old_ds._obj_to_id
            # Load new-session data with the same obj map
            import copy
            cfg_new = copy.deepcopy(cfg)
            cfg_new["real"]["root"] = args.root
            cfg_new["real"]["val_every"] = 1  # all frames are "val" (no training on this data)
            cfg_new["real"].pop("test_objects", None)  # don't exclude any objects
            if cfg_new["real"].get("bg_subtract"):     # new-session backgrounds live in real_new/
                cfg_new["real"]["bg_subtract"] = "real_new"
            new_ds = SimVisuoTactileDataset(cfg_new, cfg.image_size, augment=False, split="all",
                                            data_section="real", shared_obj_map=shared_obj_map)
            model = build_model(cfg).to(args.device).eval()
            load_checkpoint(ckpt, model, device=args.device)

        dl = DataLoader(new_ds, batch_size=64, shuffle=False, num_workers=4)
        r = evaluate(model, dl, args.device)
        key = f"{prefix}{run}/cross_session_curated"
        results[key] = r
        print(f"{run:28s} {prefix:10s} {r['full_mse']:8.4f} {r['c_mae']:7.4f} {r['iou']:6.3f} "
              f"{r['peak_err']:6.3f} {r['normal_deg']:6.2f} {r['rot_deg']:6.2f} {r['trans_l1']:6.4f} {r['n']:4d}", flush=True)
        json.dump(results, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
