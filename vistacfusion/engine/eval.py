"""Evaluation: report metrics per modality config (both / tactile / rgb).

This per-config table is the fairness ablation -- the claim "RGB+tactile beats either alone"
is read directly off it.
"""
from __future__ import annotations

import torch

from . import metrics as M

CONFIGS = ("both", "tactile", "rgb")


@torch.no_grad()
def evaluate(model, loader, cfg, device, configs=CONFIGS):
    model.eval()
    report_per_config = cfg.eval.get("report_per_config", True)
    configs = configs if report_per_config else ("both",)

    acc = {c: {"absrel": 0.0, "rmse": 0.0, "nangle": 0.0,
               "rot": 0.0, "trans": 0.0, "n": 0} for c in configs}

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        bs = batch["rgb"].shape[0]
        for c in configs:
            out = model(batch["rgb"], batch["tactile"], config=c)
            a = acc[c]
            a["absrel"] += M.depth_absrel(out["depth"], batch["depth"]) * bs
            a["rmse"] += M.depth_rmse(out["depth"], batch["depth"]) * bs
            a["nangle"] += M.normal_mean_angle_deg(out["normal"], batch["normal"]) * bs
            if "se2" in out:
                a["rot"] += M.pose_rot_deg(out["se2"], batch["pose"]) * bs
                a["trans"] += M.pose_trans_l1(out["se2"], batch["pose"]) * bs
            a["n"] += bs

    report = {}
    for c, a in acc.items():
        n = max(1, a["n"])
        report[c] = {
            "depth_absrel": round(a["absrel"] / n, 4),
            "depth_rmse": round(a["rmse"] / n, 4),
            "normal_mean_angle": round(a["nangle"] / n, 3),
            "pose_rot_deg": round(a["rot"] / n, 3),
            "pose_trans": round(a["trans"] / n, 4),
        }
    return report
