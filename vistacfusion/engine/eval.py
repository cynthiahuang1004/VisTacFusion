"""Evaluation: report metrics per modality config (both / tactile / rgb).

Uses the SAME loss functions as training (MSE depth, MSE normal, 1-cos rot, L1 trans)
so train vs val numbers are directly comparable. Also reports pose_rot_deg for
interpretability.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

CONFIGS = ("both", "tactile", "rgb")


def _theta_to_bin(theta, num_bins=72):
    bin_size = 2 * math.pi / num_bins
    bins = ((theta + math.pi) / bin_size).long()
    return bins.clamp(0, num_bins - 1)


@torch.no_grad()
def evaluate(model, loader, cfg, device, configs=CONFIGS):
    model.eval()
    report_per_config = cfg.eval.get("report_per_config", True)
    configs = configs if report_per_config else ("both",)
    pose_mode = cfg.heads.pose.pose_mode
    rot_num_bins = cfg.heads.pose.get("rot_num_bins", 72)

    acc = {c: {"depth_mse": 0.0, "normal_mse": 0.0,
               "pose_rot": 0.0, "pose_trans": 0.0,
               "rot_deg": 0.0, "n": 0} for c in configs}

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        bs = batch["rgb"].shape[0]

        gt_pose = batch["pose"]
        cos_gt, sin_gt = gt_pose[:, 0], gt_pose[:, 1]
        txy_gt = gt_pose[:, 2:]

        for c in configs:
            out = model(batch["rgb"], batch["tactile"], config=c)
            a = acc[c]

            a["depth_mse"] += F.mse_loss(out["depth"], batch["depth"]).item() * bs
            a["normal_mse"] += F.mse_loss(out["normal"], batch["normal"]).item() * bs

            if pose_mode == "classification" and "rot_logits" in out:
                theta_gt = torch.atan2(sin_gt, cos_gt)
                target_bins = _theta_to_bin(theta_gt, rot_num_bins)
                a["pose_rot"] += F.cross_entropy(out["rot_logits"], target_bins).item() * bs
                a["pose_trans"] += F.l1_loss(out["trans"], txy_gt).item() * bs
            elif "se2" in out:
                se2 = out["se2"]
                cos_p, sin_p = se2[:, 0], se2[:, 1]
                a["pose_rot"] += (1.0 - (cos_p * cos_gt + sin_p * sin_gt)).mean().item() * bs
                a["pose_trans"] += F.l1_loss(se2[:, 2:], txy_gt).item() * bs

            if "se2" in out:
                cos_p, sin_p = out["se2"][:, 0], out["se2"][:, 1]
                dcos = (cos_p * cos_gt + sin_p * sin_gt).clamp(-1 + 1e-6, 1 - 1e-6)
                a["rot_deg"] += (torch.acos(dcos).mean().item() * 180.0 / math.pi) * bs

            a["n"] += bs

    report = {}
    for c, a in acc.items():
        n = max(1, a["n"])
        report[c] = {
            "depth_mse": round(a["depth_mse"] / n, 6),
            "normal_mse": round(a["normal_mse"] / n, 6),
            "pose_rot": round(a["pose_rot"] / n, 4),
            "pose_trans": round(a["pose_trans"] / n, 4),
            "pose_rot_deg": round(a["rot_deg"] / n, 3),
        }
    return report
