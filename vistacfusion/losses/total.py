"""Total multi-task loss.

Weighted sum of depth + normal + pose, with optional Kendall uncertainty weighting (learned
log-variances). supervise_dense=False skips the dense terms (for RGB-only batches when
rgb_only_supervises_dense is off).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .depth import DepthLoss
from .normal import NormalLoss
from .pose import PoseLoss


class MultiTaskLoss(nn.Module):
    def __init__(self, loss_cfg, pose_mode="regression", rot_num_bins=72):
        super().__init__()
        self.w_depth = loss_cfg.depth.weight
        self.w_normal = loss_cfg.normal.weight
        self.w_pose = loss_cfg.pose.weight
        self.uncertainty = loss_cfg.get("uncertainty_weighting", False)

        self.depth_loss = DepthLoss(
            kind=loss_cfg.depth.type,
            grad_matching_weight=loss_cfg.depth.get("grad_matching_weight", 0.0),
        )
        self.normal_loss = NormalLoss(kind=loss_cfg.normal.type)
        self.pose_loss = PoseLoss(
            rot_weight=loss_cfg.pose.rot_weight,
            trans_weight=loss_cfg.pose.trans_weight,
            pose_mode=pose_mode,
            rot_num_bins=rot_num_bins,
        )
        if self.uncertainty:
            # log(sigma^2) per task: loss_i / (2*sigma_i^2) + log(sigma_i)
            self.log_var = nn.Parameter(torch.zeros(3))

    def forward(self, pred, gt, supervise_dense=True):
        """pred: model output dict. gt: dict with depth/normal/pose/mask."""
        comps = {}
        terms = []
        weights = []
        mask = gt.get("mask")

        if supervise_dense:
            l_depth = self.depth_loss(pred["depth"], gt["depth"], mask=mask)
            l_normal = self.normal_loss(pred["normal"], gt["normal"], mask=mask)
            comps["depth"] = l_depth.detach()
            comps["normal"] = l_normal.detach()
            terms += [l_depth, l_normal]
            weights += [self.w_depth, self.w_normal]

        l_pose, pose_comps = self.pose_loss(pred, gt["pose"])
        comps["pose"] = l_pose.detach()
        comps.update({f"pose_{k}": v for k, v in pose_comps.items()})
        terms.append(l_pose)
        weights.append(self.w_pose)

        if self.uncertainty:
            # map present terms to their log_var slot: 0=depth,1=normal,2=pose
            idx = ([0, 1] if supervise_dense else []) + [2]
            total = 0.0
            for t, j in zip(terms, idx):
                total = total + torch.exp(-self.log_var[j]) * t + 0.5 * self.log_var[j]
        else:
            total = sum(w * t for w, t in zip(weights, terms))

        comps["total"] = total.detach()
        return total, comps
