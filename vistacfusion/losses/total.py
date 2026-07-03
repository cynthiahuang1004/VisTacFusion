"""Total multi-task loss.

4-task Kendall uncertainty weighting: depth, normal, pose_rot, pose_trans each get
an independent learned log-variance. supervise_dense=False skips dense terms.
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
        self.w_rot = loss_cfg.pose.rot_weight
        self.w_trans = loss_cfg.pose.trans_weight
        self.uncertainty = loss_cfg.get("uncertainty_weighting", False)

        self.depth_loss = DepthLoss(
            kind=loss_cfg.depth.type,
            grad_matching_weight=loss_cfg.depth.get("grad_matching_weight", 0.0),
        )
        self.normal_loss = NormalLoss(kind=loss_cfg.normal.type)
        self.pose_loss = PoseLoss(
            pose_mode=pose_mode,
            rot_num_bins=rot_num_bins,
        )
        if self.uncertainty:
            # 4 independent log(sigma^2): depth, normal, rot, trans
            self.log_var = nn.Parameter(torch.zeros(4))

    def forward(self, pred, gt, supervise_dense=True):
        """pred: model output dict. gt: dict with depth/normal/pose/mask."""
        comps = {}
        terms = []
        weights = []

        if supervise_dense:
            l_depth = self.depth_loss(pred["depth"], gt["depth"])
            l_normal = self.normal_loss(pred["normal"], gt["normal"])
            comps["depth"] = l_depth.detach()
            comps["normal"] = l_normal.detach()
            terms += [l_depth, l_normal]
            weights += [self.w_depth, self.w_normal]

        l_rot, l_trans = self.pose_loss(pred, gt["pose"])
        comps["pose_rot"] = l_rot.detach()
        comps["pose_trans"] = l_trans.detach()
        terms += [l_rot, l_trans]
        weights += [self.w_rot, self.w_trans]

        if self.uncertainty:
            # 0=depth, 1=normal, 2=rot, 3=trans
            idx = ([0, 1] if supervise_dense else []) + [2, 3]
            total = 0.0
            for t, j in zip(terms, idx):
                total = total + torch.exp(-self.log_var[j]) * t + 0.5 * self.log_var[j]
        else:
            total = sum(w * t for w, t in zip(weights, terms))

        comps["total"] = total.detach()
        return total, comps
