"""Pose loss (SE(2)).

Rotation = 1 - cos(theta_pred - theta_gt), computed on (cos, sin) -- no atan2 in the loss.
Translation = L1 on (t_x, t_y). Total = rot_w * rot + trans_w * trans.

GT and regression prediction are both (cos, sin, t_x, t_y). Classification mode uses
cross-entropy over theta bins + L1 translation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseLoss(nn.Module):
    def __init__(self, rot_weight=1.0, trans_weight=1.0, pose_mode="regression", rot_num_bins=72):
        super().__init__()
        self.rot_w = rot_weight
        self.trans_w = trans_weight
        self.pose_mode = pose_mode
        self.rot_num_bins = rot_num_bins

    def forward(self, pred, gt):
        """gt: [B, 4] = (cos, sin, t_x, t_y).

        regression     pred: dict with se2=[B,4].
        classification pred: dict with rot_logits=[B,bins], trans=[B,2].
        """
        cos_gt, sin_gt, txy_gt = gt[:, 0], gt[:, 1], gt[:, 2:]

        if self.pose_mode == "regression":
            se2 = pred["se2"]
            cos_p, sin_p, txy_p = se2[:, 0], se2[:, 1], se2[:, 2:]
            # 1 - cos(theta_p - theta_gt) = 1 - (cosp*cosg + sinp*sing)
            l_rot = (1.0 - (cos_p * cos_gt + sin_p * sin_gt)).mean()
            l_trans = F.l1_loss(txy_p, txy_gt)
        else:
            theta_gt = torch.atan2(sin_gt, cos_gt)               # [-pi, pi]
            bins = ((theta_gt + math.pi) / (2 * math.pi) * self.rot_num_bins).long()
            bins = bins.clamp(0, self.rot_num_bins - 1)
            l_rot = F.cross_entropy(pred["rot_logits"], bins)
            l_trans = F.l1_loss(pred["trans"], txy_gt)

        total = self.rot_w * l_rot + self.trans_w * l_trans
        return total, {"rot": l_rot.detach(), "trans": l_trans.detach()}
