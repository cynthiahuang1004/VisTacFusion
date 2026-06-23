"""Eval metrics (CLAUDE.md train.yaml eval.metrics)."""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def depth_absrel(pred, gt, eps=1e-6):
    return (((pred - gt).abs()) / gt.abs().clamp_min(eps)).mean().item()


@torch.no_grad()
def depth_rmse(pred, gt):
    return torch.sqrt(((pred - gt) ** 2).mean()).item()


@torch.no_grad()
def normal_mean_angle_deg(pred, gt, eps=1e-6):
    import torch.nn.functional as F
    p = F.normalize(pred, dim=1, eps=eps)
    g = F.normalize(gt, dim=1, eps=eps)
    cos = (p * g).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    return (torch.acos(cos).mean().item() * 180.0 / math.pi)


@torch.no_grad()
def pose_rot_deg(pred_se2, gt):
    # pred_se2, gt: [B, 4] (cos, sin, ...). angular error in degrees.
    cosp, sinp = pred_se2[:, 0], pred_se2[:, 1]
    cosg, sing = gt[:, 0], gt[:, 1]
    dcos = (cosp * cosg + sinp * sing).clamp(-1 + 1e-6, 1 - 1e-6)
    return (torch.acos(dcos).mean().item() * 180.0 / math.pi)


@torch.no_grad()
def pose_trans_l1(pred_se2, gt):
    return (pred_se2[:, 2:] - gt[:, 2:]).abs().mean().item()
