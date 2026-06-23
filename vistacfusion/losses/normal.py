"""Normal loss (CLAUDE.md 5): cosine / angular loss 1 - cos(angle), not MSE on raw vectors."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt, mask=None):
        # pred, gt: [B, 3, H, W]. L2-normalize both, then 1 - cos.
        p = F.normalize(pred, dim=1, eps=self.eps)
        g = F.normalize(gt, dim=1, eps=self.eps)
        cos = (p * g).sum(dim=1, keepdim=True)            # [B, 1, H, W]
        loss = 1.0 - cos
        if mask is not None:
            return (loss * mask).sum() / mask.sum().clamp_min(1)
        return loss.mean()
