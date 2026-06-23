"""Pose head: SE(2) regression (CLAUDE.md 3.6, gotcha 5).

pose token [B, 1, D] -> LN -> Linear(D->256) -> GELU -> Dropout -> Linear(256->4).
The 4 outputs are (a, b, t_x, t_y); (a, b) are L2-normalized to (cos theta, sin theta).
Output SE(2) = (cos, sin, t_x, t_y) = [B, 4], 3 DoF. NEVER put a raw angle through the loss;
atan2 is only for the reported metric.

Optional classification mode (`pose_mode: classification`) bins theta into ``rot_num_bins``
classes (used only if regression is unstable; CLAUDE.md ablation C). It additionally
regresses (t_x, t_y).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseHead(nn.Module):
    def __init__(self, dim=768, hidden_dim=256, dropout=0.0,
                 pose_mode="regression", rot_num_bins=72):
        super().__init__()
        self.pose_mode = pose_mode
        self.rot_num_bins = rot_num_bins
        out_dim = 4 if pose_mode == "regression" else rot_num_bins + 2
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, pose_token):
        """pose_token: [B, 1, D].

        regression     -> dict(se2=[B,4] (cos,sin,tx,ty))
        classification -> dict(rot_logits=[B,bins], trans=[B,2])
        """
        x = pose_token.squeeze(1)                 # [B, D]
        out = self.net(x)
        if self.pose_mode == "regression":
            ab = out[:, :2]
            txy = out[:, 2:]
            cos_sin = F.normalize(ab, dim=-1, eps=1e-6)   # (a,b) -> (cos,sin)
            return {"se2": torch.cat([cos_sin, txy], dim=-1)}
        rot_logits = out[:, : self.rot_num_bins]
        trans = out[:, self.rot_num_bins:]
        return {"rot_logits": rot_logits, "trans": trans}
