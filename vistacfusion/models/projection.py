"""Projection + embeddings: encoder dim -> trunk dim D (CLAUDE.md 3.3).

Per branch:
  - Linear(E -> D) applied per token (patch tokens and CLS).
  - Add a learned modality embedding (one vector per modality, broadcast to all tokens).
The 2D positional embedding for the 196 spatial queries is owned by the model (not here),
because the SAME positional emb is added to either real tactile patch tokens OR the
learnable spatial mask tokens in RGB-only mode (CLAUDE.md 4, gotcha 6/8).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BranchProjection(nn.Module):
    """Project one modality's encoder tokens to the trunk dim and add its modality embedding."""

    def __init__(self, in_dim=1024, out_dim=768):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.modality_emb = nn.Parameter(torch.zeros(1, 1, out_dim))
        nn.init.normal_(self.modality_emb, std=0.02)

    def forward(self, tokens):
        """tokens: [B, N, in_dim] -> [B, N, out_dim] + modality embedding."""
        return self.proj(tokens) + self.modality_emb


class SpatialPosEmbedding(nn.Module):
    """Learned 2D positional embedding for the ``num_tokens`` (=196) spatial queries.

    REQUIRED on the spatial queries: the DPT Reassemble reshapes 196 -> 14x14 and needs
    positional structure (CLAUDE.md 3.3). Added to whatever fills the spatial queries
    (tactile patch tokens or RGB-only mask tokens) so both carry identical position info.
    """

    def __init__(self, num_tokens=196, dim=768):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        return x + self.pos
