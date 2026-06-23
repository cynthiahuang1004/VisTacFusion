"""Full visuo-tactile multi-task model (CLAUDE.md 3, 4, 3.7).

forward(rgb, tactile, config) with config in {"both", "tactile", "rgb"} produces an
IDENTICAL decoder input in every config: 4 x [B, 196, D] spatial taps + [B, 1, D] pose token.
Only *what fills the queries* and *whether RGB exists* changes.

Tactile is the spatial anchor; RGB is read-only context (K/V only). RGB-only swaps real
tactile queries for learnable mask tokens (same count, same positional emb).

DPT tap source (CLAUDE.md 3.7):
  - v1 `trunk`              : 4 taps are the spatial queries from 4 fusion-trunk layers.
  - v2 `encoder_multiscale` : 4 taps come from 4 tactile-encoder layers (+ per-tap residual
                              RGB injection through the condensed bottleneck; gate init 0).
Both are built from day 1 and flag-switched; they are NOT checkpoint-compatible (retrain).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import build_encoder
from .fusion import AttentionBlock, FusionTrunk
from .heads.dpt import DPTHead
from .heads.pose import PoseHead
from .projection import BranchProjection, SpatialPosEmbedding

VALID_CONFIGS = ("both", "tactile", "rgb")


def _config_flags(config):
    """(use_rgb, use_tactile) for each modality config."""
    if config == "both":
        return True, True
    if config == "tactile":
        return False, True
    if config == "rgb":
        return True, False
    raise ValueError(f"config must be one of {VALID_CONFIGS}, got {config!r}")


class TapInjection(nn.Module):
    """v2 per-tap residual RGB injection: tap += gate * CrossAttn(Q=tap, K=V=bottleneck).

    ReZero-style learnable gate, init 0 -> v2 starts as PURE encoder taps and learns to
    inject. When RGB is absent the caller SKIPS this entirely (adds exactly 0), so
    single-modality = pure encoder taps (fairness, CLAUDE.md 3.7 / gotcha 9).
    """

    def __init__(self, dim, num_heads, dropout=0.0, gate_init=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, tap, bottleneck):
        q = self.norm_q(tap)
        kv = self.norm_kv(bottleneck)
        a, _ = self.attn(q, kv, kv, need_weights=False)
        return tap + self.gate * a


class VisuoTactileModel(nn.Module):
    def __init__(self, cfg):
        """cfg: the merged model config (configs/model.yaml as a namespace)."""
        super().__init__()
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.dim = cfg.trunk_dim
        self.num_spatial = cfg.tokens.num_spatial_queries        # 196
        self.tap_source = cfg.heads.dpt.tap_source               # {trunk, encoder_multiscale}

        # ---- Encoders (frozen). Shared weights or two instances. ----
        enc_cfg = cfg.encoder
        if enc_cfg.get("multiscale_layers", None) is None:
            enc_cfg = dict(enc_cfg)
            enc_cfg["multiscale_layers"] = list(cfg.heads.dpt.encoder_tap_layers)
        self.tactile_encoder = build_encoder(enc_cfg, self.image_size)
        if enc_cfg.get("share_encoder_weights", True):
            self.rgb_encoder = self.tactile_encoder
        else:
            self.rgb_encoder = build_encoder(enc_cfg, self.image_size)
        enc_dim = self.tactile_encoder.embed_dim

        # ---- Projections (E -> D) + modality embeddings ----
        self.tactile_proj = BranchProjection(enc_dim, self.dim)
        self.rgb_proj = BranchProjection(enc_dim, self.dim)
        self.spatial_pos = SpatialPosEmbedding(self.num_spatial, self.dim)
        self.use_rgb_pos = cfg.projection.get("rgb_positional_embedding", False)
        if self.use_rgb_pos:
            self.rgb_spatial_pos = SpatialPosEmbedding(self.num_spatial, self.dim)

        # ---- Mask tokens for RGB-only (CLAUDE.md 4, gotcha 6/8) ----
        self.spatial_mask = nn.Parameter(torch.zeros(1, self.num_spatial, self.dim))
        self.pose_mask = nn.Parameter(torch.zeros(1, 1, self.dim))
        nn.init.trunc_normal_(self.spatial_mask, std=0.02)
        nn.init.trunc_normal_(self.pose_mask, std=0.02)

        # ---- Fusion trunk ----
        self.trunk = FusionTrunk(cfg.fusion_trunk, self.dim)

        # ---- DPT v2 components (always built, for fairness) ----
        if self.tap_source == "encoder_multiscale":
            self.tap_proj = nn.ModuleList(
                [nn.Linear(enc_dim, self.dim) for _ in range(4)]
            )
            self.tap_pos = SpatialPosEmbedding(self.num_spatial, self.dim)
            # learnable taps for the (rare) RGB-only + v2 case (no tactile encoder taps)
            self.tap_mask = nn.Parameter(torch.zeros(4, self.num_spatial, self.dim))
            nn.init.trunc_normal_(self.tap_mask, std=0.02)
            self.tap_inject = nn.ModuleList([
                TapInjection(self.dim, cfg.fusion_trunk.num_heads, cfg.fusion_trunk.dropout,
                             gate_init=cfg.heads.dpt.get("inject_gate_init", 0.0))
                for _ in range(4)
            ])

        # ---- Heads ----
        self.dpt = DPTHead(
            embed_dim=self.dim,
            features=cfg.heads.dpt.features,
            dropout=cfg.heads.dpt.dropout,
            out_depth_channels=cfg.heads.dpt.out_depth_channels,
            out_normal_channels=cfg.heads.dpt.out_normal_channels,
        )
        self.pose_head = PoseHead(
            dim=self.dim,
            hidden_dim=cfg.heads.pose.hidden_dim,
            dropout=cfg.heads.pose.dropout,
            pose_mode=cfg.heads.pose.pose_mode,
            rot_num_bins=cfg.heads.pose.get("rot_num_bins", 72),
        )

    # ------------------------------------------------------------------ helpers
    def _build_memory(self, rgb):
        """RGB -> read-only memory M = [B, 197, D] (196 patch + 1 CLS)."""
        patch, cls = self.rgb_encoder(rgb)
        patch = self.rgb_proj(patch)
        cls = self.rgb_proj(cls)
        if self.use_rgb_pos:
            patch = self.rgb_spatial_pos(patch)
        return torch.cat([patch, cls], dim=1)            # [B, 197, D]

    def _build_queries(self, tactile, use_tactile, batch_size, device):
        """Assemble the 197 queries: 196 spatial + 1 pose."""
        if use_tactile:
            patch, cls = self.tactile_encoder(tactile)
            spatial = self.spatial_pos(self.tactile_proj(patch))   # [B, 196, D]
            pose_q = self.tactile_proj(cls)                        # [B, 1, D]
        else:
            spatial = self.spatial_pos(self.spatial_mask.expand(batch_size, -1, -1))
            pose_q = self.pose_mask.expand(batch_size, -1, -1)
        return torch.cat([spatial, pose_q], dim=1)                 # [B, 197, D]

    def _v2_taps(self, tactile, use_tactile, use_rgb, bottleneck, batch_size):
        """4 encoder-multiscale taps (+ residual RGB injection)."""
        if use_tactile:
            ms = self.tactile_encoder.forward_multiscale(tactile)  # 4 x [B, 196, E]
            taps = [self.tap_pos(proj(m)) for proj, m in zip(self.tap_proj, ms)]
        else:
            taps = [self.tap_pos(self.tap_mask[i].unsqueeze(0).expand(batch_size, -1, -1))
                    for i in range(4)]
        if use_rgb:
            taps = [inj(t, bottleneck) for inj, t in zip(self.tap_inject, taps)]
        return taps

    # ------------------------------------------------------------------ forward
    def forward(self, rgb, tactile, config="both", return_decoder_inputs=False):
        if config not in VALID_CONFIGS:
            raise ValueError(f"config must be one of {VALID_CONFIGS}, got {config!r}")
        use_rgb, use_tactile = _config_flags(config)

        ref = tactile if use_tactile else rgb
        B, device = ref.shape[0], ref.device

        memory = self._build_memory(rgb) if use_rgb else None
        queries = self._build_queries(tactile, use_tactile, B, device)

        trunk_taps, pose_token, bottleneck = self.trunk(queries, memory, use_rgb)

        if self.tap_source == "trunk":
            taps = trunk_taps                                       # 4 x [B, 196, D]
        else:  # encoder_multiscale (v2)
            taps = self._v2_taps(tactile, use_tactile, use_rgb, bottleneck, B)

        if return_decoder_inputs:
            return {"taps": taps, "pose_token": pose_token}

        depth, normal = self.dpt(taps, out_hw=(self.image_size, self.image_size))
        pose = self.pose_head(pose_token)
        out = {"depth": depth, "normal": normal}
        out.update(pose)
        return out


def build_model(cfg):
    return VisuoTactileModel(cfg)
