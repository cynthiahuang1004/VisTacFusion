"""Full visuo-tactile multi-task model — v3 (decoupled dense/pose paths).

Dense (DPT) and Pose paths are decoupled:

  DPT path:  encoder multiscale taps at native dim (1024) → optional RGB injection
             → DPT Reassemble(1024→256) → depth/normal.
             Always has tactile features; RGB injection on/off controlled independently.

  Pose path: encoder → Projection(1024→768) → Fusion Trunk(768) → Pose Head.
             Modality dropout (both/tactile/rgb) only affects this path.

Benefits:
  - DPT gets 100% tactile training (no modality dropout dilution)
  - Direct encoder taps at 1024 preserve spatial detail (matches DINOv3 Pipeline baseline)
  - RGB injection is residual with ReZero gate (init=0 → starts as pure encoder taps)
  - Pose path unchanged from v1
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
    """Per-tap residual RGB injection: tap += gate * CrossAttn(Q=tap, K=V=bottleneck).

    Q is at encoder dim (e.g. 1024), KV at trunk dim (e.g. 768).
    ReZero gate init 0 → starts as pure encoder taps, learns to inject.
    """

    def __init__(self, q_dim, kv_dim, num_heads, dropout=0.0, gate_init=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim)
        self.v_proj = nn.Linear(kv_dim, q_dim)
        self.out_proj = nn.Linear(q_dim, q_dim)
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, tap, bottleneck):
        B, N, _ = tap.shape
        q = self.q_proj(self.norm_q(tap))
        kv_normed = self.norm_kv(bottleneck)
        k = self.k_proj(kv_normed)
        v = self.v_proj(kv_normed)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, -1)
        return tap + self.gate * self.out_proj(attn)


class VisuoTactileModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.trunk_dim = cfg.trunk_dim                              # 768
        self.num_spatial = cfg.tokens.num_spatial_queries            # 196

        # ---- Encoder (frozen, shared weights) ----
        enc_cfg = cfg.encoder
        if enc_cfg.get("multiscale_layers", None) is None:
            enc_cfg = dict(enc_cfg)
            enc_cfg["multiscale_layers"] = list(cfg.heads.dpt.encoder_tap_layers)
        self.tactile_encoder = build_encoder(enc_cfg, self.image_size)
        if enc_cfg.get("share_encoder_weights", True):
            self.rgb_encoder = self.tactile_encoder
        else:
            self.rgb_encoder = build_encoder(enc_cfg, self.image_size)
        self.enc_dim = self.tactile_encoder.embed_dim               # 1024

        # ---- Pose path: projection (1024→768) + trunk ----
        self.tactile_proj = BranchProjection(self.enc_dim, self.trunk_dim)
        self.rgb_proj = BranchProjection(self.enc_dim, self.trunk_dim)
        self.spatial_pos = SpatialPosEmbedding(self.num_spatial, self.trunk_dim)
        self.use_rgb_pos = cfg.projection.get("rgb_positional_embedding", False)
        if self.use_rgb_pos:
            self.rgb_spatial_pos = SpatialPosEmbedding(self.num_spatial, self.trunk_dim)

        self.spatial_mask = nn.Parameter(torch.zeros(1, self.num_spatial, self.trunk_dim))
        self.pose_mask = nn.Parameter(torch.zeros(1, 1, self.trunk_dim))
        nn.init.trunc_normal_(self.spatial_mask, std=0.02)
        nn.init.trunc_normal_(self.pose_mask, std=0.02)

        self.trunk = FusionTrunk(cfg.fusion_trunk, self.trunk_dim)

        # ---- DPT path: direct encoder taps (1024) + RGB injection ----
        self.dpt_pos = SpatialPosEmbedding(self.num_spatial, self.enc_dim)

        self.tap_inject = nn.ModuleList([
            TapInjection(
                q_dim=self.enc_dim,
                kv_dim=self.trunk_dim,
                num_heads=cfg.fusion_trunk.num_heads,
                dropout=cfg.fusion_trunk.dropout,
                gate_init=cfg.heads.dpt.get("inject_gate_init", 0.0),
            )
            for _ in range(4)
        ])

        # ---- Heads ----
        self.dpt = DPTHead(
            embed_dim=self.enc_dim,
            features=cfg.heads.dpt.features,
            dropout=cfg.heads.dpt.dropout,
            out_depth_channels=cfg.heads.dpt.out_depth_channels,
            out_normal_channels=cfg.heads.dpt.out_normal_channels,
        )
        self.pose_head = PoseHead(
            dim=self.trunk_dim,
            hidden_dim=cfg.heads.pose.hidden_dim,
            dropout=cfg.heads.pose.dropout,
            pose_mode=cfg.heads.pose.pose_mode,
            rot_num_bins=cfg.heads.pose.get("rot_num_bins", 72),
            use_spatial_pool=cfg.heads.pose.get("use_spatial_pool", True),
        )

    # ------------------------------------------------------------------ helpers

    def _build_pose_memory(self, rgb_patch, rgb_cls):
        """RGB → pose memory M = [B, 197, trunk_dim]."""
        patch = self.rgb_proj(rgb_patch)
        cls = self.rgb_proj(rgb_cls)
        if self.use_rgb_pos:
            patch = self.rgb_spatial_pos(patch)
        return torch.cat([patch, cls], dim=1)

    def _build_pose_queries(self, tac_patch, tac_cls, use_tactile, B, device):
        """Build 197 pose queries at trunk_dim."""
        if use_tactile:
            spatial = self.spatial_pos(self.tactile_proj(tac_patch))
            pose_q = self.tactile_proj(tac_cls)
        else:
            spatial = self.spatial_pos(self.spatial_mask.expand(B, -1, -1))
            pose_q = self.pose_mask.expand(B, -1, -1)
        return torch.cat([spatial, pose_q], dim=1)

    # ------------------------------------------------------------------ forward

    def forward(self, rgb, tactile, config="both", inject_rgb_to_dpt=None,
                encoder_cache=None):
        """
        Args:
            config: modality config for the POSE path ("both"/"tactile"/"rgb").
            inject_rgb_to_dpt: whether to inject RGB into DPT taps.
                None = auto (inject when config has RGB).
                Can be overridden for decoupled dropout training.
            encoder_cache: pre-computed encoder outputs (skips frozen forward).
        """
        if config not in VALID_CONFIGS:
            raise ValueError(f"config must be one of {VALID_CONFIGS}, got {config!r}")
        use_rgb, use_tactile = _config_flags(config)

        ref = tactile if use_tactile else rgb
        B, device = ref.shape[0], ref.device

        if inject_rgb_to_dpt is None:
            inject_rgb_to_dpt = use_rgb

        # ---- Encode (frozen) ----
        tac_enc = encoder_cache.get("tactile") if encoder_cache else None
        rgb_enc = encoder_cache.get("rgb") if encoder_cache else None

        tac_patch = tac_cls = rgb_patch = rgb_cls = None
        if use_tactile:
            tac_patch, tac_cls = tac_enc if tac_enc else self.tactile_encoder(tactile)
        if use_rgb:
            rgb_patch, rgb_cls = rgb_enc if rgb_enc else self.rgb_encoder(rgb)

        # ---- Pose path: projection(768) → trunk → pose head ----
        pose_memory = self._build_pose_memory(rgb_patch, rgb_cls) if use_rgb else None
        pose_queries = self._build_pose_queries(
            tac_patch, tac_cls, use_tactile, B, device)

        trunk_taps, pose_token, bottleneck = self.trunk(
            pose_queries, pose_memory, use_rgb)

        spatial_queries = trunk_taps[-1]
        pose = self.pose_head(pose_token, spatial_queries=spatial_queries)

        # ---- DPT path: encoder multiscale taps (1024) ----
        if use_tactile:
            ms = self.tactile_encoder.forward_multiscale(tactile)
        else:
            ms = self.rgb_encoder.forward_multiscale(rgb)
        dpt_taps = [self.dpt_pos(t) for t in ms]

        if inject_rgb_to_dpt and use_rgb:
            dpt_taps = [inj(t, bottleneck)
                        for inj, t in zip(self.tap_inject, dpt_taps)]

        depth, normal = self.dpt(dpt_taps, out_hw=(self.image_size, self.image_size))

        out = {"depth": depth, "normal": normal}
        out.update(pose)
        return out


def build_model(cfg):
    return VisuoTactileModel(cfg)
