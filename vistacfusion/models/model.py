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


class CoralAdapter(nn.Module):
    """Sim→real CORAL feature alignment for one encoder branch (frozen buffers).

    Applies f' = (f − μ_s) A + μ_r to samples with domain_id == 0 (sim);
    real samples pass through untouched, so inference on real is unchanged.
    A is the global whitening-recoloring matrix; with per_object_mean the
    means are per-object (objects without real data fall back to global).
    Stats come from scripts/compute_coral_stats.py.
    """

    def __init__(self, stats, per_object_mean=True):
        super().__init__()
        for k in ("patch_mu_s", "patch_mu_r", "patch_A",
                  "cls_mu_s", "cls_mu_r", "cls_A",
                  "obj_patch_mu_s", "obj_patch_mu_r",
                  "obj_cls_mu_s", "obj_cls_mu_r"):
            self.register_buffer(k, stats[k].float())
        self.register_buffer("obj_has_real", stats["obj_has_real"].bool())
        self.per_object_mean = per_object_mean

    def _mus(self, kind, B, object_ids, device):
        mu_s = getattr(self, f"{kind}_mu_s")
        mu_r = getattr(self, f"{kind}_mu_r")
        if self.per_object_mean and object_ids is not None:
            has = self.obj_has_real[object_ids]                    # [B]
            o_s = getattr(self, f"obj_{kind}_mu_s")[object_ids]    # [B, D]
            o_r = getattr(self, f"obj_{kind}_mu_r")[object_ids]
            mu_s = torch.where(has[:, None], o_s, mu_s.expand(B, -1))
            mu_r = torch.where(has[:, None], o_r, mu_r.expand(B, -1))
            return mu_s.unsqueeze(1), mu_r.unsqueeze(1)            # [B, 1, D]
        return mu_s.view(1, 1, -1).expand(B, 1, -1), mu_r.view(1, 1, -1).expand(B, 1, -1)

    def _transform(self, x, kind, sim_mask, object_ids):
        """x: [B, N, D] → transform sim rows only (in float32)."""
        B = x.shape[0]
        mu_s, mu_r = self._mus(kind, B, object_ids, x.device)
        A = getattr(self, f"{kind}_A")
        y = (x.float() - mu_s) @ A + mu_r
        return torch.where(sim_mask[:, None, None], y.to(x.dtype), x)

    def forward(self, patch, cls, domain_ids, object_ids=None):
        if domain_ids is None:
            return patch, cls
        sim_mask = domain_ids == 0
        if not sim_mask.any():
            return patch, cls
        patch = self._transform(patch, "patch", sim_mask, object_ids)
        if cls is not None:
            cls = self._transform(cls, "cls", sim_mask, object_ids)
        return patch, cls


class CoralTapAdapter(nn.Module):
    """CORAL alignment for the DPT multiscale tap layers (per-tap statistics).

    Same transform as CoralAdapter but one (μ_s, μ_r, A) set per tap layer,
    patch tokens only. Stats from scripts/compute_coral_tap_stats.py.
    """

    def __init__(self, tap_stats, obj_has_real, per_object_mean=True):
        super().__init__()
        self.n_taps = len([k for k in tap_stats if k.startswith("tap")])
        for t in range(self.n_taps):
            s = tap_stats[f"tap{t}"]
            self.register_buffer(f"t{t}_mu_s", s["patch_mu_s"].float())
            self.register_buffer(f"t{t}_mu_r", s["patch_mu_r"].float())
            self.register_buffer(f"t{t}_A", s["patch_A"].float())
            self.register_buffer(f"t{t}_obj_mu_s", s["obj_patch_mu_s"].float())
            self.register_buffer(f"t{t}_obj_mu_r", s["obj_patch_mu_r"].float())
        self.register_buffer("obj_has_real", obj_has_real.bool())
        self.per_object_mean = per_object_mean

    def forward(self, taps, domain_ids, object_ids=None):
        if domain_ids is None:
            return taps
        sim_mask = domain_ids == 0
        if not sim_mask.any():
            return taps
        out = []
        for t, x in enumerate(taps):
            B = x.shape[0]
            mu_s = getattr(self, f"t{t}_mu_s")
            mu_r = getattr(self, f"t{t}_mu_r")
            if self.per_object_mean and object_ids is not None:
                has = self.obj_has_real[object_ids]
                o_s = getattr(self, f"t{t}_obj_mu_s")[object_ids]
                o_r = getattr(self, f"t{t}_obj_mu_r")[object_ids]
                mu_s = torch.where(has[:, None], o_s, mu_s.expand(B, -1)).unsqueeze(1)
                mu_r = torch.where(has[:, None], o_r, mu_r.expand(B, -1)).unsqueeze(1)
            else:
                mu_s = mu_s.view(1, 1, -1)
                mu_r = mu_r.view(1, 1, -1)
            A = getattr(self, f"t{t}_A")
            y = (x.float() - mu_s) @ A + mu_r
            out.append(torch.where(sim_mask[:, None, None], y.to(x.dtype), x))
        return out


class VisuoTactileModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.trunk_dim = cfg.trunk_dim                              # 768
        # num_spatial derived from encoder (196 for p16@224, 256 for p14@224)
        # Overridable via cfg for backward compat, but defaults to encoder value.

        # ---- Encoder (frozen) ----
        enc_cfg = dict(cfg.encoder)
        if enc_cfg.get("multiscale_layers", None) is None:
            enc_cfg["multiscale_layers"] = list(cfg.heads.dpt.encoder_tap_layers)
        self.tactile_encoder = build_encoder(enc_cfg, self.image_size)

        rgb_enc_cfg = cfg.get("rgb_encoder", None)
        if rgb_enc_cfg is not None:
            rgb_enc_cfg = dict(rgb_enc_cfg)
            if rgb_enc_cfg.get("multiscale_layers", None) is None:
                rgb_enc_cfg["multiscale_layers"] = list(cfg.heads.dpt.encoder_tap_layers)
            self.rgb_encoder = build_encoder(rgb_enc_cfg, self.image_size)
        elif enc_cfg.get("share_encoder_weights", True):
            self.rgb_encoder = self.tactile_encoder
        else:
            self.rgb_encoder = build_encoder(enc_cfg, self.image_size)

        self.enc_dim = self.tactile_encoder.embed_dim
        self.rgb_enc_dim = self.rgb_encoder.embed_dim
        self.num_spatial = self.tactile_encoder.num_patches          # 196 or 256

        # ---- Pose path: projection (enc_dim→768) + trunk ----
        self.tactile_proj = BranchProjection(self.enc_dim, self.trunk_dim)
        self.rgb_proj = BranchProjection(self.rgb_enc_dim, self.trunk_dim)
        self.spatial_pos = SpatialPosEmbedding(self.num_spatial, self.trunk_dim)
        self.use_rgb_pos = cfg.projection.get("rgb_positional_embedding", False)
        if self.use_rgb_pos:
            self.rgb_spatial_pos = SpatialPosEmbedding(self.num_spatial, self.trunk_dim)

        self.spatial_mask = nn.Parameter(torch.zeros(1, self.num_spatial, self.trunk_dim))
        self.pose_mask = nn.Parameter(torch.zeros(1, 1, self.trunk_dim))
        nn.init.trunc_normal_(self.spatial_mask, std=0.02)
        nn.init.trunc_normal_(self.pose_mask, std=0.02)

        self.trunk = FusionTrunk(cfg.fusion_trunk, self.trunk_dim)

        # ---- Optional object embedding (for sim+real co-training) ----
        self.use_obj_emb = cfg.tokens.get("object_embedding", False)
        if self.use_obj_emb:
            num_obj = cfg.tokens.get("num_objects", 20)
            self.obj_embedding = nn.Embedding(num_obj, self.trunk_dim)

        # ---- Optional domain embedding (sim=0, real=1) ----
        self.use_domain_emb = cfg.tokens.get("domain_embedding", False)
        if self.use_domain_emb:
            self.domain_embedding = nn.Embedding(2, self.trunk_dim)

        # ---- Optional CORAL sim→real feature alignment (pose path only) ----
        self.coral_tac = self.coral_rgb = None
        self.coral_dpt_tac = self.coral_dpt_rgb = None
        coral_cfg = cfg.get("coral", None)
        if coral_cfg is not None and coral_cfg.get("enabled", False):
            stats = torch.load(coral_cfg.get("stats_path"),
                               map_location="cpu", weights_only=True)
            pom = coral_cfg.get("per_object_mean", True)
            branches = coral_cfg.get("branches", ["tactile", "rgb"])
            if "tactile" in branches:
                self.coral_tac = CoralAdapter(stats["tactile"], per_object_mean=pom)
            if "rgb" in branches:
                self.coral_rgb = CoralAdapter(stats["rgb"], per_object_mean=pom)
            print(f"  [coral] sim→real feature alignment ON "
                  f"(branches={list(branches)}, per_object_mean={pom})")
            if coral_cfg.get("dpt_taps", False):
                dstats = torch.load(coral_cfg.get("dpt_stats_path"),
                                    map_location="cpu", weights_only=True)
                self.coral_dpt_tac = CoralTapAdapter(
                    dstats["tactile"], dstats["obj_has_real"], per_object_mean=pom)
                self.coral_dpt_rgb = CoralTapAdapter(
                    dstats["rgb"], dstats["obj_has_real"], per_object_mean=pom)
                print("  [coral] DPT tap alignment ON")

        # ---- DPT path: direct encoder taps + RGB injection ----
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
        """RGB -> pose memory M = [B, N+1, trunk_dim]."""
        patch = self.rgb_proj(rgb_patch)
        if rgb_cls is not None:
            cls = self.rgb_proj(rgb_cls)
        else:
            cls = self.pose_mask.expand(patch.shape[0], -1, -1)
        if self.use_rgb_pos:
            patch = self.rgb_spatial_pos(patch)
        return torch.cat([patch, cls], dim=1)

    def _build_pose_queries(self, tac_patch, tac_cls, use_tactile, B, device):
        """Build N+1 pose queries at trunk_dim."""
        if use_tactile:
            spatial = self.spatial_pos(self.tactile_proj(tac_patch))
            if tac_cls is not None:
                pose_q = self.tactile_proj(tac_cls)
            else:
                pose_q = self.pose_mask.expand(B, -1, -1)
        else:
            spatial = self.spatial_pos(self.spatial_mask.expand(B, -1, -1))
            pose_q = self.pose_mask.expand(B, -1, -1)
        return torch.cat([spatial, pose_q], dim=1)

    # ------------------------------------------------------------------ forward

    def forward(self, rgb, tactile, config="both", inject_rgb_to_dpt=None,
                encoder_cache=None, object_ids=None, domain_ids=None):
        """
        Args:
            config: modality config for the POSE path ("both"/"tactile"/"rgb").
            inject_rgb_to_dpt: whether to inject RGB into DPT taps.
                None = auto (inject when config has RGB).
                Can be overridden for decoupled dropout training.
            encoder_cache: pre-computed encoder outputs (skips frozen forward).
            object_ids: (B,) int tensor of object class indices (for co-training).
            domain_ids: (B,) int tensor (0=sim, 1=real) for domain embedding.
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

        # ---- CORAL: move sim features onto the real distribution (pose path only;
        #      DPT taps below stay raw) ----
        if domain_ids is not None:
            if self.coral_tac is not None and tac_patch is not None:
                tac_patch, tac_cls = self.coral_tac(
                    tac_patch, tac_cls, domain_ids, object_ids)
            if self.coral_rgb is not None and rgb_patch is not None:
                rgb_patch, rgb_cls = self.coral_rgb(
                    rgb_patch, rgb_cls, domain_ids, object_ids)

        # ---- Pose path: projection(768) → trunk → pose head ----
        pose_memory = self._build_pose_memory(rgb_patch, rgb_cls) if use_rgb else None
        pose_queries = self._build_pose_queries(
            tac_patch, tac_cls, use_tactile, B, device)

        if self.use_obj_emb and object_ids is not None:
            obj_emb = self.obj_embedding(object_ids).unsqueeze(1)  # (B, 1, D)
            pose_queries = pose_queries + obj_emb

        if self.use_domain_emb and domain_ids is not None:
            dom_emb = self.domain_embedding(domain_ids).unsqueeze(1)  # (B, 1, D)
            pose_queries = pose_queries + dom_emb

        trunk_taps, pose_token, bottleneck = self.trunk(
            pose_queries, pose_memory, use_rgb)

        spatial_queries = trunk_taps[-1]
        pose = self.pose_head(pose_token, spatial_queries=spatial_queries)

        # ---- DPT path: encoder multiscale taps (1024) ----
        tac_ms = encoder_cache.get("tactile_ms") if encoder_cache else None
        rgb_ms = encoder_cache.get("rgb_ms") if encoder_cache else None
        if use_tactile:
            ms = tac_ms if tac_ms is not None else self.tactile_encoder.forward_multiscale(tactile)
            if self.coral_dpt_tac is not None and domain_ids is not None:
                ms = self.coral_dpt_tac(ms, domain_ids, object_ids)
            dpt_taps = [self.dpt_pos(t) for t in ms]
        elif self.rgb_enc_dim == self.enc_dim and self.rgb_encoder.num_patches == self.num_spatial:
            ms = rgb_ms if rgb_ms is not None else self.rgb_encoder.forward_multiscale(rgb)
            if self.coral_dpt_rgb is not None and domain_ids is not None:
                ms = self.coral_dpt_rgb(ms, domain_ids, object_ids)
            dpt_taps = [self.dpt_pos(t) for t in ms]
        else:
            dpt_taps = None

        if dpt_taps is not None:
            if inject_rgb_to_dpt and use_rgb:
                dpt_taps = [inj(t, bottleneck)
                            for inj, t in zip(self.tap_inject, dpt_taps)]
            depth, normal = self.dpt(dpt_taps, out_hw=(self.image_size, self.image_size))
        else:
            depth = torch.zeros(B, 1, self.image_size, self.image_size, device=device)
            normal = torch.zeros(B, 3, self.image_size, self.image_size, device=device)

        out = {"depth": depth, "normal": normal}
        out.update(pose)
        return out


class SingleEncoderModel(nn.Module):
    """Single-encoder baseline: frozen encoder → DPT + PoseHead, no fusion trunk.

    Architecture:
        tactile → encoder (frozen) → CLS [B,1,E] + patches [B,N,E]
                                        ↓                ↓
                                  multiscale taps     CLS + mean(patches)
                                        ↓                ↓
                                   DPT decoder        Pose head
                                  (depth+normal)      (SE(2) pose)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_size = cfg.image_size

        enc_cfg = dict(cfg.encoder)
        if enc_cfg.get("multiscale_layers", None) is None:
            enc_cfg["multiscale_layers"] = list(cfg.heads.dpt.encoder_tap_layers)
        self.encoder = build_encoder(enc_cfg, self.image_size)

        self.enc_dim = self.encoder.embed_dim
        self.num_spatial = self.encoder.num_patches

        self.use_obj_emb = cfg.tokens.get("object_embedding", False)
        if self.use_obj_emb:
            num_obj = cfg.tokens.get("num_objects", 20)
            self.obj_embedding = nn.Embedding(num_obj, self.enc_dim)

        self.dpt_pos = SpatialPosEmbedding(self.num_spatial, self.enc_dim)

        self.dpt = DPTHead(
            embed_dim=self.enc_dim,
            features=cfg.heads.dpt.features,
            dropout=cfg.heads.dpt.dropout,
            out_depth_channels=cfg.heads.dpt.out_depth_channels,
            out_normal_channels=cfg.heads.dpt.out_normal_channels,
        )
        self.pose_head = PoseHead(
            dim=self.enc_dim,
            hidden_dim=cfg.heads.pose.hidden_dim,
            dropout=cfg.heads.pose.dropout,
            pose_mode=cfg.heads.pose.pose_mode,
            rot_num_bins=cfg.heads.pose.get("rot_num_bins", 72),
            use_spatial_pool=cfg.heads.pose.get("use_spatial_pool", True),
        )

    @property
    def tactile_encoder(self):
        return self.encoder

    @property
    def rgb_encoder(self):
        return self.encoder

    def forward(self, rgb, tactile, config="both", inject_rgb_to_dpt=None,
                encoder_cache=None, object_ids=None):
        B, device = tactile.shape[0], tactile.device

        tac_enc = encoder_cache.get("tactile") if encoder_cache else None
        tac_patch, tac_cls = tac_enc if tac_enc else self.encoder(tactile)

        if self.use_obj_emb and object_ids is not None:
            obj_emb = self.obj_embedding(object_ids).unsqueeze(1)
            tac_patch = tac_patch + obj_emb
            if tac_cls is not None:
                tac_cls = tac_cls + obj_emb

        tac_ms = encoder_cache.get("tactile_ms") if encoder_cache else None
        ms = tac_ms if tac_ms is not None else self.encoder.forward_multiscale(tactile)
        dpt_taps = [self.dpt_pos(t) for t in ms]
        depth, normal = self.dpt(dpt_taps, out_hw=(self.image_size, self.image_size))

        pose = self.pose_head(tac_cls, spatial_queries=tac_patch)

        out = {"depth": depth, "normal": normal}
        out.update(pose)
        return out


def build_model(cfg):
    model_type = cfg.get("model_type", "visuo_tactile")
    if model_type == "single_encoder":
        return SingleEncoderModel(cfg)
    if model_type == "mvitac":
        from .mvitac import MViTacPoseModel
        mc = cfg.mvitac
        return MViTacPoseModel(
            tactile_ckpt=mc.tactile_ckpt,
            vision_ckpt=mc.vision_ckpt,
            hidden_dim=cfg.heads.pose.hidden_dim,
            dropout=cfg.heads.pose.dropout,
            num_objects=cfg.tokens.get("num_objects", 20),
            use_obj_emb=cfg.tokens.get("object_embedding", True),
        )
    return VisuoTactileModel(cfg)
