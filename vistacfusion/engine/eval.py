"""Evaluation: report metrics per modality config (both / tactile / rgb).

Supports pre-computed encoder cache to skip the frozen DINOv3 forward pass
on every eval call (val has no augmentation → encoder output is deterministic).
"""
from __future__ import annotations

import torch

from . import metrics as M

CONFIGS = ("both", "tactile", "rgb")


@torch.no_grad()
def precompute_encoder_cache(model, loader, device):
    """Run the frozen encoder once on the full val set, return cached features (CPU, fp16)."""
    model.eval()
    tac_p, tac_c, rgb_p, rgb_c = [], [], [], []
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        tactile = batch["tactile"].to(device, non_blocking=True)
        tp, tc = model.tactile_encoder(tactile)
        rp, rc = model.rgb_encoder(rgb)
        tac_p.append(tp.half().cpu())
        tac_c.append(tc.half().cpu())
        rgb_p.append(rp.half().cpu())
        rgb_c.append(rc.half().cpu())
    cache = {
        "tactile_patch": torch.cat(tac_p),
        "tactile_cls": torch.cat(tac_c),
        "rgb_patch": torch.cat(rgb_p),
        "rgb_cls": torch.cat(rgb_c),
    }
    n = cache["tactile_patch"].shape[0]
    mb = sum(v.nbytes for v in cache.values()) / 1024 ** 2
    print(f"  encoder cache: {n} samples, {mb:.0f} MB (fp16, CPU)")
    return cache


def _slice_cache(cache, start, end, device):
    """Slice the flat cache into a per-batch dict for model.forward(encoder_cache=...)."""
    return {
        "tactile": (
            cache["tactile_patch"][start:end].to(device, dtype=torch.float32),
            cache["tactile_cls"][start:end].to(device, dtype=torch.float32),
        ),
        "rgb": (
            cache["rgb_patch"][start:end].to(device, dtype=torch.float32),
            cache["rgb_cls"][start:end].to(device, dtype=torch.float32),
        ),
    }


@torch.no_grad()
def evaluate(model, loader, cfg, device, configs=CONFIGS, encoder_cache=None):
    model.eval()
    report_per_config = cfg.eval.get("report_per_config", True)
    configs = configs if report_per_config else ("both",)

    acc = {c: {"absrel": 0.0, "rmse": 0.0, "nangle": 0.0,
               "rot": 0.0, "trans": 0.0, "n": 0} for c in configs}

    sample_idx = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        bs = batch["rgb"].shape[0]
        mask = batch.get("mask")

        batch_enc = None
        if encoder_cache is not None:
            batch_enc = _slice_cache(encoder_cache, sample_idx, sample_idx + bs, device)
        sample_idx += bs

        for c in configs:
            out = model(batch["rgb"], batch["tactile"], config=c,
                        encoder_cache=batch_enc)
            a = acc[c]
            a["absrel"] += M.depth_absrel(out["depth"], batch["depth"], mask=mask) * bs
            a["rmse"] += M.depth_rmse(out["depth"], batch["depth"], mask=mask) * bs
            a["nangle"] += M.normal_mean_angle_deg(out["normal"], batch["normal"], mask=mask) * bs
            if "se2" in out:
                a["rot"] += M.pose_rot_deg(out["se2"], batch["pose"]) * bs
                a["trans"] += M.pose_trans_l1(out["se2"], batch["pose"]) * bs
            a["n"] += bs

    report = {}
    for c, a in acc.items():
        n = max(1, a["n"])
        report[c] = {
            "depth_absrel": round(a["absrel"] / n, 4),
            "depth_rmse": round(a["rmse"] / n, 4),
            "normal_mean_angle": round(a["nangle"] / n, 3),
            "pose_rot_deg": round(a["rot"] / n, 3),
            "pose_trans": round(a["trans"] / n, 4),
        }
    return report
