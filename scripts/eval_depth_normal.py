"""Eval with use_rendered_normals=False (depth-computed normal GT)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from vistacfusion.utils.config import merge_configs
from vistacfusion.models.model import build_model
from vistacfusion.data.dataset import build_datasets
from vistacfusion.engine.eval import precompute_encoder_cache, _slice_cache
from vistacfusion.engine.train import load_checkpoint

DEPTH_CKPT = "outputs/20260712_050236/best_depth.pt"
POSE_CKPT = "outputs/20260712_050236/best_pose.pt"
CONFIGS_3 = ["both", "tactile", "rgb"]

cfg = merge_configs("configs/model.yaml", "configs/train.yaml", "configs/data.yaml")
cfg.sim.use_rendered_normals = False
device = torch.device("cuda")

print("Loading depth model...")
depth_model = build_model(cfg).to(device)
load_checkpoint(DEPTH_CKPT, depth_model, device=device)
depth_model.eval()

print("Loading pose model...")
pose_model = build_model(cfg).to(device)
load_checkpoint(POSE_CKPT, pose_model, device=device)
pose_model.eval()

_, val_ds = build_datasets(cfg)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

print("Building encoder cache...")
enc_cache = precompute_encoder_cache(depth_model, val_loader, device)

metrics = {c: {
    "depth_mse": 0.0, "normal_mse": 0.0, "normal_mse_norm": 0.0,
    "normal_ang_mean": 0.0, "normal_ang_median": 0.0,
    "pose_rot_l1": 0.0, "pose_rot_deg": 0.0, "pose_trans_l1": 0.0, "n": 0,
} for c in CONFIGS_3}

print(f"Evaluating {len(val_ds)} samples (use_rendered_normals=False)...")
si = 0
for batch in val_loader:
    bs = batch["rgb"].shape[0]
    bd = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    be = _slice_cache(enc_cache, si, si + bs, device)

    for c in CONFIGS_3:
        with torch.no_grad():
            out = depth_model(bd["rgb"], bd["tactile"], config=c, encoder_cache=be)
            po = pose_model(bd["rgb"], bd["tactile"], config=c, encoder_cache=be)
            out["se2"] = po.get("se2", po.get("trans"))

        m = metrics[c]
        m["depth_mse"] += F.mse_loss(out["depth"], bd["depth"]).item() * bs
        m["normal_mse"] += F.mse_loss(out["normal"], bd["normal"]).item() * bs

        pn = F.normalize(out["normal"].cpu().float(), dim=1)
        gn = F.normalize(batch["normal"].float(), dim=1)
        m["normal_mse_norm"] += F.mse_loss(pn, gn).item() * bs
        cs = (pn * gn).sum(dim=1).clamp(-1, 1)
        ang = torch.acos(cs) * (180.0 / math.pi)
        m["normal_ang_mean"] += ang.mean().item() * bs
        m["normal_ang_median"] += ang.median().item() * bs

        if "se2" in out:
            se2 = out["se2"].float().cpu()
            gt = batch["pose"]
            cp, sp = se2[:, 0], se2[:, 1]
            cg, sg = gt[:, 0], gt[:, 1]
            tp = torch.atan2(sp, cp)
            tg = torch.atan2(sg, cg)
            re = (tp - tg).abs()
            re = torch.min(re, 2 * math.pi - re)
            m["pose_rot_l1"] += ((cp - cg).abs() + (sp - sg).abs()).mean().item() * bs
            m["pose_rot_deg"] += (re * 180 / math.pi).mean().item() * bs
            m["pose_trans_l1"] += F.l1_loss(se2[:, 2:], gt[:, 2:]).item() * bs
        m["n"] += bs

    si += bs
    if si % 320 == 0:
        print(f"  {si}/{len(val_ds)} done")

print(f"  {len(val_ds)}/{len(val_ds)} done")
print()
print("=" * 72)
print(f'{"Metric":30s} {"Both":>12s} {"Tactile":>12s} {"RGB":>12s}')
print("=" * 72)
for key in ["depth_mse", "normal_mse", "normal_mse_norm", "normal_ang_mean",
            "normal_ang_median", "pose_rot_l1", "pose_rot_deg", "pose_trans_l1"]:
    vals = [metrics[c][key] / max(1, metrics[c]["n"]) for c in CONFIGS_3]
    fmt = ".2f" if "deg" in key else ".6f"
    print(f"  {key:28s} {vals[0]:{fmt}} {vals[1]:{fmt}} {vals[2]:{fmt}}")
print("=" * 72)
