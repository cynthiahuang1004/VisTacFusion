"""Training loop with modality dropout.

One shared model. Each step samples a config in {both, tactile, rgb} with the configured
probabilities; the three configs are inference-time input modes, trained jointly.

Run:  python -m vistacfusion.engine.train --model configs/model.yaml \
              --train configs/train.yaml --data configs/data.yaml
With dataset:synthetic + encoder.checkpoint:null this runs end-to-end on the mock encoder
(no DINOv3 weights needed).
"""
from __future__ import annotations

import argparse
import random

import torch
from torch.utils.data import DataLoader

from ..data.dataset import build_datasets
from ..losses.total import MultiTaskLoss
from ..models.model import build_model
from ..utils.config import merge_configs
from ..utils.misc import param_count_str, set_seed
from .eval import evaluate


def sample_config(md_cfg):
    """Sample a modality config {both, tactile, rgb} by the dropout probabilities."""
    if not md_cfg.get("enabled", True):
        return "both"
    r = random.random()
    p_both = md_cfg.p_both
    p_tac = md_cfg.p_tactile_only
    if r < p_both:
        return "both"
    if r < p_both + p_tac:
        return "tactile"
    return "rgb"


def move_batch(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, scaler, criterion, cfg, device, epoch):
    model.train()
    md_cfg = cfg.modality_dropout
    dense_on_rgb_only = md_cfg.get("rgb_only_supervises_dense", True)
    running = {}
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        config = sample_config(md_cfg)
        supervise_dense = config != "rgb" or dense_on_rgb_only

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            out = model(batch["rgb"], batch["tactile"], config=config)
            gt = {"depth": batch["depth"], "normal": batch["normal"], "pose": batch["pose"]}
            loss, comps = criterion(out, gt, supervise_dense=supervise_dense)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        for k, v in comps.items():
            running[k] = running.get(k, 0.0) + float(v)
        if step % cfg.log_every == 0:
            msg = "  ".join(f"{k}={running[k] / (step + 1):.4f}" for k in sorted(running))
            print(f"[epoch {epoch:03d} | step {step:04d}/{len(loader)} | cfg={config:7s}] {msg}")
    return {k: v / len(loader) for k, v in running.items()}


def build_optimizer(model, optim_cfg):
    # Encoders are frozen; only trainable params get optimized.
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params, lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay,
        betas=tuple(optim_cfg.betas),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="configs/model.yaml")
    ap.add_argument("--train", default="configs/train.yaml")
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = merge_configs(args.model, args.train, args.data)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds = build_datasets(cfg)
    lc = cfg.loader
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=lc.num_workers, pin_memory=lc.pin_memory,
                              drop_last=True,
                              persistent_workers=lc.persistent_workers and lc.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=lc.num_workers, pin_memory=lc.pin_memory)

    model = build_model(cfg).to(device)
    print(f"Model: {param_count_str(model)}")

    criterion = MultiTaskLoss(
        cfg.loss, pose_mode=cfg.heads.pose.pose_mode,
        rot_num_bins=cfg.heads.pose.get("rot_num_bins", 72),
    ).to(device)
    optimizer = build_optimizer(model, cfg.optim)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    max_epochs = args.epochs if args.epochs is not None else cfg.schedule.max_epochs
    for epoch in range(max_epochs):
        train_one_epoch(model, train_loader, optimizer, scaler, criterion, cfg, device, epoch)
        if (epoch + 1) % cfg.ckpt_every_epochs == 0 or epoch == max_epochs - 1:
            metrics = evaluate(model, val_loader, cfg, device)
            print(f"[epoch {epoch:03d}] val metrics: {metrics}")


if __name__ == "__main__":
    main()
