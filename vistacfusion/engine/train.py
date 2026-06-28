"""Training loop with modality dropout + multi-GPU DDP support.

Single GPU:
    python -m vistacfusion.engine.train --model configs/model.yaml \
           --train configs/train.yaml --data configs/data.yaml

Multi-GPU (e.g. 2 GPUs):
    torchrun --nproc_per_node=2 -m vistacfusion.engine.train \
           --model configs/model.yaml --train configs/train.yaml --data configs/data.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from ..data.dataset import build_datasets
from ..losses.total import MultiTaskLoss
from ..models.model import build_model
from ..utils.config import merge_configs
from ..utils.misc import param_count_str, set_seed
from .eval import evaluate, precompute_encoder_cache


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return not is_distributed() or dist.get_rank() == 0


def setup_distributed():
    if "RANK" not in os.environ:
        return None, 0, 1
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}"), rank, world_size


def sample_config(md_cfg):
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


def sync_config(config, device):
    """Broadcast modality config from rank 0 so all GPUs use the same config per step."""
    if not is_distributed():
        return config
    mapping = {"both": 0, "tactile": 1, "rgb": 2}
    reverse = {0: "both", 1: "tactile", 2: "rgb"}
    t = torch.tensor([mapping[config]], device=device)
    dist.broadcast(t, src=0)
    return reverse[t.item()]


def move_batch(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def build_scheduler(optimizer, cfg, steps_per_epoch):
    sched_cfg = cfg.schedule
    warmup_steps = sched_cfg.get("warmup_steps", 0)
    max_epochs = sched_cfg.max_epochs
    total_steps = max_epochs * steps_per_epoch
    min_lr_ratio = sched_cfg.get("min_lr_ratio", 0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion,
                    cfg, device, epoch, writer=None):
    model.train()
    md_cfg = cfg.modality_dropout
    dense_on_rgb_only = md_cfg.get("rgb_only_supervises_dense", True)
    running = {}
    global_step_base = epoch * len(loader)
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        config = sample_config(md_cfg)
        config = sync_config(config, device)
        supervise_dense = config != "rgb" or dense_on_rgb_only

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            out = model(batch["rgb"], batch["tactile"], config=config)
            gt = {"depth": batch["depth"], "normal": batch["normal"],
                  "pose": batch["pose"], "mask": batch.get("mask")}
            loss, comps = criterion(out, gt, supervise_dense=supervise_dense)

        if torch.isnan(loss) or torch.isinf(loss):
            if is_main_process():
                print(f"  [WARN] NaN/Inf loss at epoch {epoch} step {step}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for k, v in comps.items():
            running[k] = running.get(k, 0.0) + float(v)

        if is_main_process() and step % cfg.log_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            msg = "  ".join(f"{k}={running[k] / (step + 1):.4f}" for k in sorted(running))
            print(f"[epoch {epoch:03d} | step {step:04d}/{len(loader)} | cfg={config:7s} | "
                  f"lr={lr_now:.2e}] {msg}")

        if is_main_process() and writer is not None and step % cfg.log_every == 0:
            gs = global_step_base + step
            for k, v in comps.items():
                writer.add_scalar(f"train_step/{k}", float(v), gs)
            writer.add_scalar("train_step/lr", optimizer.param_groups[0]["lr"], gs)
    return {k: v / len(loader) for k, v in running.items()}


def save_loss_plots(history, plot_dir):
    """Save loss curve PNGs from accumulated training/val history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    # --- Train losses ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Loss", fontsize=14)
    for ax, key, label in [
        (axes[0, 0], "total", "Total"),
        (axes[0, 1], "depth", "Depth (SSI)"),
        (axes[1, 0], "normal", "Normal (cosine)"),
        (axes[1, 1], "pose", "Pose"),
    ]:
        vals = [h["train"].get(key, float("nan")) for h in history]
        ax.plot(epochs, vals, "b-o", markersize=2)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "train_loss.png"), dpi=150)
    plt.close(fig)

    # --- Val metrics per config ---
    for config in ["both", "tactile", "rgb"]:
        if config not in history[0].get("val", {}):
            continue
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Val Metrics — {config}", fontsize=14)
        for ax, key, label in [
            (axes[0, 0], "depth_absrel", "Depth AbsRel"),
            (axes[0, 1], "depth_rmse", "Depth RMSE"),
            (axes[0, 2], "normal_mean_angle", "Normal Angle (°)"),
            (axes[1, 0], "pose_rot_deg", "Pose Rot (°)"),
            (axes[1, 1], "pose_trans", "Pose Trans L1"),
        ]:
            vals = [h["val"].get(config, {}).get(key, float("nan")) for h in history]
            ax.plot(epochs, vals, "r-o", markersize=2)
            ax.set_title(label)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        axes[1, 2].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"val_{config}.png"), dpi=150)
        plt.close(fig)


def build_optimizer(model, optim_cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params, lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay,
        betas=tuple(optim_cfg.betas),
    )


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_metric):
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save({
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_metric": best_metric,
    }, path)
    print(f"  -> checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", float("inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="configs/model.yaml")
    ap.add_argument("--train", default="configs/train.yaml")
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    ap.add_argument("--output-dir", type=str, default="outputs",
                    help="Directory for checkpoints and logs")
    args = ap.parse_args()

    # Distributed setup (no-op when launched without torchrun)
    ddp_device, rank, world_size = setup_distributed()

    cfg = merge_configs(args.model, args.train, args.data)
    set_seed(cfg.seed + rank)

    if ddp_device is not None:
        device = ddp_device
    else:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print(f"Device: {device} | World size: {world_size}")
        os.makedirs(args.output_dir, exist_ok=True)

    train_ds, val_ds = build_datasets(cfg)
    if is_main_process():
        print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    lc = cfg.loader
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed() else None
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=lc.num_workers, pin_memory=lc.pin_memory,
        drop_last=True,
        persistent_workers=lc.persistent_workers and lc.num_workers > 0,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=lc.num_workers, pin_memory=lc.pin_memory)

    model = build_model(cfg).to(device)
    if is_main_process():
        print(f"Model: {param_count_str(model)}")

    if is_distributed():
        model = DDP(model, device_ids=[device.index], find_unused_parameters=True)

    criterion = MultiTaskLoss(
        cfg.loss, pose_mode=cfg.heads.pose.pose_mode,
        rot_num_bins=cfg.heads.pose.get("rot_num_bins", 72),
    ).to(device)
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    start_epoch = 0
    best_metric = float("inf")
    if args.resume:
        if is_main_process():
            print(f"Resuming from {args.resume}")
        start_epoch, best_metric = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device)
        start_epoch += 1
        if is_main_process():
            print(f"  resumed at epoch {start_epoch}, best_metric={best_metric:.4f}")

    # Pre-compute frozen encoder outputs for val (no augmentation → deterministic)
    val_enc_cache = None
    writer = None
    history = []
    plot_dir = None
    history_path = None
    if is_main_process():
        raw_model = model.module if isinstance(model, DDP) else model
        print("Pre-computing val encoder cache...")
        val_enc_cache = precompute_encoder_cache(raw_model, val_loader, device)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))
        plot_dir = os.path.join(args.output_dir, "plots")
        history_path = os.path.join(args.output_dir, "history.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)

    max_epochs = args.epochs if args.epochs is not None else cfg.schedule.max_epochs
    for epoch in range(start_epoch, max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion,
            cfg, device, epoch, writer=writer)
        elapsed = time.time() - t0

        if is_main_process():
            print(f"[epoch {epoch:03d}] train done in {elapsed:.0f}s  "
                  f"avg_total={train_metrics.get('total', 0):.4f}")

            for k, v in train_metrics.items():
                writer.add_scalar(f"train_epoch/{k}", v, epoch)

            raw_model = model.module if isinstance(model, DDP) else model
            val_metrics = evaluate(raw_model, val_loader, cfg, device,
                                   encoder_cache=val_enc_cache)
            print(f"[epoch {epoch:03d}] val metrics: {val_metrics}")

            for config_name, metrics in val_metrics.items():
                for mk, mv in metrics.items():
                    writer.add_scalar(f"val_{config_name}/{mk}", mv, epoch)

            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            save_loss_plots(history, plot_dir)

            score = val_metrics.get("both", {}).get("depth_absrel", float("inf"))
            if score < best_metric:
                best_metric = score
                save_checkpoint(
                    os.path.join(args.output_dir, "best.pt"),
                    model, optimizer, scheduler, scaler, epoch, best_metric)
                print(f"  ** new best: depth_absrel={best_metric:.4f}")

            save_checkpoint(
                os.path.join(args.output_dir, f"epoch_{epoch:03d}.pt"),
                model, optimizer, scheduler, scaler, epoch, best_metric)
            writer.flush()

        if is_distributed():
            dist.barrier()

    if is_main_process():
        print("Training complete.")
        if writer is not None:
            writer.close()
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
