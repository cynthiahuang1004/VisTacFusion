"""Inference on real or sim data — load a checkpoint, predict depth/normal/pose, visualize.

Usage:
    python -m vistacfusion.engine.inference \
        --checkpoint outputs/best.pt \
        --tactile /path/to/tactile.png \
        --rgb /path/to/rgb.png \
        --output-dir results/

    # Batch mode: run on a full session directory
    python -m vistacfusion.engine.inference \
        --checkpoint outputs/best.pt \
        --session-dir /path/to/session/sensor_0000/ \
        --output-dir results/

    # Evaluate on sim val set with GT comparison
    python -m vistacfusion.engine.inference \
        --checkpoint outputs/best.pt \
        --eval-sim \
        --output-dir results/
"""
from __future__ import annotations

import argparse
import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..models.model import build_model
from ..utils.config import merge_configs


def load_model(cfg, checkpoint_path, device):
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (pose head architecture change?)")
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def preprocess_image(img_path, image_size):
    """Load image → same preprocessing as training: center square crop (non-square
    inputs), the FIXED 1/sqrt(2) crop the model was trained in, then ImageNet norm."""
    from ..data.transforms import fixed_center_crop

    img = np.array(Image.open(img_path), dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # non-square (e.g. real 640x480): center square crop first, no aspect distortion
    h, w = img.shape[:2]
    if h != w:
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]

    # the model lives in the fixed-crop world — apply the SAME crop as training
    img = fixed_center_crop(img, out_size=image_size)

    t = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1)
    mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0)


def _extract_outputs(out):
    """Extract depth, normal, pose from model output dict."""
    depth = out["depth"][0, 0].cpu().numpy()
    normal = out["normal"][0].cpu().permute(1, 2, 0).numpy()
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8)
    pose = out.get("se2", out.get("trans"))
    if pose is not None:
        pose = pose[0].cpu().numpy()
    else:
        pose = np.zeros(4)
    theta = np.degrees(np.arctan2(pose[1], pose[0]))
    return depth, normal, pose, theta


def predict(model, rgb_tensor, tactile_tensor, config="both", device="cpu",
            pose_model=None):
    """Run inference. If pose_model is given, use it for pose and model for depth/normal."""
    rgb = rgb_tensor.to(device)
    tac = tactile_tensor.to(device)

    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        out = model(rgb, tac, config=config)
        if pose_model is not None:
            pose_out = pose_model(rgb, tac, config=config)
            out["se2"] = pose_out.get("se2", pose_out.get("trans"))

    return _extract_outputs(out)


CONFIGS_3 = ["both", "tactile", "rgb"]
CONFIG_LABELS = {"both": "Both (RGB+Tactile)", "tactile": "Tactile Only", "rgb": "RGB Only"}


def center_crop_square(img):
    """Center-crop a HxW(xC) image to a square."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0:y0+s, x0:x0+s]


def visualize_three_configs(tactile_img, rgb_img, results,
                            gt_depth=None, gt_normal=None, gt_pose=None,
                            save_path=None):
    """Plot 3-config comparison: rows = [both, tactile, rgb, (GT)], cols = [input, depth, normal, pose]."""
    has_gt = gt_depth is not None
    nrows = 4 if has_gt else 3
    fig, axes = plt.subplots(nrows, 4, figsize=(18, 4.5 * nrows))

    for row, cfg_name in enumerate(CONFIGS_3):
        depth, normal, pose, theta = results[cfg_name]
        normal_vis = (normal * 0.5 + 0.5).clip(0, 1)

        axes[row, 0].imshow(tactile_img if cfg_name != "rgb" else rgb_img)
        axes[row, 0].set_ylabel(CONFIG_LABELS[cfg_name], fontsize=12, fontweight="bold")
        axes[row, 0].set_title("Input" if row == 0 else "")

        axes[row, 1].imshow(depth, cmap="viridis")
        axes[row, 1].set_title("Depth" if row == 0 else "")

        axes[row, 2].imshow(normal_vis)
        axes[row, 2].set_title("Normal" if row == 0 else "")

        pose_text = f"θ = {theta:.1f}°\ntx = {pose[2]:.3f}\nty = {pose[3]:.3f}"
        axes[row, 3].text(0.5, 0.5, pose_text,
                          transform=axes[row, 3].transAxes, fontsize=18,
                          ha="center", va="center", family="monospace")
        axes[row, 3].set_title("Pose" if row == 0 else "")
        axes[row, 3].axis("off")

    if has_gt:
        axes[3, 0].imshow(tactile_img)
        axes[3, 0].set_ylabel("Ground Truth", fontsize=12, fontweight="bold")
        axes[3, 1].imshow(gt_depth, cmap="viridis")
        gt_normal_vis = (gt_normal * 0.5 + 0.5).clip(0, 1)
        axes[3, 2].imshow(gt_normal_vis)
        if gt_pose is not None:
            gt_theta = np.degrees(np.arctan2(gt_pose[1], gt_pose[0]))
            gt_text = f"θ = {gt_theta:.1f}°\ntx = {gt_pose[2]:.3f}\nty = {gt_pose[3]:.3f}"
            axes[3, 3].text(0.5, 0.5, gt_text,
                            transform=axes[3, 3].transAxes, fontsize=18,
                            ha="center", va="center", family="monospace",
                            color="green")
        axes[3, 3].axis("off")

    for ax in axes.flat:
        if ax.images or ax.texts:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved: {save_path}")
    plt.close(fig)


def run_single(model, rgb_path, tactile_path, cfg, device, output_dir,
               gt_depth=None, gt_normal=None, name="sample", pose_model=None, **kwargs):
    """Run inference on a single pair with all 3 configs."""
    image_size = cfg.image_size
    rgb_t = preprocess_image(rgb_path, image_size)
    tac_t = preprocess_image(tactile_path, image_size)

    results = {}
    for config in CONFIGS_3:
        depth, normal, pose, theta = predict(model, rgb_t, tac_t, config=config,
                                             device=device, pose_model=pose_model)
        results[config] = (depth, normal, pose, theta)
        np.save(osp.join(output_dir, f"{name}_{config}_depth.npy"), depth)
        np.save(osp.join(output_dir, f"{name}_{config}_pose.npy"), pose)

    tactile_img = center_crop_square(np.array(Image.open(tactile_path)))
    rgb_img = center_crop_square(np.array(Image.open(rgb_path)))

    save_path = osp.join(output_dir, f"{name}.png")
    visualize_three_configs(tactile_img, rgb_img, results,
                            gt_depth=gt_depth, gt_normal=gt_normal, save_path=save_path)

    for cfg_name in CONFIGS_3:
        _, _, pose, theta = results[cfg_name]
        print(f"  {name} [{cfg_name:8s}]: θ={theta:.1f}°  tx={pose[2]:.4f}  ty={pose[3]:.4f}")

    return results


def run_session(model, session_dir, cfg, device, output_dir, max_samples=None,
                pose_model=None):
    """Run inference on all samples in a session directory."""
    samples_dir = osp.join(session_dir, "samples")
    rgb_dir = osp.join(session_dir, "rgb")
    raw_dir = osp.join(session_dir, "raw_data")

    pngs = sorted(f for f in os.listdir(samples_dir) if f.endswith(".png"))
    if max_samples:
        pngs = pngs[:max_samples]

    print(f"Running inference on {len(pngs)} samples from {session_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for png in pngs:
        idx = int(osp.splitext(png)[0])
        tactile_path = osp.join(samples_dir, png)
        rgb_path = osp.join(rgb_dir, png)

        if not osp.exists(rgb_path):
            print(f"  skipping {png}: no RGB")
            continue

        gt_depth = gt_normal = None
        gt_depth_path = osp.join(raw_dir, f"{idx:04d}_gt.npy")
        if osp.exists(gt_depth_path):
            from ..data.dataset import depth_to_normal
            from ..data.transforms import FIXED_CROP, fixed_center_crop
            gt_depth_raw = fixed_center_crop(
                np.load(gt_depth_path).astype(np.float32))
            gt_depth = gt_depth_raw * 1000.0
            # fixed crop world: view = 17.5mm * (1/sqrt2), constant pixel size
            px = cfg.sim.get("gel_view_m", 0.017502) * FIXED_CROP / cfg.image_size
            gt_normal = depth_to_normal(gt_depth_raw, px, px)

        run_single(model, rgb_path, tactile_path, cfg, device, output_dir,
                   gt_depth=gt_depth, gt_normal=gt_normal, name=f"{idx:04d}",
                   pose_model=pose_model)


def run_real_dir(model, real_dir, cfg, device, output_dir, max_samples=None,
                 pose_model=None):
    """Run inference on a real data directory with rgb_images/ and tactile_images/."""
    rgb_dir = osp.join(real_dir, "rgb_images")
    tac_dir = osp.join(real_dir, "tactile_images")

    rgb_files = {osp.splitext(f)[0]: f for f in os.listdir(rgb_dir)}
    tac_files = {osp.splitext(f)[0]: f for f in os.listdir(tac_dir)}
    common = sorted(set(rgb_files) & set(tac_files), key=lambda x: int(x) if x.isdigit() else x)

    if max_samples:
        common = common[:max_samples]

    print(f"Running real inference on {len(common)} paired samples from {real_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for name in common:
        rgb_path = osp.join(rgb_dir, rgb_files[name])
        tac_path = osp.join(tac_dir, tac_files[name])
        run_single(model, rgb_path, tac_path, cfg, device, output_dir, name=name,
                   pose_model=pose_model)


def _angular_error_deg(pred_normal, gt_normal):
    """Compute per-pixel angular error in degrees between predicted and GT normals.
    Both are (H, W, 3) unit vectors in [-1, 1]. Returns mean and median in degrees."""
    p = pred_normal / (np.linalg.norm(pred_normal, axis=-1, keepdims=True) + 1e-8)
    g = gt_normal / (np.linalg.norm(gt_normal, axis=-1, keepdims=True) + 1e-8)
    cos_sim = np.clip((p * g).sum(axis=-1), -1, 1)
    angles = np.degrees(np.arccos(cos_sim))
    return float(np.mean(angles)), float(np.median(angles))


def _depth_metrics(pred, gt):
    """Compute depth metrics. Full-image MSE (matches training eval); contact-only
    MAE/RMSE/delta<1.25 for paper reporting. Both inputs are (H, W) arrays."""
    mse_full = float(((pred - gt) ** 2).mean())
    mask = gt > 0
    contact = {}
    if mask.sum() > 0:
        p, g = pred[mask], gt[mask]
        abs_diff = np.abs(p - g)
        contact["MAE"] = float(abs_diff.mean())
        contact["RMSE"] = float((abs_diff ** 2).mean()) ** 0.5
        ratio = np.maximum(p / np.clip(g, 1e-6, None), g / np.clip(p, 1e-6, None))
        contact["delta<1.25"] = float((ratio < 1.25).mean()) * 100
    return {"MSE": mse_full, **contact}


def _rotation_error_deg(pred_pose, gt_pose):
    """Compute rotation error in degrees between pred and gt pose (cos,sin,tx,ty)."""
    theta_pred = np.arctan2(pred_pose[1], pred_pose[0])
    theta_gt = np.arctan2(gt_pose[1], gt_pose[0])
    err = abs(theta_pred - theta_gt)
    err = min(err, 2 * np.pi - err)
    return float(np.degrees(err))


def run_eval_sim(model, cfg, device, output_dir, num_vis=20, pose_model=None,
                 ckpt_info=None):
    """Run full quantitative eval on sim val set + one visualization per object.

    Uses batched DataLoader + encoder cache for speed (~3 min vs ~30 min).

    Outputs:
      - eval_metrics.txt: quantitative metrics for all 3 configs
      - sim_val_vis/<object_name>.png: one representative sample per object
    """
    import math as _math
    from torch.utils.data import DataLoader
    from ..data.dataset import build_datasets, SimVisuoTactileDataset
    from .eval import precompute_encoder_cache, _slice_cache
    _, val_ds = build_datasets(cfg)

    os.makedirs(output_dir, exist_ok=True)

    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])

    # --- Group samples by object, pick one representative per object for vis ---
    obj_vis_idx = {}
    if isinstance(val_ds, SimVisuoTactileDataset):
        for i, (unit, _sample_idx) in enumerate(val_ds.samples):
            obj_name = osp.basename(osp.dirname(osp.dirname(unit)))
            if obj_name not in obj_vis_idx:
                obj_vis_idx[obj_name] = i
    else:
        for i in range(min(len(val_ds), 20)):
            obj_vis_idx[f"sample_{i:03d}"] = i

    vis_indices = set(obj_vis_idx.values())

    # --- Pre-compute encoder cache (run frozen encoder once) ---
    eval_model = pose_model if pose_model is not None else model
    batch_size = 32
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print("Pre-computing encoder cache for val set...")
    encoder_cache = precompute_encoder_cache(eval_model, val_loader, device)

    # --- Batched evaluation with cache ---
    metrics = {cfg_name: {
        "depth_mse": 0.0, "normal_mse": 0.0,
        "depth_mae": [], "depth_rmse": [], "depth_d1": [],
        "normal_ang_mean": 0.0, "normal_ang_median": 0.0,
        "pose_rot_deg": 0.0, "pose_trans_l1": 0.0, "n": 0,
    } for cfg_name in CONFIGS_3}

    vis_results = {}

    print(f"Evaluating {len(val_ds)} val samples (3 configs, batched)...")
    sample_idx = 0
    for batch in val_loader:
        bs = batch["rgb"].shape[0]
        batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
        batch_enc = _slice_cache(encoder_cache, sample_idx, sample_idx + bs, device)

        for cfg_name in CONFIGS_3:
            with torch.no_grad(), torch.autocast(device_type=device.type,
                                                  enabled=device.type == "cuda"):
                out = model(batch_dev["rgb"], batch_dev["tactile"],
                            config=cfg_name, encoder_cache=batch_enc)
                if pose_model is not None:
                    pose_out = pose_model(batch_dev["rgb"], batch_dev["tactile"],
                                         config=cfg_name, encoder_cache=batch_enc)
                    out["se2"] = pose_out.get("se2", pose_out.get("trans"))

            m = metrics[cfg_name]
            m["depth_mse"] += F.mse_loss(out["depth"], batch_dev["depth"]).item() * bs
            m["normal_mse"] += F.mse_loss(out["normal"], batch_dev["normal"]).item() * bs

            # Per-sample metrics for angular error and pose
            pred_depth = out["depth"].cpu()
            pred_normal = out["normal"].cpu()
            gt_normal = batch["normal"]

            # Angular error (batched)
            p_n = F.normalize(pred_normal.float(), dim=1)
            g_n = F.normalize(gt_normal.float(), dim=1)
            cos_sim = (p_n * g_n).sum(dim=1).clamp(-1, 1)
            angles = torch.acos(cos_sim) * (180.0 / _math.pi)
            m["normal_ang_mean"] += angles.mean().item() * bs
            m["normal_ang_median"] += angles.median().item() * bs

            # Pose rotation error
            if "se2" in out:
                se2 = out["se2"].float().cpu()
                gt_pose = batch["pose"]
                cos_p, sin_p = se2[:, 0], se2[:, 1]
                cos_g, sin_g = gt_pose[:, 0], gt_pose[:, 1]
                theta_pred = torch.atan2(sin_p, cos_p)
                theta_gt = torch.atan2(sin_g, cos_g)
                rot_err = (theta_pred - theta_gt).abs()
                rot_err = torch.min(rot_err, 2 * _math.pi - rot_err)
                m["pose_rot_deg"] += (rot_err * 180.0 / _math.pi).mean().item() * bs
                m["pose_trans_l1"] += F.l1_loss(se2[:, 2:], gt_pose[:, 2:]).item() * bs

            # Contact-only depth metrics (per-sample)
            for j in range(bs):
                pd = pred_depth[j, 0].numpy()
                gd = batch["depth"][j, 0].numpy()
                mask = gd > 0
                if mask.sum() > 0:
                    p, g = pd[mask], gd[mask]
                    abs_diff = np.abs(p - g)
                    m["depth_mae"].append(float(abs_diff.mean()))
                    m["depth_rmse"].append(float((abs_diff ** 2).mean()) ** 0.5)
                    ratio = np.maximum(p / np.clip(g, 1e-6, None),
                                       g / np.clip(p, 1e-6, None))
                    m["depth_d1"].append(float((ratio < 1.25).mean()) * 100)

            # Save vis data for representative samples
            for j in range(bs):
                global_idx = sample_idx + j
                if global_idx in vis_indices:
                    d = out["depth"][j, 0].cpu().numpy()
                    n = out["normal"][j].cpu().permute(1, 2, 0).numpy()
                    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
                    if "se2" in out:
                        p = out["se2"][j].float().cpu().numpy()
                    else:
                        p = np.zeros(4)
                    th = float(np.degrees(np.arctan2(p[1], p[0])))
                    vis_results.setdefault(global_idx, {})[cfg_name] = (d, n, p, th)

            m["n"] += bs

        sample_idx += bs
        if sample_idx % (batch_size * 10) == 0:
            print(f"  {sample_idx}/{len(val_ds)} samples done")

    print(f"  {len(val_ds)}/{len(val_ds)} samples done")

    # --- Aggregate metrics ---
    summary = {}
    for cfg_name in CONFIGS_3:
        m = metrics[cfg_name]
        n = max(1, m["n"])
        summary[cfg_name] = {
            "depth_mse": m["depth_mse"] / n,
            "normal_mse": m["normal_mse"] / n,
            "depth_mae": float(np.mean(m["depth_mae"])) if m["depth_mae"] else float("nan"),
            "depth_rmse": float(np.mean(m["depth_rmse"])) if m["depth_rmse"] else float("nan"),
            "depth_d1": float(np.mean(m["depth_d1"])) if m["depth_d1"] else float("nan"),
            "normal_ang_mean": m["normal_ang_mean"] / n,
            "normal_ang_median": m["normal_ang_median"] / n,
            "pose_rot_deg": m["pose_rot_deg"] / n,
            "pose_trans_l1": m["pose_trans_l1"] / n,
        }

    # --- Write eval_metrics.txt ---
    metrics_dir = osp.dirname(output_dir)
    metrics_path = osp.join(metrics_dir, "eval_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("VisTacFusion Evaluation Results\n")
        if ckpt_info:
            f.write(f"Checkpoint: {ckpt_info}\n")
        f.write(f"Val samples: {len(val_ds)}\n\n")
        for cfg_name in CONFIGS_3:
            s = summary[cfg_name]
            f.write(f"[{cfg_name}]\n")
            f.write(f"  Depth MSE:               {s['depth_mse']:.6f}\n")
            f.write(f"  Depth MAE:               {s['depth_mae']:.6f}\n")
            f.write(f"  Depth RMSE:              {s['depth_rmse']:.6f}\n")
            f.write(f"  Depth delta<1.25 (%):    {s['depth_d1']:.2f}\n")
            f.write(f"  Normal MSE:              {s['normal_mse']:.6f}\n")
            f.write(f"  Normal Ang Mean (deg):   {s['normal_ang_mean']:.2f}\n")
            f.write(f"  Normal Ang Median (deg): {s['normal_ang_median']:.2f}\n")
            f.write(f"  Pose Rot Error (deg):    {s['pose_rot_deg']:.2f}\n")
            f.write(f"  Pose Trans L1:           {s['pose_trans_l1']:.6f}\n")
            f.write("\n")

    print(f"\nMetrics saved -> {metrics_path}")

    # --- Print summary table ---
    print("\n" + "=" * 72)
    print(f"{'Metric':30s} {'Both':>12s} {'Tactile':>12s} {'RGB':>12s}")
    print("=" * 72)
    for key in ["depth_mse", "depth_mae", "depth_rmse", "depth_d1",
                "normal_mse", "normal_ang_mean", "normal_ang_median",
                "pose_rot_deg", "pose_trans_l1"]:
        vals = [summary[c].get(key, float("nan")) for c in CONFIGS_3]
        fmt = ".2f" if "deg" in key or "d1" in key else ".6f"
        print(f"  {key:28s} {vals[0]:{fmt}} {vals[1]:{fmt}} {vals[2]:{fmt}}")
    print("=" * 72)

    # --- One visualization per object ---
    print(f"\nSaving one visualization per object ({len(obj_vis_idx)} objects)...")
    for obj_name, idx in sorted(obj_vis_idx.items()):
        sample = val_ds[idx]
        results = vis_results.get(idx, {})
        if not results:
            continue

        gt_depth = sample["depth"][0].numpy()
        gt_normal = sample["normal"].permute(1, 2, 0).numpy()
        gt_pose = sample["pose"].numpy()
        tac_img = (sample["tactile"].permute(1, 2, 0).numpy() * std + mean).clip(0, 255).astype(np.uint8)
        rgb_img = (sample["rgb"].permute(1, 2, 0).numpy() * std + mean).clip(0, 255).astype(np.uint8)

        save_path = osp.join(output_dir, f"{obj_name}.png")
        visualize_three_configs(tac_img, rgb_img, results,
                                gt_depth=gt_depth, gt_normal=gt_normal,
                                gt_pose=gt_pose, save_path=save_path)
    print(f"Visualizations saved -> {output_dir}/")


def main():
    ap = argparse.ArgumentParser(description="Inference on real/sim data")
    ap.add_argument("--train-dir", required=True,
                    help="Training output directory (e.g. outputs/20260629_022120)")
    ap.add_argument("--checkpoint", default=None,
                    help="Single checkpoint file (overrides best_depth/best_pose)")
    ap.add_argument("--model", default="configs/model.yaml")
    ap.add_argument("--train", default="configs/train.yaml")
    ap.add_argument("--data", default="configs/data.yaml")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tactile", help="Single tactile image path (pair with --rgb)")
    mode.add_argument("--session-dir", help="Session sensor directory for batch inference (sim)")
    mode.add_argument("--real-dir", help="Real data directory with rgb_images/ and tactile_images/")
    mode.add_argument("--eval-sim", action="store_true", help="Evaluate on sim val set")
    mode.add_argument("--eval-all", action="store_true",
                      help="Run both sim val + real data")

    ap.add_argument("--rgb", help="Single RGB image path (pair with --tactile)")
    ap.add_argument("--real-dir-path",
                    default="/media/hdd2/ihsuan/VisTacFusion/datasets/real_data",
                    help="Real data path for --eval-all mode")
    ap.add_argument("--suffix", default="best",
                    help="Output folder suffix (e.g. best, last, epoch050)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--num-vis", type=int, default=20)
    ap.add_argument("--device", default=None,
                    help="Device (e.g. cuda:2). Default: first available GPU.")
    args = ap.parse_args()

    cfg = merge_configs(args.model, args.train, args.data)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_name = osp.basename(args.train_dir.rstrip("/"))
    base_dir = osp.join("eval_results", f"{train_name}_{args.suffix}")

    if args.checkpoint:
        depth_model = load_model(cfg, args.checkpoint, device)
        pose_model = None
        ckpt_info = args.checkpoint
        print(f"Using single checkpoint: {args.checkpoint}")
    else:
        depth_ckpt = osp.join(args.train_dir, "best_depth.pt")
        pose_ckpt = osp.join(args.train_dir, "best_pose.pt")

        if not osp.exists(depth_ckpt):
            fallback = osp.join(args.train_dir, "best.pt")
            if osp.exists(fallback):
                depth_ckpt = fallback
            else:
                raise FileNotFoundError(f"No best_depth.pt or best.pt in {args.train_dir}")

        depth_model = load_model(cfg, depth_ckpt, device)
        pose_model = None
        if osp.exists(pose_ckpt) and pose_ckpt != depth_ckpt:
            pose_model = load_model(cfg, pose_ckpt, device)
            ckpt_info = f"depth={depth_ckpt}, pose={pose_ckpt}"
            print(f"Using split checkpoints: depth={osp.basename(depth_ckpt)}, pose={osp.basename(pose_ckpt)}")
        else:
            ckpt_info = depth_ckpt
            print(f"Using single checkpoint: {osp.basename(depth_ckpt)}")

    if args.tactile:
        if not args.rgb:
            ap.error("--rgb is required when using --tactile")
        out = osp.join(base_dir, "single")
        os.makedirs(out, exist_ok=True)
        run_single(depth_model, args.rgb, args.tactile, cfg, device, out,
                   name="prediction", pose_model=pose_model)

    elif args.session_dir:
        out = osp.join(base_dir, "session_vis")
        os.makedirs(out, exist_ok=True)
        run_session(depth_model, args.session_dir, cfg, device, out,
                    max_samples=args.max_samples, pose_model=pose_model)

    elif args.real_dir:
        out = osp.join(base_dir, "real_vis")
        os.makedirs(out, exist_ok=True)
        run_real_dir(depth_model, args.real_dir, cfg, device, out,
                     max_samples=args.max_samples, pose_model=pose_model)

    elif args.eval_sim:
        out = osp.join(base_dir, "sim_val_vis")
        os.makedirs(out, exist_ok=True)
        run_eval_sim(depth_model, cfg, device, out, num_vis=args.num_vis,
                     pose_model=pose_model, ckpt_info=ckpt_info)

    elif args.eval_all:
        sim_out = osp.join(base_dir, "sim_val_vis")
        real_out = osp.join(base_dir, "real_vis")
        os.makedirs(sim_out, exist_ok=True)
        os.makedirs(real_out, exist_ok=True)
        run_eval_sim(depth_model, cfg, device, sim_out, num_vis=args.num_vis,
                     pose_model=pose_model, ckpt_info=ckpt_info)
        run_real_dir(depth_model, args.real_dir_path, cfg, device, real_out,
                     max_samples=args.max_samples, pose_model=pose_model)

    print(f"\nResults saved to: {base_dir}/")


if __name__ == "__main__":
    main()
