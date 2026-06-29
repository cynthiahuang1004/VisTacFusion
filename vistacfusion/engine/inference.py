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
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def preprocess_image(img_path, image_size):
    """Load PNG → ImageNet-normalized tensor [1, 3, H, W]."""
    img = np.array(Image.open(img_path), dtype=np.float32)
    t = torch.from_numpy(np.ascontiguousarray(img)).float()
    if t.ndim == 2:
        t = t.unsqueeze(-1).expand(-1, -1, 3)
    t = t.permute(2, 0, 1)
    if t.shape[1] != image_size or t.shape[2] != image_size:
        t = F.interpolate(t.unsqueeze(0), size=(image_size, image_size),
                          mode="bilinear", align_corners=False).squeeze(0)
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
                            gt_depth=None, gt_normal=None, save_path=None):
    """Plot 3-config comparison: rows = [both, tactile, rgb, (GT)], cols = [input, depth, normal]."""
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

        axes[row, 3].text(0.5, 0.5,
                          f"θ = {theta:.1f}°\ntx = {pose[2]:.3f}\nty = {pose[3]:.3f}",
                          transform=axes[row, 3].transAxes, fontsize=14,
                          ha="center", va="center", family="monospace")
        axes[row, 3].set_title("Pose" if row == 0 else "")
        axes[row, 3].axis("off")

    if has_gt:
        axes[3, 0].imshow(tactile_img)
        axes[3, 0].set_ylabel("Ground Truth", fontsize=12, fontweight="bold")
        axes[3, 1].imshow(gt_depth, cmap="viridis")
        gt_normal_vis = (gt_normal * 0.5 + 0.5).clip(0, 1)
        axes[3, 2].imshow(gt_normal_vis)
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
            gt_depth_raw = np.load(gt_depth_path).astype(np.float32)
            gt_depth = gt_depth_raw * 1000.0
            import json
            session_json = osp.join(osp.dirname(session_dir), "session.json")
            if osp.exists(session_json):
                with open(session_json) as f:
                    sess = json.load(f)
                px = (sess["X_MAX"] - sess["X_MIN"]) / cfg.image_size
                py = (sess["Y_MAX"] - sess["Y_MIN"]) / cfg.image_size
                gt_normal = depth_to_normal(gt_depth_raw, px, py)

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


def run_eval_sim(model, cfg, device, output_dir, num_vis=20, pose_model=None):
    """Run inference on sim val set with GT comparison, all 3 configs."""
    from ..data.dataset import build_datasets
    _, val_ds = build_datasets(cfg)

    indices = np.linspace(0, len(val_ds) - 1, num_vis, dtype=int)
    os.makedirs(output_dir, exist_ok=True)

    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])

    print(f"Evaluating {num_vis} val samples (3 configs each)...")
    for i, idx in enumerate(indices):
        sample = val_ds[idx]
        rgb_t = sample["rgb"].unsqueeze(0)
        tac_t = sample["tactile"].unsqueeze(0)

        results = {}
        for config in CONFIGS_3:
            depth, normal, pose, theta = predict(model, rgb_t, tac_t, config=config,
                                                 device=device, pose_model=pose_model)
            results[config] = (depth, normal, pose, theta)

        gt_depth = sample["depth"][0].numpy()
        gt_normal = sample["normal"].permute(1, 2, 0).numpy()
        tac_img = (sample["tactile"].permute(1, 2, 0).numpy() * std + mean).clip(0, 255).astype(np.uint8)
        rgb_img = (sample["rgb"].permute(1, 2, 0).numpy() * std + mean).clip(0, 255).astype(np.uint8)

        save_path = osp.join(output_dir, f"val_{i:03d}.png")
        visualize_three_configs(tac_img, rgb_img, results,
                                gt_depth=gt_depth, gt_normal=gt_normal, save_path=save_path)

        gt_pose = sample["pose"].numpy()
        gt_theta = np.degrees(np.arctan2(gt_pose[1], gt_pose[0]))
        for cfg_name in CONFIGS_3:
            _, _, pose, theta = results[cfg_name]
            print(f"  val_{i:03d} [{cfg_name:8s}]: pred θ={theta:.1f}° gt θ={gt_theta:.1f}°")


def main():
    ap = argparse.ArgumentParser(description="Inference on real/sim data")
    ap.add_argument("--train-dir", required=True,
                    help="Training output directory (e.g. outputs/20260629_022120)")
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
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--num-vis", type=int, default=20)
    args = ap.parse_args()

    cfg = merge_configs(args.model, args.train, args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_name = osp.basename(args.train_dir.rstrip("/"))
    base_dir = osp.join("eval_results", f"{train_name}_best")

    # Load best_depth for depth/normal, best_pose for pose
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
        print(f"Using split checkpoints: depth={osp.basename(depth_ckpt)}, pose={osp.basename(pose_ckpt)}")
    else:
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
                     pose_model=pose_model)

    elif args.eval_all:
        sim_out = osp.join(base_dir, "sim_val_vis")
        real_out = osp.join(base_dir, "real_vis")
        os.makedirs(sim_out, exist_ok=True)
        os.makedirs(real_out, exist_ok=True)
        run_eval_sim(depth_model, cfg, device, sim_out, num_vis=args.num_vis,
                     pose_model=pose_model)
        run_real_dir(depth_model, args.real_dir_path, cfg, device, real_out,
                     max_samples=args.max_samples, pose_model=pose_model)

    print(f"\nResults saved to: {base_dir}/")


if __name__ == "__main__":
    main()
