"""Qualitative figure v3: sim inputs EXACTLY pose-matched to the real samples.

Changes vs v2:
  * Sim tactile = G(real _gt.npy): the tactile GAN (outputs/tactile_gan/G_final.pt)
    applied to the real sample's own GT depth, so rot/trans/depth match exactly.
  * Sim RGB = Blender render at the real sample's pose (render_rgb_at_pose.py),
    displayed with the training-time sim.rgb_zoom 1.15 before the 0.816 crop.
  * Real display unchanged: fixed_center_crop(crop=0.816).

Layout: 4 rows (objects) × 12 columns grouped as:
  Sim (Tac, RGB) | Real (Tac, RGB) | GT (D, N) | Vis (D, N) | Tac (D, N) | Both (D, N)
"""
import sys, os
sys.path.insert(0, "/media/hdd2/ihsuan/VisTacFusion")
import os.path as osp
import json, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from vistacfusion.utils.config import Config, merge_configs
from vistacfusion.models.model import build_model
from vistacfusion.engine.inference import preprocess_image, _extract_outputs
from vistacfusion.data.dataset import depth_to_normal, zoom_center
from vistacfusion.data.transforms import FIXED_CROP, fixed_center_crop

DEVICE = "cuda:0"
CHECKPOINT = "/media/hdd2/ihsuan/VisTacFusion/outputs/ICRA2027/ablation_c816_sim148_sitr_mae_md_b70_inj075/best_depth.pt"
MODEL_CFG = "/media/hdd2/ihsuan/VisTacFusion/ablation/encoder/tac_sitr_rgb_mae.yaml"
TRAIN_CFG = "/media/hdd2/ihsuan/VisTacFusion/configs/train.yaml"
DATA_CFG = "/media/hdd2/ihsuan/VisTacFusion/configs/data.yaml"
REAL_ROOT = "/media/hdd2/ihsuan/gs_blender/real_data"
SIM_ROOT = "/media/hdd2/ihsuan/gs_blender/renders_v3"
OUT_DIR = "/media/hdd2/ihsuan/VisTacFusion/paper/figures"

IMAGE_SIZE = 224
FIXED_CROP_VAL = 0.816
RGB_ZOOM = 1.15          # sim RGB camera scale alignment (matches sim.rgb_zoom in training)
GEL_VIEW_M = 0.017502
# Pose-exact sim inputs (generated at each real sample's GT pose):
#   sim_tac_g/: G(real _gt.npy) with the tactile GAN, depth jitter off
#   sim_rgb/:  Blender RGB render at the reconstructed real pose (placement verified
#              by z-buffer depth reconstruction, IoU = 1.0)
SIM_AT_POSE = "/media/hdd2/ihsuan/VisTacFusion/eval_results/fig6_pose_matched_sim"

SAMPLES = [
    {"obj": "pattern_32",                 "label": "V-shape",    "real_idx": 180},
    {"obj": "pattern_35",                 "label": "Corner",     "real_idx": 290},
    {"obj": "pattern_04_3_lines_angle_1", "label": "Y-junction", "real_idx": 240},
    {"obj": "pattern_31_rod",             "label": "Rod",        "real_idx": 30},
]


def load_display_image(path, rgb_zoom=1.0):
    img = Image.open(path)
    if rgb_zoom != 1.0:
        img = zoom_center(img.convert("RGB"), rgb_zoom)
    raw = np.array(img, dtype=np.float32)
    return fixed_center_crop(raw, crop=FIXED_CROP_VAL).astype(np.uint8)


def load_gt(gt_depth_path):
    raw = np.load(gt_depth_path).astype(np.float32)
    cropped = fixed_center_crop(raw, crop=FIXED_CROP_VAL)
    depth = cropped * 1000.0
    px = GEL_VIEW_M * FIXED_CROP_VAL / IMAGE_SIZE
    normal = depth_to_normal(cropped, px, px)
    return depth, normal


def normal_to_rgb(normal):
    return (normal * 0.5 + 0.5).clip(0, 1)


def main():
    print("Loading model...")
    cfg = merge_configs(MODEL_CFG, TRAIN_CFG, DATA_CFG)
    cfg["fixed_crop"] = FIXED_CROP_VAL
    model = build_model(cfg).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}")

    all_data = []
    for s in SAMPLES:
        obj = s["obj"]
        print(f"\nProcessing {obj}...")

        # Real data paths
        real_base = osp.join(REAL_ROOT, obj, "session_000", "sensor_0000")
        real_tac_path = osp.join(real_base, "samples", f"{s['real_idx']:04d}.png")
        real_rgb_path = osp.join(real_base, "rgb", f"{s['real_idx']:04d}.png")
        real_depth_path = osp.join(real_base, "raw_data", f"{s['real_idx']:04d}_gt.npy")

        # Sim data at the SAME pose as the real sample
        tag = f"{obj}__session_000__{s['real_idx']:04d}.png"
        sim_tac_path = osp.join(SIM_AT_POSE, "sim_tac_g", tag)
        sim_rgb_path = osp.join(SIM_AT_POSE, "sim_rgb", tag)

        # Display images (real: crop only; sim RGB: zoom 1.15 then crop, as in training)
        real_tac_disp = load_display_image(real_tac_path)
        real_rgb_disp = load_display_image(real_rgb_path)
        sim_tac_disp = load_display_image(sim_tac_path)
        sim_rgb_disp = load_display_image(sim_rgb_path, rgb_zoom=RGB_ZOOM)

        # GT
        gt_depth, gt_normal = load_gt(real_depth_path)

        # Inference in 3 modes
        tac_t = preprocess_image(real_tac_path, IMAGE_SIZE, fixed_crop=FIXED_CROP_VAL).to(DEVICE)
        rgb_t = preprocess_image(real_rgb_path, IMAGE_SIZE, fixed_crop=FIXED_CROP_VAL).to(DEVICE)

        preds = {}
        for config in ["rgb", "tactile", "both"]:
            with torch.no_grad():
                out = model(rgb_t, tac_t, config=config)
            depth, normal, pose, theta = _extract_outputs(out)
            preds[config] = {"depth": depth, "normal": normal}

        all_data.append({
            "label": s["label"],
            "sim_tac": sim_tac_disp,
            "sim_rgb": sim_rgb_disp,
            "real_tac": real_tac_disp,
            "real_rgb": real_rgb_disp,
            "gt_depth": gt_depth,
            "gt_normal": gt_normal,
            "vis_depth": preds["rgb"]["depth"],
            "vis_normal": preds["rgb"]["normal"],
            "tac_depth": preds["tactile"]["depth"],
            "tac_normal": preds["tactile"]["normal"],
            "both_depth": preds["both"]["depth"],
            "both_normal": preds["both"]["normal"],
        })

    # --- Build figure ---
    print("\nBuilding figure...")
    n_rows = len(all_data)
    # 6 groups of 2 = 12 columns
    group_names = ["Sim input", "Real input", "Ground truth", "Vis-only", "Tac-only", "Both"]
    sub_labels = ["Tactile", "Visual", "Tactile", "Visual"] + ["Depth", "Normal"] * 4

    fig_w = 7.16  # IEEE double-column
    label_frac = 0.045
    gap_frac = 0.012
    n_gaps = 5
    usable = 1.0 - label_frac - n_gaps * gap_frac
    col_frac = usable / 12
    row_aspect = col_frac * fig_w  # make cells square
    header_h = 0.25  # inches
    fig_h = n_rows * row_aspect + header_h

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Compute x positions (in figure fraction)
    xs = []
    gx_starts = []
    x = label_frac
    for gi in range(6):
        gx_starts.append(x)
        for _ in range(2):
            xs.append(x)
            x += col_frac
        if gi < 5:
            x += gap_frac

    header_frac = header_h / fig_h
    # Header rows are placed in inches above the top of the image grid.
    grid_top = 1 - header_frac
    inch = 1.0 / fig_h
    y_sub = grid_top + 0.015 * inch      # sub-label baseline
    y_bracket = grid_top + 0.105 * inch  # bracket line
    y_group = grid_top + 0.125 * inch    # group-name baseline

    # Group headers + brackets
    for gi, gn in enumerate(group_names):
        gx0 = gx_starts[gi]
        gx1 = gx0 + 2 * col_frac
        mid = (gx0 + gx1) / 2
        fig.text(mid, y_group, gn,
                 ha='center', va='bottom', fontsize=6.5, fontweight='bold')
        fig.add_artist(plt.Line2D([gx0, gx1], [y_bracket, y_bracket],
                       transform=fig.transFigure, color='black', lw=0.5, clip_on=False))
        for bx in [gx0, gx1]:
            fig.add_artist(plt.Line2D([bx, bx], [y_bracket, y_bracket - 0.025 * inch],
                           transform=fig.transFigure, color='black', lw=0.5, clip_on=False))

    # Sub-labels
    for ci, sl in enumerate(sub_labels):
        fig.text(xs[ci] + col_frac / 2, y_sub, sl,
                 ha='center', va='bottom', fontsize=5)

    # Image grid
    row_frac = (1.0 - header_frac) / n_rows
    pad = 0.003

    items = [
        ("sim_tac", "img"), ("sim_rgb", "img"),
        ("real_tac", "img"), ("real_rgb", "img"),
        ("gt_depth", "depth"), ("gt_normal", "normal"),
        ("vis_depth", "depth"), ("vis_normal", "normal"),
        ("tac_depth", "depth"), ("tac_normal", "normal"),
        ("both_depth", "depth"), ("both_normal", "normal"),
    ]

    for ri, row in enumerate(all_data):
        y_top = 1 - header_frac - ri * row_frac

        # Row label
        fig.text(label_frac * 0.4, y_top - row_frac / 2, row["label"],
                 ha='center', va='center', fontsize=6.5, fontweight='bold',
                 rotation=90)

        # Depth range from GT
        d_min = row["gt_depth"].min()
        d_max = row["gt_depth"].max()
        if d_max - d_min < 1e-8:
            d_max = d_min + 1e-6

        for ci, (key, kind) in enumerate(items):
            ax = fig.add_axes([
                xs[ci] + pad,
                y_top - row_frac + pad,
                col_frac - 2 * pad,
                row_frac - 2 * pad,
            ])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.3)
                sp.set_color('#999')

            data = row[key]
            if kind == "img":
                ax.imshow(data)
            elif kind == "depth":
                ax.imshow(data, cmap='viridis', vmin=d_min, vmax=d_max)
            elif kind == "normal":
                ax.imshow(normal_to_rgb(data))

    for ext in ["png", "pdf"]:
        out = osp.join(OUT_DIR, f"qualitative_results.{ext}")
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.02)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
