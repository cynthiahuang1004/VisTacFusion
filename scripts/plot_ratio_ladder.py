#!/usr/bin/env python3
"""Plot ratio ladder curves from history.json data.

Depth MSE / Normal MSE use best-depth epoch.
Rot L1 / Rot (deg) / Trans L1 use best-pose epoch.
"""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/media/hdd2/ihsuan/VisTacFusion/outputs"
OUT_DIR = "/media/hdd2/ihsuan/VisTacFusion/eval_results"

RATIOS = [
    ("1:0",   "realonly", 0),
    ("1:0.3", "sim34",    0.3),
    ("1:0.6", "sim68",    0.6),
    ("1:1.0", "sim120",   1.0),
    ("1:1.6", "sim190",   1.6),
    ("1:2.1", "sim250",   2.1),
    ("1:2.7", "sim315",   2.7),
    ("1:3.3", "sim380",   3.3),
    ("1:4.9", "sim570",   4.9),
    ("1:6.5", "sim760",   6.5),
]

METRICS = [
    ("depth_mse",  "Depth MSE"),
    ("normal_mse", "Normal MSE"),
    ("rot_l1",     "Rot L1"),
    ("rot_deg",    "Rot (deg)"),
    ("trans_l1",   "Trans L1"),
]

DENSE_METRICS = {"depth_mse", "normal_mse"}

MODES = ["both", "tactile", "rgb"]
MODE_LABELS = {"both": "both (RGB + tactile)", "tactile": "tactile-only", "rgb": "rgb-only"}
MODE_COLORS = {"both": "#1f77b4", "tactile": "#ff7f0e", "rgb": "#2ca02c"}
MODE_MARKERS = {"both": "o", "tactile": "s", "rgb": "^"}


def _get(v, key):
    ALIASES = {
        "depth_mse": ["depth_mse"],
        "normal_mse": ["normal_mse"],
        "rot_l1": ["pose_rot_l1", "rot_l1"],
        "rot_deg": ["pose_rot_deg", "rot_deg"],
        "trans_l1": ["pose_trans", "trans_l1"],
    }
    for alias in ALIASES.get(key, [key]):
        if alias in v:
            return v[alias]
    return None


def load_best_per_mode(history_path):
    with open(history_path) as f:
        hist = json.load(f)

    results = {}
    for mode in MODES:
        # Find best depth epoch and best pose epoch separately
        best_depth_e, best_depth_val = None, 1e9
        best_pose_e, best_pose_val = None, 1e9

        for i, rec in enumerate(hist):
            v = rec.get("val", {}).get(mode, {})
            d = _get(v, "depth_mse")
            r = _get(v, "rot_deg")
            if d is not None and d < best_depth_val:
                best_depth_val = d
                best_depth_e = i
            if r is not None and r < best_pose_val:
                best_pose_val = r
                best_pose_e = i

        if best_depth_e is None or best_pose_e is None:
            continue

        bd = hist[best_depth_e]["val"][mode]
        bp = hist[best_pose_e]["val"][mode]

        results[mode] = {
            "depth_mse":  _get(bd, "depth_mse"),
            "normal_mse": _get(bd, "normal_mse"),
            "rot_l1":     _get(bp, "rot_l1"),
            "rot_deg":    _get(bp, "rot_deg"),
            "trans_l1":   _get(bp, "trans_l1"),
        }
    return results


def gather(prefix):
    data = {}
    for label, suffix, ratio_val in RATIOS:
        if suffix == "realonly":
            d = os.path.join(BASE, "ratio_realonly")
        else:
            d = os.path.join(BASE, f"{prefix}_{suffix}")
        hf = os.path.join(d, "history.json")
        if os.path.isfile(hf):
            data[label] = load_best_per_mode(hf)
        else:
            print(f"  MISSING: {hf}")
    return data


def plot_ladder(data, title, out_path):
    fig, axes = plt.subplots(3, 5, figsize=(24, 12), constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    x_labels = [r[0] for r in RATIOS]
    x_pos = np.arange(len(x_labels))

    for row, mode in enumerate(MODES):
        for col, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row, col]
            vals = []
            for label, _, _ in RATIOS:
                if label in data and mode in data[label]:
                    v = data[label][mode].get(metric_key, None)
                    vals.append(v)
                else:
                    vals.append(None)

            valid_x = [x_pos[i] for i, v in enumerate(vals) if v is not None]
            valid_y = [v for v in vals if v is not None]

            color = MODE_COLORS[mode]
            marker = MODE_MARKERS[mode]
            ax.plot(valid_x, valid_y, color=color, marker=marker, markersize=6,
                    linewidth=1.5, label=MODE_LABELS[mode])

            for xi, yi in zip(valid_x, valid_y):
                if metric_key == "rot_deg":
                    txt = f"{yi:.2f}"
                elif metric_key in ("depth_mse", "normal_mse"):
                    txt = f"{yi:.4f}"
                else:
                    txt = f"{yi:.4f}"
                ax.annotate(txt, (xi, yi), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=6.5, color=color)

            if "1:0" in data and mode in data["1:0"]:
                baseline = data["1:0"][mode].get(metric_key, None)
                if baseline is not None:
                    ax.axhline(baseline, color=color, linestyle="--", alpha=0.4, linewidth=1)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(metric_label, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(MODE_LABELS[mode], fontsize=10, fontweight="bold")
            if row == 2:
                ax.set_xlabel("real : sim", fontsize=9)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


os.makedirs(OUT_DIR, exist_ok=True)

print("=== Blender 3-Session ===")
bl3s = gather("ratio_bl3s")
plot_ladder(bl3s, "Blender sim tactile — real:sim ratio ladder  (dense: best depth epoch; pose: best pose epoch; dashed = real-only)",
            os.path.join(OUT_DIR, "ratio_ladder_blender.png"))

print("\n=== GAN 3-Session ===")
g3s = gather("ratio_g3s")
plot_ladder(g3s, "GAN (G-rendered) sim tactile — real:sim ratio ladder  (dense: best depth epoch; pose: best pose epoch; dashed = real-only)",
            os.path.join(OUT_DIR, "ratio_ladder_gan.png"))

print("\nDone.")


print("\n=== GAN 3-Session TransFilt ===")
g3s_tf = {}
for label, suffix, ratio_val in RATIOS:
    if suffix == "realonly":
        d = os.path.join(BASE, "ratio_realonly")
    else:
        d = os.path.join(BASE, f"ratio_g3s_{suffix}_transfilt")
    hf = os.path.join(d, "history.json")
    if os.path.isfile(hf):
        g3s_tf[label] = load_best_per_mode(hf)
    else:
        print(f"  MISSING: {hf}")
plot_ladder(g3s_tf, "GAN (G-rendered) sim tactile + translation filter — real:sim ratio ladder  (dense: best depth epoch; pose: best pose epoch; dashed = real-only)",
            os.path.join(OUT_DIR, "ratio_ladder_gan_transfilt.png"))
