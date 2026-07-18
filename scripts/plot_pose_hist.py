"""Plot histogram of per-sample pose rotation error on the val set — all 3 configs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from vistacfusion.utils.config import merge_configs
from vistacfusion.models.model import build_model
from vistacfusion.data.dataset import build_datasets

CKPT = "outputs/20260712_050236/best_pose.pt"
OUT_DIR = "eval_results/20260712_050236_best"

CONFIGS = ["both", "tactile", "rgb"]
COLORS = {"both": "#4C72B0", "tactile": "#DD8452", "rgb": "#55A868"}
LABELS = {"both": "RGB+Tactile", "tactile": "Tactile-only", "rgb": "RGB-only"}


def main():
    cfg = merge_configs("configs/model.yaml", "configs/train.yaml", "configs/data.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    _, val_ds = build_datasets(cfg)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    results = {}

    for config in CONFIGS:
        all_rot_deg = []
        all_trans_err = []

        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(device)
                tactile = batch["tactile"].to(device)
                gt_pose = batch["pose"].to(device)

                out = model(rgb, tactile, config=config)
                se2 = out["se2"]

                cos_p, sin_p = se2[:, 0], se2[:, 1]
                cos_g, sin_g = gt_pose[:, 0], gt_pose[:, 1]

                dot = (cos_p * cos_g + sin_p * sin_g).clamp(-1 + 1e-6, 1 - 1e-6)
                rot_deg = torch.acos(dot) * 180.0 / math.pi
                all_rot_deg.append(rot_deg.cpu().numpy())

                trans_err = (se2[:, 2:] - gt_pose[:, 2:]).abs().sum(dim=-1)
                all_trans_err.append(trans_err.cpu().numpy())

        rot_deg = np.concatenate(all_rot_deg)
        trans_err = np.concatenate(all_trans_err)
        results[config] = {"rot_deg": rot_deg, "trans_err": trans_err}

        print(f"[{config}] Samples: {len(rot_deg)}")
        print(f"  Rotation (deg): mean={rot_deg.mean():.2f}, median={np.median(rot_deg):.2f}, "
              f"std={rot_deg.std():.2f}, max={rot_deg.max():.2f}")
        print(f"  Trans (L1): mean={trans_err.mean():.4f}, median={np.median(trans_err):.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, config in enumerate(CONFIGS):
        rot_deg = results[config]["rot_deg"]
        trans_err = results[config]["trans_err"]
        color = COLORS[config]
        label = LABELS[config]

        # Rotation
        ax = axes[0, i]
        ax.hist(rot_deg, bins=60, range=(0, 180), color=color, edgecolor="white", alpha=0.85)
        ax.axvline(rot_deg.mean(), color="red", linestyle="--",
                   label=f"Mean: {rot_deg.mean():.1f}°")
        ax.axvline(np.median(rot_deg), color="orange", linestyle="--",
                   label=f"Median: {np.median(rot_deg):.1f}°")
        ax.set_xlabel("Rotation Error (degrees)")
        ax.set_ylabel("Count")
        ax.set_title(f"Rotation — {label}")
        ax.legend()

        # Translation
        ax = axes[1, i]
        ax.hist(trans_err, bins=60, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(trans_err.mean(), color="red", linestyle="--",
                   label=f"Mean: {trans_err.mean():.3f}")
        ax.axvline(np.median(trans_err), color="orange", linestyle="--",
                   label=f"Median: {np.median(trans_err):.3f}")
        ax.set_xlabel("Translation Error (L1, normalized)")
        ax.set_ylabel("Count")
        ax.set_title(f"Translation — {label}")
        ax.legend()

    plt.suptitle("Pose Error Distribution (val set) — v3 epoch 60", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "pose_error_hist.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
