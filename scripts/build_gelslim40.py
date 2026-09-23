"""Convert the MMint GelSlim 4.0 depth dataset (gelslim_depth/datasets/{test,train}_data/*.pt) to the
real_filtered layout used by VisTacFusion, for the CROSS-SENSOR experiment (sensor B).

Only the LEFT finger image / depth are used. The 12 pattern objects are the same physical objects as our
GelSlim 5.0 data (names differ slightly -> OBJ_MAP). Depth in the .pt files is metric (mm, negative =
indentation, ~1 mm press with the 35 mm grasp width); we store metres, positive = indentation, like our GT.
Image: central 320x320 crop (12 mm x 12 mm) -> 224x224. Depth: resized to the image grid, same crop.
Pose: rotation_euler[2] = in_hand_pose theta (rad); sample_x/y = in_hand_pose y/z (m). NOTE the theta
origin differs from our object frames by a per-object constant -> evaluate rotation after removing the
per-object median offset (eval option) or use it as-is for a pessimistic number.

usage: python scripts/build_gelslim40.py --split test --out /media/hdd2/ihsuan/gs_blender/real_gelslim40_test
       python scripts/build_gelslim40.py --split train --per-obj 300 --out /media/hdd2/ihsuan/gs_blender/real_gelslim40_train
"""
import argparse
import json
import os
import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image

SRC = "/media/hdd2/ihsuan/gelslim_depth/datasets"
OBJ_MAP = {  # GelSlim 4.0 dataset name -> our name
    "pattern_01_2_lines_angle_1": "pattern_01_2_lines_angle_1_2",
    "pattern_02_2_lines_angle_2": "pattern_01_2_lines_angle_2",
    "pattern_03_2_lines_angle_3": "pattern_01_2_lines_angle_3",
    "pattern_04_3_lines_angle_1": "pattern_04_3_lines_angle_1",
    "pattern_05_3_lines_angle_2": "pattern_04_3_lines_angle_2",
    "pattern_06_5_lines_angle_1": "pattern_06_5_lines_angle_1",
    "pattern_31_rod": "pattern_31_rod", "pattern_32": "pattern_32", "pattern_33": "pattern_33",
    "pattern_35": "pattern_35", "pattern_36": "pattern_36", "pattern_37": "pattern_37",
}
H, W = 320, 427
CROP = (W - H) // 2   # 53 -> central 320x320
OUT_SIZE = 224


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "train", "val"], default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-obj", type=int, default=None, help="evenly spaced subset per object")
    ap.add_argument("--objects", nargs="*", default=None)
    args = ap.parse_args()
    review = []
    for src_name, our_name in OBJ_MAP.items():
        if args.objects and our_name not in args.objects:
            continue
        pt = f"{SRC}/{args.split}_data/{src_name}_{args.split}.pt"
        if not osp.exists(pt):
            print("missing", pt); continue
        d = torch.load(pt, mmap=True, weights_only=False, map_location="cpu")
        n = d["tactile_image"].shape[0]
        sel = np.arange(n) if not args.per_obj or n <= args.per_obj else np.linspace(0, n - 1, args.per_obj).round().astype(int)
        unit = f"{args.out}/{our_name}/session_000/sensor_0000"
        for sub in ("samples", "rgb", "raw_data"):
            os.makedirs(f"{unit}/{sub}", exist_ok=True)
        cells = []
        for k, i in enumerate(sel):
            img = d["tactile_image"][i, :3].numpy().transpose(1, 2, 0)
            img = np.clip(img * (255 if img.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
            dep = -d["depth_image"][i, 0].numpy()                       # mm, positive = indentation
            dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_LINEAR)  # depth grid (327x420) -> image grid
            img_c = cv2.resize(img[:, CROP:CROP + H], (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
            dep_c = cv2.resize(dep[:, CROP:CROP + H], (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)
            dep_c = np.clip(dep_c, 0, None).astype(np.float32) * 1e-3   # metres
            y, z, th = [float(v) for v in d["in_hand_pose"][i]]
            press = float(dep_c.max())
            Image.fromarray(img_c).save(f"{unit}/samples/{k:04d}.png")
            Image.fromarray(img_c).save(f"{unit}/rgb/{k:04d}.png")          # placeholder (tactile-only models)
            np.save(f"{unit}/raw_data/{k:04d}_gt.npy", dep_c)
            json.dump({"obj_name": our_name, "sample_x": y, "sample_y": z, "sample_z": press,
                       "location": [y, -z, -press], "rotation_euler": [0.0, 0.0, th], "scale": [0.001] * 3,
                       "fixed_scale": 1000.0, "source": {"dataset": "gelslim40", "file": osp.basename(pt), "index": int(i)}},
                      open(f"{unit}/raw_data/{k:04d}_pose.json", "w"))
            cells.append({"gx": k, "gy": 0, "cx": y, "cy": z, "rx": 0.0, "ry": 0.0, "rz": th,
                          "depth_min": press, "depth_max": press, "contact_frac": float((dep_c > 0).mean())})
            if k in (0, len(sel) // 2):
                m = (dep_c > 0).astype(np.uint8); cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                v = img_c.copy(); cv2.drawContours(v, cs, -1, (255, 255, 0), 1); review.append(v)
        json.dump({"obj": our_name, "base_rotation": [0.0, 0.0, 0.0], "rotation_step": 0.0, "fixed_scale": 1000.0,
                   "_target_size_mm": 82.0, "NUM_OBJ_SAMPLES": len(sel), "OBJ_DEPTH_MIN": 0.0008, "OBJ_DEPTH_MAX": 0.0012,
                   "z_anchor": 0.0, "capture": f"GelSlim 4.0 (MMint gelslim_depth {args.split})", "_sensor_view_mm": 12.0,
                   "valid_cells": cells}, open(f"{args.out}/{our_name}/session_000/session.json", "w"), indent=1)
        print(f"{our_name:32s} {len(sel)} frames  press mean {np.mean([c['depth_min'] for c in cells])*1e3:.2f} mm", flush=True)
    if review:
        rows = [np.concatenate(review[i:i + 6], 1) for i in range(0, len(review) - len(review) % 6, 6)]
        Image.fromarray(np.concatenate(rows, 0)).save(f"{args.out}/review.png")
        print("review grid:", f"{args.out}/review.png")


if __name__ == "__main__":
    main()
