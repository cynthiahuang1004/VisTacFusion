"""Convert new_real_test/ (second capture session, 2026-09-22) into the real_filtered layout.

Output: <out>/<sim_obj>/session_<run:03d>/sensor_0000/{samples,rgb,raw_data}/ + session.json,
matching gs_blender/real_filtered so SimVisuoTactileDataset loads it unchanged
(data.real.root = <out>). One real run -> one session; runs are numbered as in new_real_test.

Per frame: GT depth from the mesh z-buffer at the robot pose (same robot_to_gt / gt_depth_mm
as run_all_inference.py, saved in metres as raw_data/XXXX_gt.npy), pose json with sample_x/
sample_y (m, camera frame, sign convention of reorganize_to_sim_format.py) and
rotation_euler[2] = theta. Frames kept: pose/image lag <= 0.5 s, press >= --min-press mm,
every --stride-th frame within a press (frames at full press are near-duplicates).
RGB: the rgb_images frame nearest in time (for completeness; not used by tactile-only models).
Background: outputs/session_bg/real_new/<obj>__session_<run>.png from validation/bg_selection.json.

usage: python scripts/build_real_filtered_new.py --out /media/hdd2/ihsuan/gs_blender/real_filtered_new
"""
import argparse
import bisect
import json
import math
import os
import os.path as osp
import shutil

import cv2
import numpy as np
import trimesh
from PIL import Image

T = "/media/hdd2/ihsuan/VisTacFusion/new_real_test"
MESH_DIR = "/media/hdd2/ihsuan/gs_blender/meshes"
S = 224
FC = 0.816
VIEW = 0.017502 * FC
OBJ_CENTER = np.array([0.53809766, -0.102125])
Z_ANCHOR = 0.00925000011920929   # from real_filtered session.json
# real_name -> (sim_name, theta_offset_rad, bias_x_m, bias_y_m)   (run_all_inference.py)
OBJ_MAP = {
    "pattern_01_2_lines_angles_1": ("pattern_01_2_lines_angle_1_2", math.radians(0),   1.761e-3, -1.041e-3),
    "pattern_01_2_lines_angles_2": ("pattern_01_2_lines_angle_2",   math.radians(90),  2.598e-3, -0.892e-3),
    "pattern_01_2_lines_angles_3": ("pattern_01_2_lines_angle_3",   math.radians(0),   1.365e-3,  0.378e-3),
    "pattern_04_3_lines_angles_1": ("pattern_04_3_lines_angle_1",   math.radians(0),   2.768e-3, -0.486e-3),
    "pattern_04_3_lines_angles_2": ("pattern_04_3_lines_angle_2",   math.radians(90),  0.834e-3, -0.692e-3),
    "pattern_06_5_lines_angle_1":  ("pattern_06_5_lines_angle_1",   math.radians(0),   1.928e-3, -0.260e-3),
    "pattern_31_rod":              ("pattern_31_rod",               math.radians(90), -0.064e-3, -0.766e-3),
    "pattern_32":                  ("pattern_32",                   math.radians(90),  2.580e-3, -0.860e-3),
    "pattern_33":                  ("pattern_33",                   math.radians(90),  2.550e-3, -0.970e-3),
    "pattern_35":                  ("pattern_35",                   math.radians(90),  2.720e-3, -0.340e-3),
    "pattern_36":                  ("pattern_36",                   math.radians(90),  2.311e-3, -0.563e-3),
    "pattern_37":                  ("pattern_37",                   math.radians(90),  2.374e-3, -1.082e-3),
}

_zmap = {}


def get_zmap(sim_name):
    if sim_name not in _zmap:
        mesh = trimesh.load(f"{MESH_DIR}/{sim_name}.obj", force="mesh")
        V, F = mesh.vertices, mesh.faces
        LO = mesh.bounds[0][:2]; BB = mesh.bounds.mean(0)[:2]; PPM = 20; RS = 82 * PPM
        zmap = np.full((RS, RS), 999.0, np.float32)
        for fi in range(len(F)):
            v = V[F[fi]]
            px = np.clip((v[:, 0] - LO[0]) * PPM, 0, RS - 1)
            py = np.clip((v[:, 1] - LO[1]) * PPM, 0, RS - 1)
            tri = np.stack([px, py], 1).astype(np.int32).reshape(-1, 1, 2)
            x0, y0 = int(px.min()), int(py.min())
            x1, y1 = min(int(px.max()) + 1, RS), min(int(py.max()) + 1, RS)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = np.zeros((y1 - y0, x1 - x0), np.uint8)
            tl = tri.copy(); tl[:, :, 0] -= x0; tl[:, :, 1] -= y0
            cv2.fillConvexPoly(roi, tl, 1)
            ys, xs = np.where(roi > 0)
            zmap[ys + y0, xs + x0] = np.minimum(zmap[ys + y0, xs + x0], v[:, 2].min())
        _zmap[sim_name] = (zmap, LO, BB, PPM, RS)
    return _zmap[sim_name]


uu, vv = np.meshgrid(np.arange(S) + 0.5, np.arange(S) + 0.5)


def gt_depth_mm(sim_name, theta, x_mm, y_mm, press_mm):
    zmap, LO, BB, PPM, RS = get_zmap(sim_name); ZMIN = float(zmap.min())
    cx, cy = x_mm * 1e-3, -y_mm * 1e-3
    wx = -(uu - S / 2) * VIEW / S; wy = -(vv - S / 2) * VIEW / S
    c, s = math.cos(theta), math.sin(theta); dx, dy = wx - cx, wy - cy
    mx = (c * dx + s * dy) * 1000 + BB[0]; my = (-s * dx + c * dy) * 1000 + BB[1]
    jx = np.clip(((mx - LO[0]) * PPM).astype(int), 0, RS - 1)
    jy = np.clip(((my - LO[1]) * PPM).astype(int), 0, RS - 1)
    return np.clip(ZMIN + press_mm - zmap[jy, jx], 0, None).astype(np.float32)


def robot_to_gt(quat_xyzw, ee_xy, theta_off, bias_x, bias_y, contact_z, ee_z):
    theta = -2 * math.atan2(quat_xyzw[0], quat_xyzw[1]) + theta_off
    d = np.array(ee_xy) - OBJ_CENTER
    cx = d[1] + bias_x; cy = d[0] + bias_y
    return theta, cx * 1000, -cy * 1000, max(0.0, (contact_z - ee_z) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/media/hdd2/ihsuan/gs_blender/real_filtered_new")
    ap.add_argument("--min-press", type=float, default=0.3, help="mm")
    ap.add_argument("--stride", type=int, default=3, help="keep every k-th contact frame")
    ap.add_argument("--max-lag", type=float, default=0.5)
    ap.add_argument("--calib", default=None,
                    help="pose_calibration_new.json from calibrate_new_real_poses.py (per-run dtheta/dx/dy/dz)")
    ap.add_argument("--min-iou", type=float, default=0.0,
                    help="skip runs whose calibrated IoU is below this")
    args = ap.parse_args()
    calib = json.load(open(args.calib)) if args.calib else {}
    bg_sel = json.load(open(f"{T}/validation/bg_selection.json"))
    bg_out = "/media/hdd2/ihsuan/VisTacFusion/outputs/session_bg/real_new"
    os.makedirs(bg_out, exist_ok=True)
    total = 0
    for real_name in sorted(OBJ_MAP):
        if not osp.isdir(f"{T}/{real_name}"):
            continue
        sim_name, th_off, bx, by = OBJ_MAP[real_name]
        for run in sorted((d for d in os.listdir(f"{T}/{real_name}") if d.isdigit()), key=int):
            base = f"{T}/{real_name}/{run}"
            tp, rp = f"{base}/tactile_images/poses.json", f"{base}/rgb_images/poses.json"
            if not osp.exists(tp):
                continue
            tm = json.load(open(tp)); cz = tm["contact_z_m"]
            c = calib.get(f"{real_name}/{run}", {})
            if args.calib and not c:
                print(f"{real_name}/{run}: no calibration, skipped"); continue
            if c and c.get("iou_after", 1.0) < args.min_iou:
                print(f"{real_name}/{run}: IoU {c['iou_after']:.2f} < {args.min_iou}, skipped"); continue
            dth, dx, dy, dz = (c.get("dtheta_deg", 0.0), c.get("dx_mm", 0.0), c.get("dy_mm", 0.0), c.get("dz_mm", 0.0))
            rgb_frames = json.load(open(rp))["frames"] if osp.exists(rp) else []
            rgb_t = sorted((f["image_ros_timestamp_sec"], f["filename"]) for f in rgb_frames)
            rgb_ts = [t for t, _ in rgb_t]
            unit = f"{args.out}/{sim_name}/session_{int(run):03d}/sensor_0000"
            for sub in ("samples", "rgb", "raw_data"):
                os.makedirs(f"{unit}/{sub}", exist_ok=True)
            kept = []; cells = []; k_in_press = 0; prev_contact = False
            for f in tm["frames"]:
                if abs(f["pose_ros_timestamp_sec"] - f["image_ros_timestamp_sec"]) > args.max_lag:
                    continue
                ee = f["robot_pose_xyz_xyzw"]
                th, x_mm, y_mm, press = robot_to_gt(ee[3:], ee[:2], th_off, bx, by, cz, ee[2])
                th += math.radians(dth); x_mm += dx; y_mm += dy; press = max(0.0, press + dz)
                contact = press >= args.min_press
                if not contact:
                    prev_contact = False; continue
                k_in_press = k_in_press + 1 if prev_contact else 0
                prev_contact = True
                if k_in_press % args.stride:
                    continue
                idx = len(kept)
                g = gt_depth_mm(sim_name, th, x_mm, y_mm, press) * 1e-3
                np.save(f"{unit}/raw_data/{idx:04d}_gt.npy", g)
                shutil.copy(f"{base}/tactile_images/{f['filename']}", f"{unit}/samples/{idx:04d}.png")
                if rgb_ts:
                    j = min(bisect.bisect_left(rgb_ts, f["image_ros_timestamp_sec"]), len(rgb_ts) - 1)
                    shutil.copy(f"{base}/rgb_images/{rgb_t[j][1]}", f"{unit}/rgb/{idx:04d}.png")
                sx, sy = x_mm * 1e-3, y_mm * 1e-3
                pose = {"obj_name": sim_name, "sample_x": sx, "sample_y": sy, "sample_z": press * 1e-3,
                        "location": [sx, -sy, float(-press * 1e-3 - Z_ANCHOR)],
                        "rotation_euler": [0.0, 0.0, float(th)], "scale": [0.001] * 3, "fixed_scale": 1000.0,
                        "source": {"run": run, "frame": f["frame_id"], "iteration": f["iteration"],
                                   "press_mm": press, "target_rotation_deg": f.get("target_rotation_deg")}}
                json.dump(pose, open(f"{unit}/raw_data/{idx:04d}_pose.json", "w"))
                cells.append({"gx": idx, "gy": 0, "cx": sx, "cy": sy, "rx": 0.0, "ry": 0.0, "rz": float(th),
                              "depth_min": press * 1e-3, "depth_max": press * 1e-3, "contact_frac": 0.1})
                kept.append(idx)
            sess = {"obj": sim_name, "base_rotation": [0.0, 0.0, float(th_off)], "rotation_step": 0.0,
                    "fixed_scale": 1000.0, "_target_size_mm": 82.0, "NUM_OBJ_SAMPLES": len(kept),
                    "OBJ_DEPTH_MIN": 0.0008, "OBJ_DEPTH_MAX": 0.0012, "z_anchor": -Z_ANCHOR,
                    "capture": "new_real_test 2026-09-22", "source_run": run,
                    "calibration": c, "valid_cells": cells}
            json.dump(sess, open(f"{args.out}/{sim_name}/session_{int(run):03d}/session.json", "w"), indent=1)
            key = f"{real_name}/{run}"
            if key in bg_sel:
                Image.open(f"{T}/{bg_sel[key]['file']}").convert("RGB").save(
                    f"{bg_out}/{sim_name}__session_{int(run):03d}.png")
            print(f"{real_name}/{run} -> {sim_name}/session_{int(run):03d}: {len(kept)} frames", flush=True)
            total += len(kept)
    print("total", total)


if __name__ == "__main__":
    main()
