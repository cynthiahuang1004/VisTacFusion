"""Per-run pose/contact calibration for new_real_test (objects were re-placed on 2026-09-22, so the
first-session OBJ_MAP offsets no longer hold).

For each object/run: sample contact frames, extract the image contact mask (rgb_grad_mask from
filter_real_data.py), and grid-search corrections (dtheta, dx, dy, dz) applied on top of OBJ_MAP
that maximise the mean IoU between the mask and the mesh-projected GT depth > 0
(gt_depth_mm from build_real_filtered_new.py). Coarse-to-fine; dz shifts contact_z (press depth).

Writes new_real_test/validation/pose_calibration_new.json:
  {"<real_name>/<run>": {"dtheta_deg", "dx_mm", "dy_mm", "dz_mm", "iou_before", "iou_after", "n"}}
build_real_filtered_new.py --calib <json> applies it.

usage: python scripts/calibrate_new_real_poses.py [--runs pattern_33/1 ...] [--workers 8]
"""
import argparse
import json
import math
import os
import os.path as osp
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, "/media/hdd2/ihsuan/gs_blender")
from build_real_filtered_new import OBJ_MAP, T, get_zmap, gt_depth_mm, robot_to_gt  # noqa: E402
from filter_real_data import rgb_grad_mask  # noqa: E402

OUT_JSON = f"{T}/validation/pose_calibration_new.json"


def iou(mask, depth):
    m = mask > 0; d = depth > 0
    return np.logical_and(m, d).sum() / (np.logical_or(m, d).sum() + 1e-6)


def frames_of(real_name, run, n_frames, min_press=0.5):
    """Evenly spaced contact frames (distinct press iterations preferred)."""
    m = json.load(open(f"{T}/{real_name}/{run}/tactile_images/poses.json"))
    cz = m["contact_z_m"]; out = []
    for f in m["frames"]:
        if abs(f["pose_ros_timestamp_sec"] - f["image_ros_timestamp_sec"]) > 0.5:
            continue
        ee = f["robot_pose_xyz_xyzw"]
        press = max(0.0, (cz - ee[2]) * 1000)
        if press >= min_press:
            out.append((f, press))
    if not out:
        return [], cz
    # one frame per iteration (the deepest), then subsample evenly
    best = {}
    for f, p in out:
        it = f["iteration"]
        if it not in best or p > best[it][1]:
            best[it] = (f, p)
    sel = [v[0] for v in best.values()]
    if len(sel) < n_frames:      # add intermediate-depth frames from the same iterations
        extra = [f for f, p in out if f not in sel]
        step = max(1, len(extra) // max(1, n_frames - len(sel)))
        sel += extra[::step][: n_frames - len(sel)]
    return sel[:n_frames], cz


def calibrate_run(args):
    real_name, run, n_frames = args
    sim_name, th_off, bx, by = OBJ_MAP[real_name]
    get_zmap(sim_name)
    frames, cz = frames_of(real_name, run, n_frames)
    if not frames:
        return f"{real_name}/{run}", None
    masks, poses = [], []
    for f in frames:
        im = np.asarray(Image.open(f"{T}/{real_name}/{run}/tactile_images/{f['filename']}").convert("RGB"))
        masks.append(rgb_grad_mask(im) > 0)
        ee = f["robot_pose_xyz_xyzw"]
        poses.append(robot_to_gt(ee[3:], ee[:2], th_off, bx, by, cz, ee[2]))   # theta, x_mm, y_mm, press

    def score(dth, dx, dy, dz):
        s = 0.0
        for m, (th, x, y, p) in zip(masks, poses):
            g = gt_depth_mm(sim_name, th + math.radians(dth), x + dx, y + dy, max(0.0, p + dz))
            s += iou(m, g)
        return s / len(masks)

    before = score(0, 0, 0, 0)
    # stage 1: theta + xy, coarse
    best, bs = (0.0, 0.0, 0.0, 0.0), before
    for dth in np.arange(-45, 46, 3.0):
        for dx in np.arange(-6, 6.1, 1.0):
            for dy in np.arange(-6, 6.1, 1.0):
                s = score(dth, dx, dy, 0.0)
                if s > bs:
                    bs, best = s, (dth, dx, dy, 0.0)
    # stage 2: refine theta/xy and add dz
    for _ in range(2):
        dth0, dx0, dy0, dz0 = best
        for dth in np.arange(dth0 - 3, dth0 + 3.1, 0.5):
            for dx in np.arange(dx0 - 1, dx0 + 1.01, 0.25):
                for dy in np.arange(dy0 - 1, dy0 + 1.01, 0.25):
                    s = score(dth, dx, dy, dz0)
                    if s > bs:
                        bs, best = s, (dth, dx, dy, dz0)
        dth0, dx0, dy0, _ = best
        for dz in np.arange(-1.5, 1.51, 0.1):
            s = score(dth0, dx0, dy0, dz)
            if s > bs:
                bs, best = s, (dth0, dx0, dy0, dz)
    res = dict(dtheta_deg=float(best[0]), dx_mm=float(best[1]), dy_mm=float(best[2]), dz_mm=float(best[3]),
               iou_before=float(before), iou_after=float(bs), n=len(masks))
    return f"{real_name}/{run}", res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    jobs = []
    for real_name in sorted(OBJ_MAP):
        if not osp.isdir(f"{T}/{real_name}"):
            continue
        for run in sorted((d for d in os.listdir(f"{T}/{real_name}") if d.isdigit()), key=int):
            if osp.exists(f"{T}/{real_name}/{run}/tactile_images/poses.json"):
                if args.runs is None or f"{real_name}/{run}" in args.runs:
                    jobs.append((real_name, run, args.frames))
    results = json.load(open(OUT_JSON)) if osp.exists(OUT_JSON) else {}
    with ProcessPoolExecutor(args.workers) as ex:
        for key, res in ex.map(calibrate_run, jobs):
            if res is None:
                print(key, "no contact frames", flush=True); continue
            results[key] = res
            print(f"{key:36s} IoU {res['iou_before']:.3f} -> {res['iou_after']:.3f}  "
                  f"dth={res['dtheta_deg']:+6.1f} dx={res['dx_mm']:+5.2f} dy={res['dy_mm']:+5.2f} dz={res['dz_mm']:+5.2f}", flush=True)
            json.dump(results, open(OUT_JSON, "w"), indent=1)


if __name__ == "__main__":
    main()
