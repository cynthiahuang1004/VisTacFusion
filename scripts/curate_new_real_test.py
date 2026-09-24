"""Curate a small, high-quality cross-session TEST set from new_real_test.

The new session is used for inference only, so we keep just the frames whose GT can be trusted:
  1. start from the per-run calibration (best of v1 per-run / v2 per-object, by IoU),
  2. for every contact frame do a local refinement of (theta, x, y) around it, maximising the IoU
     between the black-hat ridge mask and the projected GT depth > 0 (press depth stays from the robot),
  3. keep frames with visible contact, refined IoU >= --min-iou, at most --per-run per run,
     and the best --per-obj per object,
  4. write them in the real_filtered layout to --out and a review grid to
     new_real_test/validation/curated_review.png (tactile + GT contour | GT depth).

usage: python scripts/curate_new_real_test.py [--per-obj 10] [--min-iou 0.4] [--workers 8]
"""
import argparse
import json
import math
import os
import os.path as osp
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from build_real_filtered_new import OBJ_MAP, T, Z_ANCHOR, get_zmap, gt_depth_mm, robot_to_gt  # noqa: E402
from calibrate_new_real_poses import blackhat_mask, iou  # noqa: E402

VAL = f"{T}/validation"
PRED_ROOT = f"{T}/output"          # user's VisTacFusion inference on the new session (depth/ = viridis-coloured pred)
_VIRIDIS_BG = np.array([68, 1, 84], np.float32)


def pred_mask(real_name, run, frame):
    """Contact mask from the model's predicted depth (colour distance from the viridis background)."""
    p = f"{PRED_ROOT}/{real_name}/{run}/depth/{frame:06d}.png"
    if not osp.exists(p):
        return None
    im = np.asarray(Image.open(p).convert("RGB"), np.float32)
    return (np.abs(im - _VIRIDIS_BG).sum(-1) > 60)


def parse_log(path):
    """calibration_blackhat.log (per run) or calibration_v2.log (per object) -> dict."""
    out = {}
    if not osp.exists(path):
        return out
    for line in open(path):
        m = re.match(r"(\S+)\s+IoU\s+([\d.]+)\s+->\s+([\d.]+)\s+dth=\s*([-+\d.]+)\s+dx=\s*([-+\d.]+)\s+dy=\s*([-+\d.]+)(?:\s+dz=\s*([-+\d.]+))?", line)
        if m:
            out[m.group(1)] = dict(iou=float(m.group(3)), dth=float(m.group(4)), dx=float(m.group(5)),
                                   dy=float(m.group(6)), dz=float(m.group(7) or 0.0))
    return out


def start_calib(real_name, run, v1, v2):
    a = v1.get(f"{real_name}/{run}"); b = v2.get(real_name)
    c = (a if a["iou"] >= b["iou"] else b) if (a and b) else (a or b or dict(dth=0.0, dx=0.0, dy=0.0, iou=0.0))
    return dict(c, dz=0.0)          # dz from IoU is width-biased; trust the robot press depth


def refine_run(args):
    real_name, run, calib, min_press, stride, max_lag, mask_kind = args
    sim_name, th_off, bx, by = OBJ_MAP[real_name]
    get_zmap(sim_name)
    m = json.load(open(f"{T}/{real_name}/{run}/tactile_images/poses.json")); cz = m["contact_z_m"]
    res = []; k = 0; prev = False
    for f in m["frames"]:
        if abs(f["pose_ros_timestamp_sec"] - f["image_ros_timestamp_sec"]) > max_lag:
            continue
        ee = f["robot_pose_xyz_xyzw"]
        th, x, y, p = robot_to_gt(ee[3:], ee[:2], th_off, bx, by, cz, ee[2])
        th += math.radians(calib["dth"]); x += calib["dx"]; y += calib["dy"]; p = max(0.0, p + calib["dz"])
        if p < min_press:
            prev = False; continue
        k = k + 1 if prev else 0; prev = True
        if k % stride:
            continue
        im = np.asarray(Image.open(f"{T}/{real_name}/{run}/tactile_images/{f['filename']}").convert("RGB"))
        mask = pred_mask(real_name, run, f["frame_id"]) if mask_kind == "pred" else blackhat_mask(im)
        if mask is None or mask.mean() < 0.01:
            continue                                   # no visible contact -> unusable
        # local refinement around the run calibration
        best, bs = (0.0, 0.0, 0.0), iou(mask, gt_depth_mm(sim_name, th, x, y, p))
        for dth in np.arange(-3, 3.01, 1.0):
            for dx in np.arange(-1.5, 1.51, 0.5):
                for dy in np.arange(-1.5, 1.51, 0.5):
                    s = iou(mask, gt_depth_mm(sim_name, th + math.radians(dth), x + dx, y + dy, p))
                    if s > bs:
                        bs, best = s, (dth, dx, dy)
        dth0, dx0, dy0 = best
        for dth in np.arange(dth0 - 0.5, dth0 + 0.51, 0.25):
            for dx in np.arange(dx0 - 0.25, dx0 + 0.26, 0.125):
                for dy in np.arange(dy0 - 0.25, dy0 + 0.26, 0.125):
                    s = iou(mask, gt_depth_mm(sim_name, th + math.radians(dth), x + dx, y + dy, p))
                    if s > bs:
                        bs, best = s, (dth, dx, dy)
        res.append(dict(real_name=real_name, run=run, frame=f["frame_id"], file=f["filename"],
                        iteration=f["iteration"], theta=th + math.radians(best[0]), x_mm=x + best[1],
                        y_mm=y + best[2], press_mm=p, iou=float(bs), mask_area=float(mask.mean())))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/media/hdd2/ihsuan/gs_blender/real_filtered_new_curated")
    ap.add_argument("--per-obj", type=int, default=10)
    ap.add_argument("--per-run", type=int, default=3)
    ap.add_argument("--per-iter", type=int, default=1, help="frames kept per press iteration (different press depths)")
    ap.add_argument("--min-iou", type=float, default=0.25, help="absolute IoU floor")
    ap.add_argument("--rel-iou", type=float, default=0.75, help="keep frames >= rel * best IoU of the object")
    ap.add_argument("--min-press", type=float, default=0.4)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mask", choices=["blackhat", "pred"], default="blackhat",
                    help="contact evidence: image black-hat ridges, or the predicted depth of new_real_test/output")
    ap.add_argument("--grid-dir", default=None, help="per-object review grids go here (default validation/curated_<mask>/)")
    args = ap.parse_args()

    v1 = parse_log(f"{VAL}/calibration_blackhat.log")
    v2 = parse_log(f"{VAL}/calibration_v2.log")
    jobs = []
    for real_name in sorted(OBJ_MAP):
        if not osp.isdir(f"{T}/{real_name}"):
            continue
        for run in sorted((d for d in os.listdir(f"{T}/{real_name}") if d.isdigit()), key=int):
            if osp.exists(f"{T}/{real_name}/{run}/tactile_images/poses.json"):
                jobs.append((real_name, run, start_calib(real_name, run, v1, v2), args.min_press, args.stride, 0.5, args.mask))
    cands = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for res in ex.map(refine_run, jobs):
            for r in res:
                cands.setdefault(r["real_name"], []).append(r)

    # selection: IoU threshold, diversity across runs / iterations, top per object
    selected = {}
    for real_name, lst in cands.items():
        top = max((r["iou"] for r in lst), default=0.0)
        thr = max(args.min_iou, args.rel_iou * top)
        lst = [r for r in lst if r["iou"] >= thr]
        lst.sort(key=lambda r: -r["iou"])
        per_run, per_iter, keep = {}, {}, []
        for r in lst:
            key_it = (r["run"], r["iteration"])
            if per_run.get(r["run"], 0) >= args.per_run or per_iter.get(key_it, 0) >= args.per_iter:
                continue
            keep.append(r); per_run[r["run"]] = per_run.get(r["run"], 0) + 1; per_iter[key_it] = per_iter.get(key_it, 0) + 1
            if len(keep) >= args.per_obj:
                break
        selected[real_name] = keep
        print(f"{real_name:30s} candidates={len(cands[real_name]):4d}  best IoU={top:.2f} thr={thr:.2f} pass={len(lst):4d}  kept={len(keep)}", flush=True)
    json.dump(selected, open(f"{VAL}/curated_selection_{args.mask}.json", "w"), indent=1)
    grid_dir = args.grid_dir or f"{VAL}/curated_{args.mask}"; os.makedirs(grid_dir, exist_ok=True)

    # write dataset (one session per real run, only selected frames) + review grid
    if osp.exists(args.out):
        shutil.rmtree(args.out)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tiles = []
    for real_name, keep in selected.items():
        sim_name, th_off, _, _ = OBJ_MAP[real_name]
        by_run = {}
        for r in keep:
            by_run.setdefault(r["run"], []).append(r)
        for run, rs in by_run.items():
            unit = f"{args.out}/{sim_name}/session_{int(run):03d}/sensor_0000"
            for sub in ("samples", "rgb", "raw_data"):
                os.makedirs(f"{unit}/{sub}", exist_ok=True)
            cells = []
            for idx, r in enumerate(sorted(rs, key=lambda r: r["frame"])):
                g = gt_depth_mm(sim_name, r["theta"], r["x_mm"], r["y_mm"], r["press_mm"])
                np.save(f"{unit}/raw_data/{idx:04d}_gt.npy", (g * 1e-3).astype(np.float32))
                src = f"{T}/{real_name}/{run}/tactile_images/{r['file']}"
                Image.open(src).convert("RGB").save(f"{unit}/samples/{idx:04d}.png")
                Image.open(src).convert("RGB").save(f"{unit}/rgb/{idx:04d}.png")   # placeholder (tactile-only models)
                sx, sy = r["x_mm"] * 1e-3, r["y_mm"] * 1e-3
                json.dump({"obj_name": sim_name, "sample_x": sx, "sample_y": sy, "sample_z": r["press_mm"] * 1e-3,
                           "location": [sx, -sy, float(-r["press_mm"] * 1e-3 - Z_ANCHOR)],
                           "rotation_euler": [0.0, 0.0, float(r["theta"])], "scale": [0.001] * 3, "fixed_scale": 1000.0,
                           "source": {"run": run, "frame": r["frame"], "iteration": r["iteration"],
                                      "refined_iou": r["iou"]}},
                          open(f"{unit}/raw_data/{idx:04d}_pose.json", "w"))
                cells.append({"gx": idx, "gy": 0, "cx": sx, "cy": sy, "rx": 0.0, "ry": 0.0, "rz": float(r["theta"]),
                              "depth_min": r["press_mm"] * 1e-3, "depth_max": r["press_mm"] * 1e-3, "contact_frac": 0.1})
                im = np.asarray(Image.open(src).convert("RGB")).copy()
                cs, _ = cv2.findContours((g > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im, cs, -1, (255, 255, 0), 1)
                dc = cv2.applyColorMap((np.clip(g / max(g.max(), 1e-6), 0, 1) * 255).astype(np.uint8),
                                       cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
                tiles.append((f"{real_name[8:]} r{run} f{r['frame']} p={r['press_mm']:.1f} IoU={r['iou']:.2f}",
                              np.concatenate([im, dc], 1)))
            json.dump({"obj": sim_name, "base_rotation": [0.0, 0.0, float(th_off)], "rotation_step": 0.0,
                       "fixed_scale": 1000.0, "_target_size_mm": 82.0, "NUM_OBJ_SAMPLES": len(rs),
                       "OBJ_DEPTH_MIN": 0.0008, "OBJ_DEPTH_MAX": 0.0012, "z_anchor": -Z_ANCHOR,
                       "capture": "new_real_test 2026-09-22 (curated)", "source_run": run, "valid_cells": cells},
                      open(f"{args.out}/{sim_name}/session_{int(run):03d}/session.json", "w"), indent=1)
    by_obj = {}
    for t, im in tiles:
        by_obj.setdefault(t.split(" r")[0], []).append((t, im))
    for obj, ts in by_obj.items():
        cols = 5; rows = -(-len(ts) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 2.25), squeeze=False)
        for ax in axes.flat: ax.axis("off")
        for ax, (t, im) in zip(axes.flat, ts): ax.imshow(im); ax.set_title(t, fontsize=7)
        fig.tight_layout(h_pad=0.3, w_pad=0.2); fig.savefig(f"{grid_dir}/{obj}.png", dpi=100); plt.close(fig)
    print(f"total curated frames: {len(tiles)} -> {args.out}; review grids in {grid_dir}/")


if __name__ == "__main__":
    main()
