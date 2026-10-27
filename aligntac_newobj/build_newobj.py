"""Write the four new objects in the real_filtered layout and a merged 16-object root.

Inputs : gs_blender/real_new_objects_raw/<obj>/{<i>.jpg, tactile_poses.json}, aligntac_newobj/calib_newobj.json
Outputs: gs_blender/real_filtered_plus4/
           <old 12 objects>  -> symlinks to gs_blender/real_filtered/<obj>
           <obj>_v2/session_000/{session.json, sensor_0000/{samples,rgb,raw_data,calibration}}
             samples/NNNN.png   tactile image (re-encoded PNG)
             rgb/NNNN.png       symlink to the tactile image (the new captures have no RGB; models use tactile only)
             raw_data/NNNN_gt.npy, NNNN_pose.json   mesh depth (m) and contact state from the calibrated robot pose
         ablation/pilot_robosoft/testA_plus4_frames.json  = testA frames + the new objects' test frames (same rule)

usage (cwd = VisTacFusion): conda run -n vistacfusion python aligntac_newobj/build_newobj.py
"""
import json, math, os, shutil, sys
import numpy as np
from PIL import Image

sys.path.insert(0, "scripts"); sys.path.insert(0, "aligntac_newobj")
import build_real_filtered_new as B          # noqa: E402
from calibrate_newobj import RAW, load, robot_state   # noqa: E402

B.VIEW = 0.017502
G = "/media/hdd2/ihsuan/gs_blender"; OLD = f"{G}/real_filtered"; OUT = f"{G}/real_filtered_plus4"
Z_ANCHOR = B.Z_ANCHOR
cal = json.load(open("aligntac_newobj/calib_newobj.json"))
os.makedirs(OUT, exist_ok=True)
for o in sorted(os.listdir(OLD)):                                   # the 12 session-1 objects, unchanged
    if os.path.isdir(f"{OLD}/{o}") and not os.path.exists(f"{OUT}/{o}"):
        os.symlink(f"{OLD}/{o}", f"{OUT}/{o}")

test_new = []
for obj, c in sorted(cal.items()):
    sim = c["sim"]; ids, P = load(obj)
    th, x, y, p = robot_state(P, np.array(c["c_xy_m"]), c["z_mean_m"], math.radians(c["theta_off_deg"]), c["bx_mm"], c["by_mm"], c["p0_mm"])
    unit = f"{OUT}/{sim}/session_000/sensor_0000"
    if os.path.exists(f"{OUT}/{sim}"): shutil.rmtree(f"{OUT}/{sim}")
    for sub in ("samples", "rgb", "raw_data", "calibration"): os.makedirs(f"{unit}/{sub}")
    cells, id_map = [], {}
    for k, (i, a, sx_mm, sy_mm, pr) in enumerate(zip(ids, th, x, y, p)):
        Image.open(f"{RAW}/{obj}/{i}.jpg").convert("RGB").save(f"{unit}/samples/{k:04d}.png")
        os.symlink(f"../samples/{k:04d}.png", f"{unit}/rgb/{k:04d}.png")
        np.save(f"{unit}/raw_data/{k:04d}_gt.npy", B.gt_depth_mm(sim, float(a), float(sx_mm), float(sy_mm), float(pr)) * 1e-3)
        sx, sy = sx_mm * 1e-3, sy_mm * 1e-3
        json.dump({"obj_name": sim, "sample_x": sx, "sample_y": sy, "sample_z": pr * 1e-3,
                   "location": [sx, -sy, float(-pr * 1e-3 - Z_ANCHOR)], "rotation_euler": [0.0, 0.0, float(a)],
                   "scale": [0.001] * 3, "fixed_scale": 1000.0,
                   "source": {"capture": "new objects 2026-10", "image": f"{i}.jpg", "press_mm": float(pr)}},
                  open(f"{unit}/raw_data/{k:04d}_pose.json", "w"))
        cells.append({"gx": k, "gy": 0, "cx": sx, "cy": sy, "rx": 0.0, "ry": 0.0, "rz": float(a),
                      "depth_min": pr * 1e-3, "depth_max": pr * 1e-3, "contact_frac": 0.1})
        id_map[f"{k:04d}"] = str(i)
    json.dump(id_map, open(f"{unit}/id_map.json", "w"), indent=0)
    json.dump({"obj": sim, "base_rotation": [0.0, 0.0, math.radians(c["theta_off_deg"])], "rotation_step": 0.0,
               "fixed_scale": 1000.0, "_target_size_mm": 82.0, "NUM_OBJ_SAMPLES": len(ids),
               "OBJ_DEPTH_MIN": 0.0008, "OBJ_DEPTH_MAX": 0.0012, "z_anchor": -Z_ANCHOR,
               "capture": "new objects 2026-10 (aligntac_newobj/calibrate_newobj.py)", "calibration": c,
               "valid_cells": cells}, open(f"{OUT}/{sim}/session_000/session.json", "w"), indent=1)
    # test frames: same rule as testA (idx % 10 in (5, 8), never one of the K-shot frames for K = 10, 25, 50, 100)
    n = len(ids); train = [i for i in range(n) if i % 10]; kshot = set()
    for K in (10, 25, 50, 100):
        kshot |= {train[j] for j in np.linspace(0, len(train) - 1, K).round().astype(int)}
    t = [i for i in range(n) if i % 10 in (5, 8) and i not in kshot]
    test_new += [f"{sim}/session_000/sensor_0000/{i:04d}" for i in t]
    print(f"{obj} -> {sim}: {n} frames, press {p.min():.2f}-{p.max():.2f} mm, test {len(t)}", flush=True)

tA = json.load(open("ablation/pilot_robosoft/testA_frames.json"))
json.dump({"root": OUT, "rule": tA["rule"] + "; plus the four new objects under the same rule",
           "frames": tA["frames"] + test_new}, open("ablation/pilot_robosoft/testA_plus4_frames.json", "w"), indent=0)
print("test frames:", len(tA["frames"]), "+", len(test_new))
