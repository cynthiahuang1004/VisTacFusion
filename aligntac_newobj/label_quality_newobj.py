"""Pseudo-label quality per new object: teacher labels (real_pseudo_p4_k25_ro_ng) and aligned mesh labels
(real_pseudo_p4_k25_ro_al) against the robot-pose GT of real_filtered_plus4, on the unlabeled training frames.
Contact MAE (mm, over GT contact pixels), IoU (contact > 0.05 mm) and rotation error (deg) of the label pose.
usage (cwd = VisTacFusion): python aligntac_newobj/label_quality_newobj.py"""
import glob, json, math, os
import numpy as np
G = "/media/hdd2/ihsuan/gs_blender"; GT = f"{G}/real_filtered_plus4"
ROOTS = {"teacher": f"{G}/real_pseudo_p4_k25_ro_ng", "aligned": f"{G}/real_pseudo_p4_k25_ro_al"}
res = {}
for obj in ("button_v2", "edge_v2", "hex_key_v2", "marble_v2"):
    for name, root in ROOTS.items():
        rd = f"{root}/{obj}/session_000/sensor_0000/raw_data"; c_sum = c_n = 0.0; ious, rots = [], []
        for f in sorted(glob.glob(f"{rd}/*_gt.npy")):
            if os.path.islink(f) and os.path.realpath(f).startswith(os.path.realpath(GT)):
                continue                                            # labeled (K-shot) or validation frame: true label
            i = os.path.basename(f)[:4]; g = np.load(f"{GT}/{obj}/session_000/sensor_0000/raw_data/{i}_gt.npy") * 1e3
            p = np.load(f) * 1e3; m = g > 0; pm = p > 0.05
            c_sum += np.abs(p - g)[m].sum(); c_n += m.sum(); ious.append((pm & m).sum() / max(1, (pm | m).sum()))
            pj, gj = f"{rd}/{i}_pose.json", f"{GT}/{obj}/session_000/sensor_0000/raw_data/{i}_pose.json"
            if os.path.exists(pj):
                a = json.load(open(pj))["rotation_euler"][2] - json.load(open(gj))["rotation_euler"][2]
                rots.append(abs(math.degrees(math.atan2(math.sin(a), math.cos(a)))))
        res[f"{obj}/{name}"] = dict(c_mae=c_sum / max(1, c_n), iou=float(np.mean(ious)), rot=float(np.mean(rots)) if rots else None,
                                    rot_med=float(np.median(rots)) if rots else None, n=len(ious))
        r = res[f"{obj}/{name}"]; print(f"{obj:11s} {name:8s} n {r['n']:4d}  cMAE {r['c_mae']:.3f}  IoU {r['iou']:.3f}  rot {r['rot']}  med {r['rot_med']}", flush=True)
json.dump(res, open("aligntac_newobj/label_quality_newobj.json", "w"), indent=1)
