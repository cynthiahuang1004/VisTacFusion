"""Pose error of the pseudo-label poses of the unlabeled sensor-A frames (K=25) against their withheld GT, per label root:
mean |rotation| (deg) and mean translation error (mm, (|dx|+|dy|)/2 as for the student metric).  -> ablation/pilot_robosoft/label_pose_errors.json"""
import json, math, os.path as osp
import numpy as np
G = "/media/hdd2/ihsuan/gs_blender"
ROOTS = {"tp": "real_pseudo_k25_tp", "ro_tp0": "real_pseudo_k25_ro_tp0", "rcp": "real_pseudo_k25_rcp", "ro_al": "real_pseudo_k25_ro_al", "al": "real_pseudo_k25_al", "rc100": "real_pseudo_k25_rc100"}
out = {}
for tag, root in ROOTS.items():
    fits = json.load(open(f"{G}/{root}/rc_fits.json")); rot, tr, trot, ttr = [], [], [], []
    for t, f in fits.items():
        P = json.load(open(t.replace("samples", "raw_data").replace(".png", "_pose.json")))
        th, x, y = f["pose"]; gth = P["rotation_euler"][2]; gx, gy = P["sample_x"] * 1e3, P["sample_y"] * 1e3
        rot.append(abs((math.degrees(th - gth) + 180) % 360 - 180)); tr.append((abs(x - gx) + abs(y - gy)) / 2)
        if "tp" in f:                                             # the teacher's (unrefined) pose stored with the fit
            tth, tx, ty = f["tp"]; trot.append(abs((math.degrees(tth - gth) + 180) % 360 - 180)); ttr.append((abs(tx - gx) + abs(ty - gy)) / 2)
    out[tag] = dict(root=root, n=len(rot), rot=float(np.mean(rot)), trans=float(np.mean(tr)), rot_med=float(np.median(rot)), trans_med=float(np.median(tr)),
                    tp_rot=float(np.mean(trot)) if trot else None, tp_trans=float(np.mean(ttr)) if ttr else None)
    print(f"{tag:7s} n={len(rot)} fitted: rot {out[tag]['rot']:.2f} deg trans {out[tag]['trans']:.2f} mm | teacher pose: rot {out[tag]['tp_rot']} trans {out[tag]['tp_trans']}")
json.dump(out, open("ablation/pilot_robosoft/label_pose_errors.json", "w"), indent=1)
