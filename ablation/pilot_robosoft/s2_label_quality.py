"""Pseudo-label quality on the session-2 unlabeled pool, measured ONLY on pool frames that also have a curated label
(real_s2_poolcheck; matched by object/session/raw frame id). Curated labels are never used for training."""
import glob, json, math, os.path as osp, sys, numpy as np
G = "/media/hdd2/ihsuan/gs_blender"
gt = {}
for pj in glob.glob(f"{G}/real_s2_poolcheck/*/session_*/sensor_0000/raw_data/*_pose.json"):
    d = json.load(open(pj)); gt[(pj.split("/")[-5], pj.split("/")[-4], int(d["source"]["frame"]))] = (pj, d)
def wrap(a): return abs((a + 180) % 360 - 180)
for K in (25, 100):
    res = {"teacher": [], "aligned": []}
    for pj in glob.glob(f"{G}/real_s2_pool/*/session_*/sensor_0000/raw_data/*_pose.json"):
        key = (pj.split("/")[-5], pj.split("/")[-4], int(json.load(open(pj))["source"]["frame"]))
        if key not in gt: continue
        gpj, gd = gt[key]; g = np.load(gpj.replace("_pose.json", "_gt.npy")) * 1000
        for name, root in (("teacher", f"real_pseudo_s2_k{K}_ng"), ("aligned", f"real_pseudo_s2_k{K}_al")):
            q = pj.replace("real_s2_pool", root)
            if not osp.exists(q): continue
            d = json.load(open(q)); m = np.load(q.replace("_pose.json", "_gt.npy")) * 1000
            a, b = m > 0.05, g > 0
            res[name].append((float((a & b).sum() / max(1, (a | b).sum())), float(np.abs(m - g)[b].mean()) if b.any() else np.nan,
                              wrap(math.degrees(d["rotation_euler"][2] - gd["rotation_euler"][2])),
                              1000 * (abs(d["sample_x"] - gd["sample_x"]) + abs(d["sample_y"] - gd["sample_y"])) / 2))
    for name, r in res.items():
        r = np.array(r); print(f"K={K:3d} {name:8s} n={len(r)}  IoU {np.nanmean(r[:,0]):.3f}  cMAE {np.nanmean(r[:,1]):.3f}  rot mean {r[:,2].mean():.2f} median {np.median(r[:,2]):.2f}  trans {r[:,3].mean():.2f} mm")
