"""Failure analysis of the labeling-stage alignment (sensor A, K=25, renderer configuration = real_pseudo_k25_al):
per-frame rotation / translation error of the aligned pose against the withheld GT pose of the unlabeled frames.
usage: python paper_RoboSoft/scripts/alignment_failures.py [root=real_pseudo_k25_al]
"""
import json, math, os.path as osp, sys, collections, numpy as np
G = "/media/hdd2/ihsuan/gs_blender"; root = sys.argv[1] if len(sys.argv) > 1 else "real_pseudo_k25_al"
fits = json.load(open(f"{G}/{root}/rc_fits.json"))
rows = collections.defaultdict(list)
for tac, f in fits.items():
    P = json.load(open(tac.replace("samples", "raw_data").replace(".png", "_pose.json")))
    th, x, y = f["pose"]; gth = P["rotation_euler"][2]; gx, gy = P["sample_x"] * 1e3, P["sample_y"] * 1e3
    rot = abs((math.degrees(th - gth) + 180) % 360 - 180); tr = math.hypot(x - gx, y - gy)
    tth, tx, ty = f["tp"]; rot0 = abs((math.degrees(tth - gth) + 180) % 360 - 180); tr0 = math.hypot(tx - gx, ty - gy)
    rows[tac.split("/")[-5]].append((rot, tr, rot0, tr0))
allv = np.array([v for o in rows for v in rows[o]])
print(f"{len(allv)} unlabeled frames | aligned rot: mean {allv[:,0].mean():.2f} median {np.median(allv[:,0]):.2f} deg | trans: mean {allv[:,1].mean():.2f} median {np.median(allv[:,1]):.2f} mm")
print(f"teacher pose: rot mean {allv[:,2].mean():.2f} med {np.median(allv[:,2]):.2f} | trans mean {allv[:,3].mean():.2f} med {np.median(allv[:,3]):.2f}")
for thr in (2, 5, 10):
    print(f"aligned rot > {thr:2d} deg: {100 * (allv[:,0] > thr).mean():5.1f} %   (teacher: {100 * (allv[:,2] > thr).mean():5.1f} %)")
for thr in (0.5, 1.0, 2.0):
    print(f"aligned trans > {thr} mm: {100 * (allv[:,1] > thr).mean():5.1f} %   (teacher: {100 * (allv[:,3] > thr).mean():5.1f} %)")
print(f"\n{'object':32s} {'n':>4s} {'rot_mean':>8s} {'rot>5':>6s} {'tr_mean':>7s} {'tr>1mm':>6s}")
for o, v in sorted(rows.items()):
    v = np.array(v); print(f"{o:32s} {len(v):4d} {v[:,0].mean():8.2f} {100*(v[:,0]>5).mean():5.1f}% {v[:,1].mean():7.2f} {100*(v[:,1]>1).mean():5.1f}%")
# share of the mean carried by the worst frames
s = np.sort(allv[:, 0])[::-1]; k = int(0.05 * len(s))
print(f"\nworst 5% of frames ({k}) carry {100 * s[:k].sum() / s.sum():.0f}% of the summed rotation error; mean without them {s[k:].mean():.2f} deg")
