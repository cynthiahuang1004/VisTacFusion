"""Pilot summary: best-by-val-depth epoch per run; A reports held-out test at that epoch."""
import glob, json, os.path as osp
OUT = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"
print(f"{'run':28s} {'ep':>3s} | {'val_depth':>9s} {'val_rot':>7s} {'val_tr':>7s} | {'TEST_depth':>10s} {'TEST_rot':>8s} {'TEST_tr':>7s}")
for p in sorted(glob.glob(f"{OUT}/pilot_*/history.json")):
    h = json.load(open(p))
    if not h: continue
    g = lambda e, k: next(iter(e[k].values()))
    b = min(h, key=lambda e: g(e, "val")["depth_mse"])          # select on val only
    br = min(h, key=lambda e: g(e, "val")["pose_rot_deg"])
    v, vr = g(b, "val"), g(br, "val")
    row = f"{osp.basename(osp.dirname(p))[6:]:28s} {len(h):3d} | {v['depth_mse']:9.5f} {vr['pose_rot_deg']:7.2f} {vr['pose_trans']:7.4f} | "
    if "test" in b:
        t, tr = g(b, "test"), g(br, "test")
        row += f"{t['depth_mse']:10.5f} {tr['pose_rot_deg']:8.2f} {tr['pose_trans']:7.4f}"
    print(row)
