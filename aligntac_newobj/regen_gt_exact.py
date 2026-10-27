"""Mirror a real root with *_gt.npy re-rendered from the same *_pose.json using the exact (interpolated) z-map for every
mesh (build_real_filtered_new.exact_zmap), full 17.5 mm view as in fix_session2_view.py. Everything else is symlinked.
usage: python aligntac_newobj/regen_gt_exact.py <src_root> <dst_root>"""
import glob, json, os, os.path as osp, sys
import numpy as np
sys.path.insert(0, "scripts"); import build_real_filtered_new as B
B.use_exact_zmap = lambda name: True; B.VIEW = 0.017502
src, dst = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"); n = 0; d_mae = []
for unit in sorted(glob.glob(f"{src}/*/session_*/sensor_0000")):
    ou = unit.replace(src, dst); os.makedirs(f"{ou}/raw_data", exist_ok=True)
    for name in os.listdir(unit):
        if name != "raw_data" and not osp.lexists(f"{ou}/{name}"):
            os.symlink(osp.realpath(f"{unit}/{name}"), f"{ou}/{name}")
    sj = osp.join(osp.dirname(unit), "session.json")
    if osp.exists(sj) and not osp.lexists(osp.join(osp.dirname(ou), "session.json")):
        os.symlink(osp.realpath(sj), osp.join(osp.dirname(ou), "session.json"))
    for f in sorted(os.listdir(f"{unit}/raw_data")):
        s, t = f"{unit}/raw_data/{f}", f"{ou}/raw_data/{f}"
        if not f.endswith("_gt.npy"):
            if not osp.lexists(t): os.symlink(osp.realpath(s), t)
            continue
        P = json.load(open(s.replace("_gt.npy", "_pose.json")))
        g = B.gt_depth_mm(P["obj_name"], P["rotation_euler"][2], P["sample_x"] * 1e3, P["sample_y"] * 1e3, P["sample_z"] * 1e3)
        old = np.load(s) * 1e3; m = (old > 0) | (g > 0)
        if m.any(): d_mae.append(float(np.abs(old - g)[m].mean()))
        if osp.lexists(t): os.remove(t)
        np.save(t, (g * 1e-3).astype(np.float32)); n += 1
for extra in glob.glob(f"{src}/*.json"):
    if not osp.lexists(f"{dst}/{osp.basename(extra)}"): os.symlink(osp.realpath(extra), f"{dst}/{osp.basename(extra)}")
print(f"{n} GT maps -> {dst}; mean |old - exact| over contact {np.mean(d_mae):.3f} mm (median {np.median(d_mae):.3f})")
