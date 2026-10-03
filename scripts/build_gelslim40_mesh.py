"""Mesh-registered GelSlim 4.0 roots: the public dataset's depth labels replaced by the mesh projection at the pose that
best explains them (scripts/tactile_render/fit_mesh_pose.py --frames all), and its pose labels by that pose, so that
labels follow the sensor-A convention (mesh at a known planar pose).  Frames, rgb and session files are linked.

usage: python scripts/build_gelslim40_mesh.py --src .../real_gelslim40_train --fits a.json b.json --out .../real_gelslim40_train_mesh
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__))))
import build_real_filtered_new as B  # noqa: E402

B.VIEW = 0.012


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--fits", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fits = {}
    for f in args.fits:
        fits.update(json.load(open(f)))
    n = 0; res = []
    for unit in sorted(glob.glob(osp.join(args.src, "*", "session_*", "sensor_0000"))):
        obj = unit.split("/")[-3]; ou = unit.replace(args.src, args.out)
        os.makedirs(osp.join(ou, "raw_data"), exist_ok=True)
        for name in ("samples", "rgb", "calibration", "id_map.json"):
            if osp.lexists(osp.join(unit, name)) and not osp.lexists(osp.join(ou, name)):
                os.symlink(osp.join(unit, name), osp.join(ou, name))
        sj = osp.join(osp.dirname(unit), "session.json")
        if osp.exists(sj):
            d = json.load(open(sj)); d["capture"] = d.get("capture", "") + " / labels re-registered to the mesh (build_gelslim40_mesh.py)"
            json.dump(d, open(osp.join(osp.dirname(ou), "session.json"), "w"), indent=1)
        for tac in sorted(glob.glob(osp.join(unit, "samples", "*.png"))):
            if tac not in fits:
                continue
            th, x, y, p, e = fits[tac]; idx = osp.basename(tac)[:4]
            np.save(osp.join(ou, "raw_data", idx + "_gt.npy"), (B.gt_depth_mm(obj, th, x, y, p) * 1e-3).astype(np.float32))
            json.dump({"obj_name": obj, "sample_x": x * 1e-3, "sample_y": y * 1e-3, "sample_z": p * 1e-3, "rotation_euler": [0.0, 0.0, th],
                       "fit_residual": e, "source": "mesh fitted to the dataset depth label"}, open(osp.join(ou, "raw_data", idx + "_pose.json"), "w"))
            n += 1; res.append(e)
    print(f"{n} frames -> {args.out}; mean fit residual {np.mean(res):.4f} mm")


if __name__ == "__main__":
    main()
