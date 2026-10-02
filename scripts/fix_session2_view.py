"""Session-2 curated set in the session-1 convention (full-view frames AND full-view depth).

real_filtered_new_curated_pred_r2 stores each tactile frame at the full 17.5 mm gel view (like real_filtered) but its
depth label at the 0.816 center view (VIEW = 0.017502 * 0.816 in build_real_filtered_new.py).  The dataset center-crops
frame and depth by the same 0.816, so the label ended up 1.225x magnified with respect to the frame.  Verified with
trained models: predicting on the 0.816-cropped frame and comparing with the stored depth gives IoU 0.73 / contact MAE
0.26 mm (pilot150_B_k100_al2), against 0.65 / 0.38 with the mismatched pairing.

This script writes <root>_fv: everything symlinked, *_gt.npy re-rendered from the mesh at the same pose over the full
view, so that the standard pipeline (crop both by fixed_crop) is correct.

usage: python scripts/fix_session2_view.py [--root ...]
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import build_real_filtered_new as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/media/hdd2/ihsuan/gs_blender/real_filtered_new_curated_pred_r2")
    args = ap.parse_args()
    out = args.root.rstrip("/") + "_fv"
    n = 0; chk = []
    for unit in sorted(glob.glob(osp.join(args.root, "*", "session_*", "sensor_0000"))):
        ou = unit.replace(args.root, out)
        os.makedirs(osp.join(ou, "raw_data"), exist_ok=True)
        for name in os.listdir(unit):
            if name != "raw_data" and not osp.lexists(osp.join(ou, name)):
                os.symlink(osp.join(unit, name), osp.join(ou, name))
        sj = osp.join(osp.dirname(unit), "session.json")
        if osp.exists(sj) and not osp.lexists(osp.join(osp.dirname(ou), "session.json")):
            os.symlink(sj, osp.join(osp.dirname(ou), "session.json"))
        for f in sorted(os.listdir(osp.join(unit, "raw_data"))):
            src, dst = osp.join(unit, "raw_data", f), osp.join(ou, "raw_data", f)
            if not f.endswith("_gt.npy"):
                if not osp.lexists(dst):
                    os.symlink(src, dst)
                continue
            P = json.load(open(src.replace("_gt.npy", "_pose.json")))
            pose = (P["obj_name"], P["rotation_euler"][2], P["sample_x"] * 1e3, P["sample_y"] * 1e3, P["sample_z"] * 1e3)
            B.VIEW = 0.017502 * B.FC                           # must reproduce the stored label
            old = np.load(src) * 1000; g = B.gt_depth_mm(*pose)
            chk.append(float(np.abs(g - old).max()))
            B.VIEW = 0.017502                                  # full view, as in real_filtered
            if osp.lexists(dst):
                os.remove(dst)
            np.save(dst, (B.gt_depth_mm(*pose) * 1e-3).astype(np.float32)); n += 1
    for extra in glob.glob(osp.join(args.root, "*.json")):
        if not osp.lexists(osp.join(out, osp.basename(extra))):
            os.symlink(extra, osp.join(out, osp.basename(extra)))
    print(f"{n} depth maps re-rendered at the full view -> {out}; max |stored - re-rendered at 0.816 view| = {max(chk):.4f} mm (median {np.median(chk):.4f})")


if __name__ == "__main__":
    main()
