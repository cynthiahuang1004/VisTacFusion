"""Canonicalize mesh poses fitted to GelSlim 4.0 labels.

A pose fitted to a depth map is only defined up to the object's rotational symmetry (a two-line pattern repeats every
180 deg, a cross every 90 deg, ...), so fits of different frames land on arbitrary equivalent branches and would make
rotation labels inconsistent.  For each object we (1) measure the symmetry period P of the mesh depth numerically and
(2) pick, per frame, the equivalent pose closest to the dataset's own in-hand rotation plus a per-object constant
offset (the circular median of the residual modulo P).  Depth labels are unchanged by this choice.

usage: python scripts/canonicalize_gs40_poses.py --root <dataset root> --fits a.json [b.json ...] --out canon.json [--symmetry sym.json]
       (the symmetry file written by the first call can be reused with --symmetry for the test split)
"""
import argparse
import json
import math
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import build_real_filtered_new as B  # noqa: E402

B.VIEW = 0.012


def symmetry_period(obj):
    """Smallest rotation (deg) that maps the mesh depth onto itself (IoU > 0.97 at several poses), else 360."""
    for n in (6, 5, 4, 3, 2):
        P = 2 * math.pi / n; ok = True
        for th in np.linspace(0, 2 * math.pi, 5, endpoint=False):
            a = B.gt_depth_mm(obj, th, 0.0, 0.0, 1.0) > 0; b = B.gt_depth_mm(obj, th + P, 0.0, 0.0, 1.0) > 0
            if (a & b).sum() / max(1, (a | b).sum()) < 0.97:
                ok = False; break
        if ok:
            return 360.0 / n
    return 360.0


def circ_median(x, period):
    """Circular median of angles (deg) on a circle of the given period."""
    x = np.asarray(x) % period; best = None
    for c in x[:: max(1, len(x) // 200)]:
        d = np.abs((x - c + period / 2) % period - period / 2).sum()
        if best is None or d < best[0]:
            best = (d, c)
    return float(best[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="original dataset root (its _pose.json gives the in-hand rotation)")
    ap.add_argument("--fits", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symmetry", default=None, help="reuse symmetry/offset json of the train split")
    args = ap.parse_args()
    fits = {}
    for f in args.fits:
        fits.update(json.load(open(f)))
    by_obj = {}
    for t, v in fits.items():
        P = json.load(open(t.replace("samples", "raw_data").replace(".png", "_pose.json")))
        by_obj.setdefault(t.split("/")[-5], []).append((t, v, math.degrees(P["rotation_euler"][2])))
    sym = json.load(open(args.symmetry)) if args.symmetry else {}
    out = {}
    for obj, items in sorted(by_obj.items()):
        if obj not in sym:
            per = symmetry_period(obj)
            res = [(math.degrees(v[0]) - thd) % per for _, v, thd in items]
            sym[obj] = {"period_deg": per, "offset_deg": circ_median(res, per)}
        per, off = sym[obj]["period_deg"], sym[obj]["offset_deg"]
        spread = []
        for t, v, thd in items:
            thf = math.degrees(v[0]); k = round((thf - thd - off) / per)
            th = thf - k * per                                   # same depth, branch nearest to the dataset rotation + offset
            out[t] = [math.radians(th)] + list(v[1:]); spread.append((th - thd - off + 180) % 360 - 180)
        print(f"{obj:30s} period {per:5.1f} deg  offset {off:6.1f} deg  residual std {np.std(spread):5.2f} deg  n={len(items)}")
    json.dump(out, open(args.out, "w"))
    json.dump(sym, open(osp.splitext(args.out)[0] + "_symmetry.json", "w"), indent=1)


if __name__ == "__main__":
    main()
