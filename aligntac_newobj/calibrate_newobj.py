"""Placement calibration for the four new objects (edge, hex_key, marble, button; sensor A, 2026-10).

Raw data: gs_blender/real_new_objects_raw/<obj>/{<i>.jpg, tactile_poses.json}; pose i = [x, y, z, qx, qy, qz, qw]
(robot end effector, metres) when tactile image i was captured. Unlike the earlier captures there is no contact
height and no per-object offset, so per object we estimate
    theta_off  rotation between the robot yaw and the mesh frame,
    bx, by     mesh-centre position relative to the mean press position (mm, sensor-frame convention of gt_depth_mm),
    p0         mean press depth (mm); press_i = p0 + (z_mean - z_i). Fixed at 1.0 mm: IoU fits overestimate it (the
               image difference includes a deformation halo); the marble's contact ring (radius 2.2-2.6 mm on a
               6.61 mm sphere) matches an elastic (Hertz, a = sqrt(R d)) contact at d = 1.0 mm, and the earlier
               captures' robot-measured presses were 0.8-1.2 mm.
Score: IoU between the image contact mask (difference to a median no-contact background) and the mesh contact
(gt_depth_mm > 0), for every object. (A first version treated `edge` as a flat block and scored a black-hat line
along the mesh boundary; that came from the per-face-minimum z-map flattening the wedge, see
build_real_filtered_new.exact_zmap. The earlier fit is kept in calib_newobj_facemin_zmap.json.) Coarse grid (theta every 5 deg over 360, xy every 1 mm) around a feature-based initial
guess, then refinement.

usage (cwd = VisTacFusion): conda run -n vistacfusion python aligntac_newobj/calibrate_newobj.py
writes aligntac_newobj/calib_newobj.json and aligntac_newobj/calib_overlays_<obj>.png
"""
import json, math, os, sys
from concurrent.futures import ProcessPoolExecutor
import cv2, numpy as np
from PIL import Image

sys.path.insert(0, "scripts")
import build_real_filtered_new as B          # noqa: E402  (gt_depth_mm, get_zmap; VIEW = full gel view)

B.VIEW = 0.017502
RAW = "/media/hdd2/ihsuan/gs_blender/real_new_objects_raw"
OUT = "aligntac_newobj"
OBJS = {"button": "button_v2", "edge": "edge_v2", "hex_key": "hex_key_v2", "marble": "marble_v2"}
N_FIT = 40          # frames used for the fit (evenly spaced)
SUB = 2             # score on a 112x112 grid for speed


def load(obj):
    J = json.load(open(f"{RAW}/{obj}/tactile_poses.json"))
    ids = sorted(int(f[:-4]) for f in os.listdir(f"{RAW}/{obj}") if f.endswith(".jpg") and str(int(f[:-4])) in J)
    P = np.array([J[str(i)] for i in ids], dtype=np.float64)
    return ids, P


def robot_state(P, c_xy, z_mean, theta_off, bx, by, p0):
    """Same algebra as build_real_filtered_new.robot_to_gt, with a per-object reference point c_xy."""
    q = P[:, 3:]; theta = -2 * np.arctan2(q[:, 0], q[:, 1]) + theta_off
    d = P[:, :2] - c_xy
    x_mm = (d[:, 1] * 1000 + bx); y_mm = -(d[:, 0] * 1000 + by)
    press = np.maximum(0.0, p0 + (z_mean - P[:, 2]) * 1000)
    return theta, x_mm, y_mm, press


def masks_for(obj, ids, bg):
    ims = [np.asarray(Image.open(f"{RAW}/{obj}/{i}.jpg").convert("RGB")).astype(np.float32) for i in ids]
    out = []
    for im in ims:
        d = np.abs(cv2.GaussianBlur(im, (5, 5), 0) - cv2.GaussianBlur(bg, (5, 5), 0)).sum(-1)
        m = (d > max(40.0, np.percentile(d, 60))).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        out.append(m > 0)
    return ims, out


_GRID = {}


def gtd(sim, theta, x_mm, y_mm, press_mm, sub=1):
    """gt_depth_mm evaluated on every `sub`-th pixel (same formula as build_real_filtered_new.gt_depth_mm)."""
    if sub not in _GRID:
        _GRID[sub] = (B.uu[::sub, ::sub], B.vv[::sub, ::sub])
    uu, vv = _GRID[sub]; S = B.S
    zmap, LO, BB, PPM, RS = B.get_zmap(sim); ZMIN = float(zmap.min())
    cx, cy = x_mm * 1e-3, -y_mm * 1e-3
    wx = -(uu - S / 2) * B.VIEW / S; wy = -(vv - S / 2) * B.VIEW / S
    c, s = math.cos(theta), math.sin(theta); dx, dy = wx - cx, wy - cy
    mx = (c * dx + s * dy) * 1000 + BB[0]; my = (-s * dx + c * dy) * 1000 + BB[1]
    jx = np.clip(((mx - LO[0]) * PPM).astype(int), 0, RS - 1); jy = np.clip(((my - LO[1]) * PPM).astype(int), 0, RS - 1)
    return np.clip(ZMIN + press_mm - zmap[jy, jx], 0, None).astype(np.float32)


def score(obj, sim, masks, st, sub=SUB):
    th, x, y, p = st; s = 0.0
    for m, a, b, c, e in zip(masks, th, x, y, p):
        g = gtd(sim, a, b, c, e, sub); mm = m[::sub, ::sub]
        gm = g > 0.0; s += (gm & mm).sum() / ((gm | mm).sum() + 1e-6)
    return s / len(masks)


def feature_init(sim):
    """Mesh-frame offset (mm, relative to the mesh bounding-box centre) of the region within 0.3 mm of the tip."""
    z, LO, BB, PPM, RS = B.get_zmap(sim); top = z < z.min() + 0.3
    jy, jx = np.nonzero(top); mx = LO[0] + (jx.mean() + 0.5) / PPM; my = LO[1] + (jy.mean() + 0.5) / PPM
    return mx - BB[0], my - BB[1]


def fit(obj):
    sim = OBJS[obj]; B.get_zmap(sim)
    ids, P = load(obj); bg = np.asarray(Image.open(f"{OUT}/bg_median.png").convert("RGB")).astype(np.float32)
    sel = np.linspace(0, len(ids) - 1, N_FIT).astype(int); ids_f, P_f = [ids[k] for k in sel], P[sel]
    _, masks = masks_for(obj, ids_f, bg)
    c_xy = P[:, :2].mean(0); z_mean = P[:, 2].mean()
    fx, fy = feature_init(sim)
    best = (-1, None)
    p0 = 1.0
    th_grid = [0.0] if obj == "marble" else np.radians(np.arange(-180, 180, 5.0))
    for t in th_grid:                                   # mesh centre seen from the sensor: rotate the feature offset into the sensor frame
        c, s_ = math.cos(t), math.sin(t)
        # mesh coords = R(theta) (sensor - centre) + BB  ->  feature at sensor centre when centre = -R^T f
        bx0 = -(c * fx - s_ * fy); by0_s = -(s_ * fx + c * fy)          # (x_mm, -y_mm) of the centre in the sensor frame
        for dx in np.arange(-12, 12.01, 1.0):
            for dy in np.arange(-12, 12.01, 1.0):
                bx, by = bx0 + dx, -(-by0_s) + dy                       # by enters as y_mm = -(d*1000 + by)
                sc = score(obj, sim, masks, robot_state(P_f, c_xy, z_mean, t, bx, by, p0), sub=4)
                if sc > best[0]: best = (sc, (t, bx, by, p0))
    # refine theta / xy, then p0
    for step_t, step_xy in ((1.0, 0.25), (0.25, 0.1)):
        t0, bx0, by0, p00 = best[1]
        tg = [t0] if obj == "marble" else t0 + np.radians(np.arange(-4 * step_t, 4.01 * step_t, step_t))
        for t in tg:
            for bx in bx0 + np.arange(-4, 4.01) * step_xy:
                for by in by0 + np.arange(-4, 4.01) * step_xy:
                    sc = score(obj, sim, masks, robot_state(P_f, c_xy, z_mean, t, bx, by, p00))
                    if sc > best[0]: best = (sc, (t, bx, by, p00))
    sc0 = score(obj, sim, masks, robot_state(P_f, c_xy, z_mean, *best[1]))
    t, bx, by, p0 = best[1]
    print(obj, 'fit', best, flush=True)
    return obj, dict(sim=sim, theta_off_deg=float(math.degrees(t)), bx_mm=float(bx), by_mm=float(by), p0_mm=float(p0), score=float(sc0),
                     c_xy_m=c_xy.tolist(), z_mean_m=float(z_mean), n_fit=N_FIT, n_frames=len(ids))


def overlays(obj, cal, n=8):
    sim = cal["sim"]; ids, P = load(obj); sel = np.linspace(0, len(ids) - 1, n).astype(int)
    st = robot_state(P[sel], np.array(cal["c_xy_m"]), cal["z_mean_m"], math.radians(cal["theta_off_deg"]), cal["bx_mm"], cal["by_mm"], cal["p0_mm"])
    row = []
    for k, (a, b, c, e) in zip(sel, zip(*st)):
        im = np.asarray(Image.open(f"{RAW}/{obj}/{ids[k]}.jpg").convert("RGB")).copy()
        g = B.gt_depth_mm(sim, a, b, c, e); m = (g > 0).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cs, -1, (255, 255, 255), 3); cv2.drawContours(im, cs, -1, (40, 220, 90), 1)
        row.append(im)
    Image.fromarray(np.concatenate(row, 1)).save(f"{OUT}/calib_overlays_{obj}.png")


if __name__ == "__main__":
    objs = sys.argv[1:] or ["marble", "button", "hex_key", "edge"]
    with ProcessPoolExecutor(4) as ex:
        res = dict(ex.map(fit, objs))
    path = f"{OUT}/calib_newobj.json"; cal = json.load(open(path)) if os.path.exists(path) else {}
    cal.update(res)
    json.dump(cal, open(path, "w"), indent=1, default=float)
    for o in objs:
        overlays(o, cal[o]); r = cal[o]
        print(f"{o:8s} score {r['score']:.3f}  theta_off {r['theta_off_deg']:+7.2f} deg  bx {r['bx_mm']:+6.2f}  by {r['by_mm']:+6.2f} mm  p0 {r['p0_mm']:.2f} mm  ({r['n_frames']} frames)", flush=True)
