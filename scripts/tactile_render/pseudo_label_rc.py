"""Render-and-compare pseudo-labeling: snap the teacher's pseudo labels onto the object mesh.

pseudo_label.py gives every unlabeled real frame the K-shot model's depth; that label is as blurred and as
mislocalized as the model.  The object mesh is known, so the true label is the mesh depth at SOME planar pose.
For every pseudo-labeled frame of an existing pseudo root we therefore
  1. take the teacher's pose (best_pose.pt, center window) and its press depth (max of the stitched depth),
  2. coarse: search (theta, x, y) so that the mesh depth agrees with the teacher depth,
  3. fine: render the mesh depth at candidate poses with the K-shot learned renderer and keep the pose whose
     render matches the real image best (full-image L1) -- the renderer verifies the label,
  4. keep the best --keep-frac of each object's frames by that render error: their label becomes the mesh depth
     at the fitted pose plus that pose (a regular label); the others keep the teacher depth and a pose flagged
     "pseudo" (skipped by the pose loss), or are dropped with --rest drop.
No label of an unlabeled frame is read; with --eval-gt the withheld labels are used to REPORT label quality only.

usage: python scripts/tactile_render/pseudo_label_rc.py --src-root .../real_pseudo_k25 --out-root .../real_pseudo_k25_rc \
           --run B_k25_gank --gan outputs/tactile_gan_k25 --device cuda:0 --eval-gt
"""
import argparse
import contextlib
import glob
import io
import json
import math
import os
import os.path as osp
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, osp.join(ROOT, "scripts")); sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import build_real_filtered_new as B  # noqa: E402
from train_tactile_gan import BG_ROOT, DEPTH_SCALE, UNetG  # noqa: E402

B.VIEW = 0.017502            # real_filtered frames are the full gel view (verified: IoU 0.98 with the stored GT)
REAL = "/media/hdd2/ihsuan/gs_blender/real_filtered"
OUT = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"
MODEL = "ablation/encoder/tac_sitr_single.yaml"
TRAIN = "ablation/pilot_robosoft/train_pilot_e50.yaml"
_uu, _vv = np.meshgrid(np.arange(0, 224, 2) + 1.0, np.arange(0, 224, 2) + 1.0)


def mesh_depth_half(obj, th, x, y, p):
    """Mesh depth (mm) on a stride-2 grid; same geometry as build_real_filtered_new.gt_depth_mm."""
    zmap, LO, BB, PPM, RS = B.get_zmap(obj); zmin = float(zmap.min())
    wx = -(_uu - 112) * B.VIEW / 224 - x * 1e-3; wy = -(_vv - 112) * B.VIEW / 224 + y * 1e-3
    c, s = math.cos(th), math.sin(th)
    jx = np.clip((((c * wx + s * wy) * 1000 + BB[0] - LO[0]) * PPM).astype(int), 0, RS - 1)
    jy = np.clip((((-s * wx + c * wy) * 1000 + BB[1] - LO[1]) * PPM).astype(int), 0, RS - 1)
    return np.clip(zmin + p - zmap[jy, jx], 0, None)


def coarse(a):
    """Fit (theta, x, y) of the mesh to the teacher depth, starting from the teacher pose."""
    dep, obj, th0, x0, y0 = a
    Dt = np.load(dep) * 1000
    p = float(np.clip(Dt.max(), 0.3, 1.6)); D2 = Dt[1::2, 1::2]
    best = (th0, x0, y0); bs = -np.abs(mesh_depth_half(obj, th0, x0, y0, p) - D2).mean()
    for rt, st, rx, sx in ((12.0, 2.0, 1.5, 0.5), (1.5, 0.5, 0.375, 0.125)):
        c0 = best
        for dth in np.arange(-rt, rt + 1e-6, st):
            for dx in np.arange(-rx, rx + 1e-6, sx):
                for dy in np.arange(-rx, rx + 1e-6, sx):
                    cand = (c0[0] + math.radians(dth), c0[1] + dx, c0[2] + dy)
                    s = -np.abs(mesh_depth_half(obj, *cand, p) - D2).mean()
                    if s > bs:
                        bs, best = s, cand
    return best, p


_zt = {}


def mesh_depth_t(obj, th, x, y, p, dev):
    """Batched mesh depth (mm) on GPU. th [N] rad; x, y, p [N] mm."""
    if obj not in _zt:
        zmap, LO, BB, PPM, RS = B.get_zmap(obj)
        _zt[obj] = (torch.from_numpy(zmap).to(dev), LO, BB, PPM, RS, float(zmap.min()))
    z, LO, BB, PPM, RS, zmin = _zt[obj]
    u = torch.arange(224, device=dev) + 0.5
    vv, uu = torch.meshgrid(u, u, indexing="ij")
    dx = (-(uu - 112) * B.VIEW / 224)[None] - (x * 1e-3)[:, None, None]
    dy = (-(vv - 112) * B.VIEW / 224)[None] + (y * 1e-3)[:, None, None]
    c, s = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
    jx = (((c * dx + s * dy) * 1000 + BB[0] - LO[0]) * PPM).long().clamp(0, RS - 1)
    jy = (((-s * dx + c * dy) * 1000 + BB[1] - LO[1]) * PPM).long().clamp(0, RS - 1)
    return (zmin + p[:, None, None] - z[jy, jx]).clamp(min=0)


@torch.no_grad()
def render_error(G, obj, cands, p, img, bg, dev, bs=128):
    """Full-image L1 (images in [0, 1]) between G(mesh depth at each candidate pose) and the real image."""
    ys = torch.linspace(-1, 1, 224, device=dev).view(224, 1).expand(224, 224)
    xs = torch.linspace(-1, 1, 224, device=dev).view(1, 224).expand(224, 224)
    out = []
    for i in range(0, len(cands), bs):
        c = cands[i:i + bs]; n = len(c)
        d = mesh_depth_t(obj, c[:, 0], c[:, 1], c[:, 2], torch.full((n,), p, device=dev), dev) / 1000 / DEPTH_SCALE * 2 - 1
        x = torch.stack([d, xs.expand(n, -1, -1), ys.expand(n, -1, -1)], 1)
        if bg is not None:
            x = torch.cat([x, bg[None].expand(n, -1, -1, -1)], 1)
        with torch.autocast("cuda"):
            r = G(x)
        out.append((r.float() - img[None]).abs().mean((1, 2, 3)) / 2)
    return torch.cat(out)


def teacher_poses(run, prefix, dev):
    """Planar pose of every train-split real frame from the teacher's best_pose.pt: {tactile path: (rz, x_mm, y_mm)}."""
    from vistacfusion.data.dataset import SimVisuoTactileDataset, build_datasets
    from vistacfusion.engine.train import load_checkpoint
    from vistacfusion.models.model import build_model
    from vistacfusion.utils.config import merge_configs
    cfg = merge_configs(MODEL, TRAIN, f"ablation/pilot_robosoft/data_{run}.yaml")
    with contextlib.redirect_stdout(io.StringIO()):
        _, val_ds = build_datasets(cfg)
        model = build_model(cfg).to(dev).eval()
        load_checkpoint(osp.join(OUT, prefix + run, "best_pose.pt"), model, device=dev)
        cfg.real.train_samples_per_session = 10 ** 6
        cfg.real.root = REAL
        ds = SimVisuoTactileDataset(cfg, cfg.image_size, augment=False, split="train", data_section="real",
                                    shared_obj_map=val_ds._obj_to_id)
    poses = {}; i0 = 0
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=32, num_workers=6):
            se2 = model(b["rgb"].to(dev), b["tactile"].to(dev), config="tactile", object_ids=b["object"].to(dev))["se2"].float().cpu()
            for j in range(len(se2)):
                unit, sidx = ds.samples[i0 + j]; half = ds.unit_meta[unit]["half"]
                c, s, tx, ty = (float(v) for v in se2[j])
                poses[osp.join(unit, "samples", f"{sidx:04d}.png")] = (math.atan2(s, c), tx * half * 1e3, ty * half * 1e3)
            i0 += len(se2)
    del model
    torch.cuda.empty_cache()
    return poses


def quality(pred, gt):
    pm, gm = pred > 0.05, gt > 0
    return float((pm & gm).sum() / max(1, (pm | gm).sum())), float(np.abs(pred - gt)[gm].mean()) if gm.any() else float("nan")


def link(src, dst):
    if osp.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src) if osp.islink(src) else src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True, help="pseudo root written by pseudo_label.py (teacher depth, --pose --link-val)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--run", required=True, help="teacher run for the pose init, e.g. B_k25_gank")
    ap.add_argument("--prefix", default="pilot150_")
    ap.add_argument("--gan", required=True, help="dir of the K-shot learned renderer (G_final.pt); 'bgcond' in the name = bg-conditioned")
    ap.add_argument("--keep-frac", type=float, default=0.5, help="per object, fraction of frames (lowest render error) that get mesh labels")
    ap.add_argument("--rest", default="teacher", choices=["teacher", "drop"], help="what the other frames get")
    ap.add_argument("--no-render", action="store_true", help="ablation: coarse fit only, rank by agreement with the teacher depth")
    ap.add_argument("--fits", default=None, help="reuse the fits of a previous run (rc_fits.json) and only re-gate / re-write")
    ap.add_argument("--teacher-pose", action="store_true",
                    help="ablation: frames that keep the teacher depth also get the teacher's (best_pose.pt) pose as a regular label")
    ap.add_argument("--pose-all", action="store_true",
                    help="frames that keep the teacher depth still get the fitted pose as a regular label (the fit is "
                         "more accurate than the teacher pose on every frame; only its dense label is gated)")
    ap.add_argument("--aligner", default=None, help="aligner.pt of train_aligner.py: fit the pose with the learned residual aligner "
                                                    "instead of the coarse + render-and-compare search")
    ap.add_argument("--aligner-iters", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-gt", action="store_true", help="report label quality against the withheld labels (evaluation only)")
    args = ap.parse_args()
    os.chdir(ROOT)
    dev = args.device
    man = json.load(open(osp.join(args.src_root, "pseudo_manifest.json")))["manifest"]
    frames = [t for t, k, _ in man if k == "pseudo"]
    if args.limit:
        frames = frames[::max(1, len(frames) // args.limit)]
    cache = json.load(open(args.fits)) if args.fits else None
    tp = {k: tuple(v["tp"]) for k, v in cache.items()} if cache else teacher_poses(args.run, args.prefix, dev)
    items = []
    for tac in frames:
        unit = osp.dirname(osp.dirname(tac)); idx = osp.basename(tac)[:4]; obj = unit.split("/")[-3]
        items.append(dict(tac=tac, obj=obj, idx=idx, src_unit=unit.replace(REAL, args.src_root), out_unit=unit.replace(REAL, args.out_root)))
    if cache:
        for it in items:
            it["pose"], it["p"], it["err"] = tuple(cache[it["tac"]]["pose"]), cache[it["tac"]]["p"], cache[it["tac"]]["err"]
        for o in sorted({it["obj"] for it in items}):
            B.get_zmap(o)
    elif args.aligner:
        # learned residual aligner (train_aligner.py): iterate the predicted pose correction from the teacher pose
        from train_aligner import Aligner, refine
        for o in sorted({it["obj"] for it in items}):
            B.get_zmap(o)
        A = Aligner().to(dev).eval(); A.load_state_dict(torch.load(args.aligner, map_location=dev))
        G = UNetG(in_ch=3).to(dev).eval(); G.load_state_dict(torch.load(osp.join(args.gan, "G_final.pt"), map_location=dev))
        print(f"{len(items)} pseudo frames, teacher={args.prefix}{args.run}, aligner={args.aligner}", flush=True)
        for i0 in range(0, len(items), 64):
            its = items[i0:i0 + 64]
            img = torch.from_numpy(np.stack([np.array(Image.open(it["tac"]).convert("RGB"), np.float32).transpose(2, 0, 1) / 127.5 - 1 for it in its])).to(dev)
            p0 = [float(np.clip(np.load(osp.join(it["src_unit"], "raw_data", it["idx"] + "_gt.npy")).max() * 1000, 0.3, 1.6)) for it in its]
            pz = torch.tensor([[*tp[it["tac"]], q] for it, q in zip(its, p0)], dtype=torch.float32, device=dev)
            pz = refine(A, [it["obj"] for it in its], img, pz, dev, iters=args.aligner_iters)
            for j, it in enumerate(its):
                it["pose"] = tuple(pz[j, :3].tolist()); it["p"] = float(pz[j, 3])
                it["err"] = float(render_error(G, it["obj"], pz[j:j + 1, :3], it["p"], img[j], None, dev)[0])   # only used by the gate
    else:
        for o in sorted({it["obj"] for it in items}):
            B.get_zmap(o)                                         # build z-maps before forking
        print(f"{len(items)} pseudo frames, teacher={args.prefix}{args.run}, renderer={args.gan}", flush=True)
        with ProcessPoolExecutor(args.workers) as ex:
            fits = list(ex.map(coarse, [(osp.join(it["src_unit"], "raw_data", it["idx"] + "_gt.npy"), it["obj"], *tp[it["tac"]]) for it in items], chunksize=8))
        bg_cond = "bgcond" in args.gan
        if not args.no_render:
            G = UNetG(in_ch=3 + 3 * bg_cond).to(dev).eval()
            G.load_state_dict(torch.load(osp.join(args.gan, "G_final.pt"), map_location=dev))
        grid = [(math.radians(a), dx, dy) for a in np.arange(-3, 3.01, 1.0) for dx in np.arange(-0.6, 0.601, 0.15) for dy in np.arange(-0.6, 0.601, 0.15)]
        grid = torch.tensor(grid, dtype=torch.float32, device=dev)
        for n, (it, (c, p)) in enumerate(zip(items, fits)):
            it["p"] = p
            if args.no_render:
                Dt = np.load(osp.join(it["src_unit"], "raw_data", it["idx"] + "_gt.npy")) * 1000
                it["pose"] = c; it["err"] = float(np.abs(B.gt_depth_mm(it["obj"], *c, p) - Dt).mean())
            else:
                img = torch.from_numpy(np.array(Image.open(it["tac"]).convert("RGB"), np.float32).transpose(2, 0, 1) / 127.5 - 1).to(dev)
                bg = None
                if bg_cond:
                    bg = torch.from_numpy(np.array(Image.open(osp.join(BG_ROOT, "real", f"{it['obj']}__{it['tac'].split('/')[-4]}.png")).convert("RGB"),
                                                   np.float32).transpose(2, 0, 1) / 127.5 - 1).to(dev)
                cands = torch.tensor(c, dtype=torch.float32, device=dev)[None] + grid
                e = render_error(G, it["obj"], cands, p, img, bg, dev)
                it["pose"] = tuple(cands[e.argmin()].tolist()); it["err"] = float(e.min())
            if n % 200 == 0:
                print(f"  {n}/{len(items)}", flush=True)
    os.makedirs(args.out_root, exist_ok=True)
    json.dump({it["tac"]: dict(pose=it["pose"], p=it["p"], err=it["err"], tp=tp[it["tac"]]) for it in items}, open(osp.join(args.out_root, "rc_fits.json"), "w"))
    # per-object gate on the render error
    keep = set()
    for o in sorted({it["obj"] for it in items}):
        its = sorted((it for it in items if it["obj"] == o), key=lambda it: it["err"])
        keep.update(it["tac"] for it in its[:int(round(len(its) * args.keep_frac))])
    # write the root: everything of the source root is linked, pseudo frames are rewritten
    for unit in sorted(glob.glob(osp.join(args.src_root, "*", "session_*", "sensor_0000"))):
        ou = unit.replace(args.src_root, args.out_root)
        os.makedirs(osp.join(ou, "raw_data"), exist_ok=True); os.makedirs(osp.join(ou, "samples"), exist_ok=True)
        for name in ("rgb", "calibration", "id_map.json"):
            if osp.lexists(osp.join(unit, name)):
                link(osp.join(unit, name), osp.join(ou, name))
        if osp.lexists(osp.join(osp.dirname(unit), "session.json")):
            link(osp.join(osp.dirname(unit), "session.json"), osp.join(osp.dirname(ou), "session.json"))
    pseudo = {it["tac"]: it for it in items}
    stats = dict(mesh=0, teacher=0, dropped=0, labeled=0)
    manifest = []
    for tac, kind, val in man:
        if kind not in ("pseudo", "labeled"):
            manifest.append((tac, kind, val)); continue
        unit = osp.dirname(osp.dirname(tac)); idx = osp.basename(tac)[:4]
        su, ou = unit.replace(REAL, args.src_root), unit.replace(REAL, args.out_root)
        dep, pose, img = (osp.join("raw_data", idx + "_gt.npy"), osp.join("raw_data", idx + "_pose.json"), osp.join("samples", idx + ".png"))
        it = pseudo.get(tac)
        if kind == "labeled" or (it is None and not args.limit) or (it is not None and tac not in keep and args.rest == "teacher"):
            for name in (dep, pose, img):
                link(osp.join(su, name), osp.join(ou, name))
            if (args.teacher_pose or args.pose_all) and kind == "pseudo" and it is not None:
                os.remove(osp.join(ou, pose))
                th, x, y = it["pose"] if args.pose_all else tp[tac]
                json.dump({"obj_name": it["obj"], "sample_x": x * 1e-3, "sample_y": y * 1e-3, "sample_z": it["p"] * 1e-3,
                           "rotation_euler": [0.0, 0.0, th], "rc_pose" if args.pose_all else "teacher_pose": True}, open(osp.join(ou, pose), "w"))
            stats["labeled" if kind == "labeled" else "teacher"] += 1
            manifest.append((tac, kind if kind == "labeled" else "pseudo", it["err"] if it else val)); continue
        if it is None or tac not in keep:
            stats["dropped"] += 1; manifest.append((tac, "drop_rc", it["err"] if it else val)); continue
        th, x, y = it["pose"]
        if osp.lexists(osp.join(ou, dep)):
            os.remove(osp.join(ou, dep))
        np.save(osp.join(ou, dep), (B.gt_depth_mm(it["obj"], th, x, y, it["p"]) / 1000).astype(np.float32))
        if osp.lexists(osp.join(ou, pose)):
            os.remove(osp.join(ou, pose))
        json.dump({"obj_name": it["obj"], "sample_x": x * 1e-3, "sample_y": y * 1e-3, "sample_z": it["p"] * 1e-3,
                   "rotation_euler": [0.0, 0.0, th], "rc": True, "render_err": it["err"]}, open(osp.join(ou, pose), "w"))
        link(osp.join(su, img), osp.join(ou, img))
        stats["mesh"] += 1; manifest.append((tac, "mesh", it["err"]))
    # validation frames (true labels, model selection only) are plain links in the source root
    nval = 0
    for unit in sorted(glob.glob(osp.join(args.src_root, "*", "session_*", "sensor_0000"))):
        ou = unit.replace(args.src_root, args.out_root)
        for p in glob.glob(osp.join(unit, "samples", "*.png")):
            if int(osp.basename(p)[:4]) % 10 == 0:
                idx = osp.basename(p)[:4]
                for name in (osp.join("raw_data", idx + "_gt.npy"), osp.join("raw_data", idx + "_pose.json"), osp.join("samples", idx + ".png")):
                    if osp.lexists(osp.join(unit, name)):
                        link(osp.join(unit, name), osp.join(ou, name)); nval += 1
    stats["val_links"] = nval
    if args.eval_gt:   # evaluation only: quality of the labels against the withheld ones
        rows = {"teacher (all)": [], "mesh (kept)": [], "teacher (kept)": [], "mesh (rejected)": [], "teacher (rejected)": []}
        for it in items:
            gu = osp.dirname(osp.dirname(it["tac"]))
            gt = np.load(osp.join(gu, "raw_data", it["idx"] + "_gt.npy")) * 1000
            P = json.load(open(osp.join(gu, "raw_data", it["idx"] + "_pose.json")))
            Dt = np.load(osp.join(it["src_unit"], "raw_data", it["idx"] + "_gt.npy")) * 1000
            rot = lambda th: abs((math.degrees(th - P["rotation_euler"][2]) + 180) % 360 - 180)  # noqa: E731
            t = quality(Dt, gt) + (rot(tp[it["tac"]][0]),)
            m = quality(B.gt_depth_mm(it["obj"], *it["pose"], it["p"]), gt) + (rot(it["pose"][0]),)
            rows["teacher (all)"].append(t)
            k = "kept" if it["tac"] in keep else "rejected"
            rows[f"mesh ({k})"].append(m); rows[f"teacher ({k})"].append(t)
        print(f"{'labels':20s} {'n':>5s} {'IoU':>6s} {'cMAE':>6s} {'rot':>6s}")
        for k, v in rows.items():
            if v:
                v = np.array(v); print(f"{k:20s} {len(v):5d} {np.nanmean(v[:, 0]):6.3f} {np.nanmean(v[:, 1]):6.3f} {np.nanmean(v[:, 2]):6.2f}")
                stats["q_" + k] = [float(np.nanmean(v[:, i])) for i in range(3)]
    print(json.dumps(stats, indent=1), flush=True)
    json.dump({"stats": stats, "manifest": manifest, "args": vars(args)}, open(osp.join(args.out_root, "pseudo_manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
