"""Test-time pose refinement with the K-shot aligner (train_aligner.py): the model predicts the planar pose, the aligner
iterates its correction against the object mesh.  Reports rotation / translation error before and after on the real
validation split (same session) and on the curated second session.

usage: python ablation/pilot_robosoft/eval_pose_refine.py --device cuda:0 B_k25_al2:pilot150_:25 B_k25_realonly:pilot150_:25 ...
       (run : prefix : K of the aligner outputs/aligner_k{K}/aligner.pt)
"""
import argparse
import contextlib
import copy
import io
import json
import math
import os
import os.path as osp
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, osp.join(ROOT, "scripts")); sys.path.insert(0, osp.join(ROOT, "scripts", "tactile_render"))
from pseudo_label_rc import B  # noqa: E402
from train_aligner import Aligner, refine  # noqa: E402
from vistacfusion.data.dataset import SimVisuoTactileDataset  # noqa: E402
from vistacfusion.engine.train import load_checkpoint  # noqa: E402
from vistacfusion.models.model import build_model  # noqa: E402
from vistacfusion.utils.config import merge_configs  # noqa: E402

OUT = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"
NEW = "/media/hdd2/ihsuan/gs_blender/real_filtered_new_curated_pred_r2_fv"
REAL = "/media/hdd2/ihsuan/gs_blender/real_filtered"


def circ_offset_removed(err_signed, objs):
    """Rotation error after removing each object's circular-median offset (the re-placement of the object)."""
    out = np.zeros_like(err_signed)
    for o in set(objs):
        j = [i for i, n in enumerate(objs) if n == o]
        e = err_signed[j]; med = np.median(e)
        out[j] = np.abs((e - med + 180) % 360 - 180)
    return out


@torch.no_grad()
def run_split(model_p, model_d, A, ds, dev, iters):
    rows = []; i0 = 0
    for b in DataLoader(ds, batch_size=32, num_workers=4):
        n = len(b["tactile"]); units = [ds.samples[i0 + j] for j in range(n)]; i0 += n
        x = b["tactile"].to(dev); oid = b["object"].to(dev)
        se2 = model_p(x, x, config="tactile", object_ids=oid)["se2"].float().cpu()
        dep = model_d(x, x, config="tactile", object_ids=oid)["depth"][:, 0].float().clamp(min=0).amax((1, 2)).cpu()
        objs = [u.split("/")[-3] for u, _ in units]
        half = [ds.unit_meta[u]["half"] * 1e3 for u, _ in units]
        img = torch.from_numpy(np.stack([np.array(Image.open(osp.join(u, "samples", f"{s:04d}.png")).convert("RGB"), np.float32).transpose(2, 0, 1) / 127.5 - 1
                                         for u, s in units])).to(dev)
        p0 = torch.tensor([[math.atan2(float(se2[j, 1]), float(se2[j, 0])), float(se2[j, 2]) * half[j], float(se2[j, 3]) * half[j],
                            float(dep[j].clamp(0.3, 1.6))] for j in range(n)], device=dev)
        p1 = refine(A, objs, img, p0.clone(), dev, iters=iters)
        for j, (u, s) in enumerate(units):
            P = json.load(open(osp.join(u, "raw_data", f"{s:04d}_pose.json")))
            g = (P["rotation_euler"][2], P["sample_x"] * 1e3, P["sample_y"] * 1e3)
            def e(p):
                return ((math.degrees(float(p[0]) - g[0]) + 180) % 360 - 180, (abs(float(p[1]) - g[1]) + abs(float(p[2]) - g[2])) / 2 / half[j])
            rows.append((objs[j],) + e(p0[j]) + e(p1[j]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run:prefix:K")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--json", default=osp.join(ROOT, "ablation/pilot_robosoft/pose_refine_results.json"))
    args = ap.parse_args()
    os.chdir(ROOT); dev = args.device
    res = json.load(open(args.json)) if osp.exists(args.json) else {}
    print(f"{'run':24s} {'split':10s} {'rot':>6s} {'->':>2s} {'rot+A':>6s} {'med':>5s} {'med+A':>6s} {'rotC':>5s} {'rotC+A':>6s} {'trans':>6s} {'tr+A':>6s} {'n':>4s}")
    for spec in args.runs:
        run, prefix, K = spec.split(":")
        cfg = merge_configs("ablation/encoder/tac_sitr_single.yaml", "ablation/pilot_robosoft/train_pilot_e50.yaml", f"ablation/pilot_robosoft/data_{run}.yaml")
        with contextlib.redirect_stdout(io.StringIO()):
            c1 = copy.deepcopy(cfg); c1["real"]["root"] = REAL
            val = SimVisuoTactileDataset(c1, cfg.image_size, augment=False, split="val", data_section="real")
            c2 = copy.deepcopy(cfg); c2["real"]["root"] = NEW; c2["real"]["val_every"] = 1; c2["real"].pop("test_objects", None)
            new = SimVisuoTactileDataset(c2, cfg.image_size, augment=False, split="all", data_section="real", shared_obj_map=val._obj_to_id)
            mp = build_model(cfg).to(dev).eval(); load_checkpoint(osp.join(OUT, prefix + run, "best_pose.pt"), mp, device=dev)
            md = build_model(cfg).to(dev).eval(); load_checkpoint(osp.join(OUT, prefix + run, "best_depth.pt"), md, device=dev)
        A = Aligner().to(dev).eval(); A.load_state_dict(torch.load(f"outputs/aligner_k{K}/aligner.pt", map_location=dev))
        for o in sorted({u.split("/")[-3] for u, _ in val.samples}):
            B.get_zmap(o)
        for name, ds in (("val", val), ("session2", new)):
            rows = run_split(mp, md, A, ds, dev, args.iters)
            objs = [r[0] for r in rows]; v = np.array([r[1:] for r in rows], float)
            r = dict(rot=float(np.abs(v[:, 0]).mean()), rot_al=float(np.abs(v[:, 2]).mean()), rot_med=float(np.median(np.abs(v[:, 0]))), rot_med_al=float(np.median(np.abs(v[:, 2]))),
                     rotc=float(circ_offset_removed(v[:, 0], objs).mean()), rotc_al=float(circ_offset_removed(v[:, 2], objs).mean()),
                     trans=float(v[:, 1].mean()), trans_al=float(v[:, 3].mean()), n=len(rows))
            res[f"{prefix}{run}/{name}"] = r
            print(f"{run:24s} {name:10s} {r['rot']:6.2f} -> {r['rot_al']:6.2f} {r['rot_med']:5.2f} {r['rot_med_al']:6.2f} {r['rotc']:5.2f} {r['rotc_al']:6.2f} {r['trans']:6.4f} {r['trans_al']:6.4f} {r['n']:4d}", flush=True)
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
