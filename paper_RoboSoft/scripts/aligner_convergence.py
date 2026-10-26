"""Aligner convergence and basin of attraction (sensor A, K=25, real-only aligner outputs/aligner_k25_realonly).
(a) From the K-shot real-only teacher's predicted pose on the 472 validation frames: rotation / translation error after
    0..N aligner iterations.  (b) From the GT pose perturbed by a rotation of magnitude r (random sign) plus a random
    in-plane shift (+-1.5 mm) and indentation offset (+-0.15 mm): error after 0..N iterations, for several r.
Writes ablation/pilot_robosoft/aligner_convergence.json.   usage: python paper_RoboSoft/scripts/aligner_convergence.py [--device cuda:0]
"""
import argparse, contextlib, copy, io, json, math, os, os.path as osp, sys
import numpy as np, torch
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

OUT = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"; REAL = "/media/hdd2/ihsuan/gs_blender/real_filtered"
RADII = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30]


def errs(p, g, half):
    """per-frame (|rotation error| deg, translation error mm) of poses p [n,4] against GT g [n,3] (theta rad, x mm, y mm)."""
    rot = (torch.rad2deg(p[:, 0] - g[:, 0]) + 180) % 360 - 180
    tr = ((p[:, 1] - g[:, 1]).abs() + (p[:, 2] - g[:, 2]).abs()) / 2
    return rot.abs().cpu().numpy(), tr.cpu().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--run", default="B_k25_realonly"); ap.add_argument("--aligner", default="outputs/aligner_k25_realonly/aligner.pt")
    ap.add_argument("--out", default="ablation/pilot_robosoft/aligner_convergence.json")
    ap.add_argument("--real-root", default=REAL, help="mirror root of held-out frames (all frames used) instead of the validation split")
    ap.add_argument("--label-metrics", action="store_true", help="also report the mesh label's contact IoU / MAE at iterations 0, 1, 2, 3, 6 (teacher init)")
    a = ap.parse_args()
    os.chdir(ROOT); dev = a.device; torch.manual_seed(0); rng = np.random.RandomState(0)
    cfg = merge_configs("ablation/encoder/tac_sitr_single.yaml", "ablation/pilot_robosoft/train_pilot_e50.yaml", f"ablation/pilot_robosoft/data_{a.run}.yaml")
    with contextlib.redirect_stdout(io.StringIO()):
        c1 = copy.deepcopy(cfg); c1["real"]["root"] = a.real_root
        if a.real_root != REAL:
            c1["real"]["val_every"] = 1; c1["real"].pop("exclude_frames", None)
        val = SimVisuoTactileDataset(c1, cfg.image_size, augment=False, split=("all" if a.real_root != REAL else "val"), data_section="real")
        mp = build_model(cfg).to(dev).eval(); load_checkpoint(osp.join(OUT, "pilot150_" + a.run, "best_pose.pt"), mp, device=dev)
        md = build_model(cfg).to(dev).eval(); load_checkpoint(osp.join(OUT, "pilot150_" + a.run, "best_depth.pt"), md, device=dev)
    A = Aligner().to(dev).eval(); A.load_state_dict(torch.load(a.aligner, map_location=dev))
    for o in sorted({u.split("/")[-3] for u, _ in val.samples}):
        B.get_zmap(o)
    N = a.iters
    teacher = {"rot": [[] for _ in range(N + 1)], "trans": [[] for _ in range(N + 1)], "iou": [[] for _ in range(N + 1)], "cmae": [[] for _ in range(N + 1)]}
    def label_metrics(p, units, objs):
        """mesh depth at the states p (full view) against the stored GT: contact IoU (mesh > 0.05 mm vs GT > 0) and contact MAE over GT contact pixels."""
        iou, cm = [], []
        for j, (u, s_) in enumerate(units):
            gt = np.load(osp.join(u, "raw_data", f"{s_:04d}_gt.npy")) * 1000; m = B.gt_depth_mm(objs[j], *p[j].tolist()); a_, b_ = m > 0.05, gt > 0
            iou.append(float((a_ & b_).sum() / max(1, (a_ | b_).sum()))); cm.append(float(np.abs(m - gt)[b_].mean()) if b_.any() else 0.0)
        return iou, cm
    basin = {r: {"rot": [[] for _ in range(N + 1)], "trans": [[] for _ in range(N + 1)]} for r in RADII}
    i0 = 0
    for b in DataLoader(val, batch_size=32, num_workers=4):
        n = len(b["tactile"]); units = [val.samples[i0 + j] for j in range(n)]; i0 += n
        x = b["tactile"].to(dev); oid = b["object"].to(dev)
        se2 = mp(x, x, config="tactile", object_ids=oid)["se2"].float().cpu()
        dep = md(x, x, config="tactile", object_ids=oid)["depth"][:, 0].float().clamp(min=0).amax((1, 2)).cpu()
        objs = [u.split("/")[-3] for u, _ in units]; half = torch.tensor([val.unit_meta[u]["half"] * 1e3 for u, _ in units])
        img = torch.from_numpy(np.stack([np.array(Image.open(osp.join(u, "samples", f"{s:04d}.png")).convert("RGB"), np.float32).transpose(2, 0, 1) / 127.5 - 1
                                         for u, s in units])).to(dev)
        g, pg = [], []
        for u, s in units:
            P = json.load(open(osp.join(u, "raw_data", f"{s:04d}_pose.json")))
            g.append((P["rotation_euler"][2], P["sample_x"] * 1e3, P["sample_y"] * 1e3))
            pg.append(float(np.clip(np.load(osp.join(u, "raw_data", f"{s:04d}_gt.npy")).max() * 1000, 0.3, 1.6)))
        g = torch.tensor(g, device=dev); pg = torch.tensor(pg, device=dev)
        # (a) teacher initialization
        p = torch.stack([torch.atan2(se2[:, 1], se2[:, 0]), se2[:, 2] * half, se2[:, 3] * half, dep.clamp(0.3, 1.6)], 1).to(dev)
        for it in range(N + 1):
            if it:
                p = refine(A, objs, img, p, dev, iters=1)
            r_, t_ = errs(p, g, half); teacher["rot"][it] += r_.tolist(); teacher["trans"][it] += t_.tolist()
            if a.label_metrics and it in (0, 1, 2, 3, 6):
                i_, c_ = label_metrics(p.cpu(), units, objs); teacher["iou"][it] += i_; teacher["cmae"][it] += c_
        # (b) perturbed GT
        for r in RADII:
            sign = torch.from_numpy(rng.choice([-1.0, 1.0], n)).to(dev)
            d = torch.from_numpy(rng.uniform(-1, 1, (n, 2)) * 1.5).to(dev).float()
            dz = torch.from_numpy(rng.uniform(-0.15, 0.15, n)).to(dev).float()
            p = torch.stack([g[:, 0] + sign * math.radians(r), g[:, 1] + d[:, 0], g[:, 2] + d[:, 1], (pg + dz).clamp(0.3, 1.6)], 1).float()
            for it in range(N + 1):
                if it:
                    p = refine(A, objs, img, p, dev, iters=1)
                r_, t_ = errs(p, g, half); basin[r]["rot"][it] += r_.tolist(); basin[r]["trans"][it] += t_.tolist()
        print(f"{i0}/{len(val)}", flush=True)
    summ = lambda L: dict(mean=[float(np.mean(v)) for v in L], median=[float(np.median(v)) for v in L], frac_gt5=[float(np.mean(np.array(v) > 5)) for v in L])
    res = {"n": len(val), "iters": N, "teacher": {"rot": summ(teacher["rot"]), "trans": summ(teacher["trans"])},
           "label": ({str(it): dict(iou=float(np.mean(teacher["iou"][it])), cmae=float(np.mean(teacher["cmae"][it]))) for it in (0, 1, 2, 3, 6)} if a.label_metrics else None),
           "basin": {str(r): {"rot": summ(basin[r]["rot"]), "trans": summ(basin[r]["trans"])} for r in RADII}}
    json.dump(res, open(a.out, "w"), indent=1)
    print("teacher rot mean per iter:", [round(v, 2) for v in res["teacher"]["rot"]["mean"]])
    for r in RADII:
        print(f"r={r:2d}: rot mean per iter", [round(v, 2) for v in res["basin"][str(r)]["rot"]["mean"]])


if __name__ == "__main__":
    main()
