"""Learned residual aligner for pseudo-label verification (DeepIM-style, K-shot).

Input: a real tactile frame and the mesh depth at a candidate planar pose.  Output: the pose correction
(d_theta, dx, dy, d_press) that brings the candidate onto the frame.  Trained without any label of the unlabeled
frames:
  * the K labeled real frames per object, with their true pose perturbed, and
  * synthetic frames rendered on the fly by the K-shot learned renderer from the mesh depth at random poses
    (pose range of each object taken from its K labeled frames), perturbed the same way.
At labeling time the correction is applied iteratively, starting from the teacher pose.

usage: python scripts/tactile_render/train_aligner.py --k 25 --gan outputs/tactile_gan_k25 --out outputs/aligner_k25 \
           [--eval-root .../real_pseudo_k25 --eval-run B_k25_gank]     (label quality vs withheld GT, evaluation only)
"""
import argparse
import json
import math
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, osp.join(ROOT, "scripts")); sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import pseudo_label_rc as RC  # noqa: E402
from pseudo_label_rc import B, REAL, mesh_depth_t, quality  # noqa: E402
from train_tactile_gan import DEPTH_SCALE, RealPairs, UNetG  # noqa: E402

SCALE = torch.tensor([math.radians(8.0), 1.5, 1.5, 0.15])     # perturbation half-ranges: rad, mm, mm, mm


class Aligner(nn.Module):
    def __init__(self, pretrained=True, arch="resnet18"):
        super().__init__()
        m = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1" if pretrained else None)
        w = m.conv1.weight.data
        m.conv1 = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        m.conv1.weight.data = torch.cat([w, w.mean(1, keepdim=True)], 1)
        m.fc = nn.Linear(m.fc.in_features, 4)
        self.net = m

    def forward(self, img, cand_mm):
        return self.net(torch.cat([img, (cand_mm / 1000 / DEPTH_SCALE * 2 - 1)[:, None]], 1))


def depth_batch(objs, pose, dev):
    """Mesh depth (mm) for a batch of (object name, [theta, x, y, p])."""
    out = torch.zeros(len(objs), 224, 224, device=dev)
    for o in set(objs):
        j = torch.tensor([i for i, n in enumerate(objs) if n == o], device=dev)
        out[j] = mesh_depth_t(o, pose[j, 0], pose[j, 1], pose[j, 2], pose[j, 3], dev)
    return out


def render(G, d_mm, dev):
    ys = torch.linspace(-1, 1, 224, device=dev).view(224, 1).expand(224, 224)
    xs = torch.linspace(-1, 1, 224, device=dev).view(1, 224).expand(224, 224)
    n = len(d_mm)
    x = torch.stack([d_mm / 1000 / DEPTH_SCALE * 2 - 1, xs.expand(n, -1, -1), ys.expand(n, -1, -1)], 1)
    with torch.no_grad(), torch.autocast("cuda"):
        return G(x).float()


def perturb(n, dev):
    d = (torch.rand(n, 4, device=dev) * 2 - 1) * SCALE.to(dev)
    small = torch.rand(n, 1, device=dev) < 0.5                  # half of the samples: fine corrections
    return torch.where(small, d * 0.3, d)


@torch.no_grad()
def refine(A, objs, img, pose, dev, iters=5):
    for _ in range(iters):
        pose = pose + A(img, depth_batch(objs, pose, dev)) * SCALE.to(dev)
        pose[:, 3] = pose[:, 3].clamp(0.3, 1.6)
    return pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--gan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs-real", type=int, default=24)
    ap.add_argument("--bs-sim", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--exclude-objects", nargs="*", default=[], help="leave-object-out: objects whose labeled frames the aligner never sees")
    ap.add_argument("--real-root", default=REAL, help="labeled real root (default: sensor A real_filtered)")
    ap.add_argument("--view", type=float, default=0.017502, help="gel view (m) covered by a stored frame (sensor B: 0.012)")
    ap.add_argument("--pose-json", default=None, help="json {tactile path: [theta, x_mm, y_mm, p_mm]} overriding the frames' _pose.json "
                                                      "(sensor B: mesh poses fitted to the labeled depth, scripts/tactile_render/fit_mesh_pose.py)")
    ap.add_argument("--range-scale", type=float, default=1.0, help="sensitivity: multiply the perturbation ranges (SCALE)")
    ap.add_argument("--arch", default="resnet18", help="sensitivity: torchvision backbone (resnet18 / resnet34 / resnet50)")
    ap.add_argument("--no-pretrain", action="store_true", help="sensitivity: ResNet-18 from scratch")
    ap.add_argument("--extra-root", default=None,
                    help="self-training round: a root labeled by a previous aligner (pseudo_label_rc.py --aligner); its mesh-labeled "
                         "frames join the real training frames with their fitted pose")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-root", default=None, help="pseudo root (teacher depth) whose pseudo frames are refined and scored against GT")
    ap.add_argument("--eval-poses", default=None, help="json {tactile path: [rz, x_m, y_m]} of teacher poses (best_pose.pt)")
    ap.add_argument("--eval-step", type=int, default=12)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT); os.makedirs(args.out, exist_ok=True)
    globals()["SCALE"] = SCALE * args.range_scale
    dev = args.device
    torch.manual_seed(0); np.random.seed(0)
    # ---- K labeled real frames ----
    B.VIEW = args.view
    rp = RealPairs(root=args.real_root, split="train", per_object=args.k if args.k > 0 else None, exclude_objects=tuple(args.exclude_objects))
    r_img, r_pose, r_obj = [], [], []
    PJ = json.load(open(args.pose_json)) if args.pose_json else None
    for dep, tac in rp.items:
        if PJ is not None and tac not in PJ:
            continue
        P = json.load(open(dep.replace("_gt.npy", "_pose.json")))
        r_img.append(np.array(Image.open(tac).convert("RGB"), np.float32).transpose(2, 0, 1) / 127.5 - 1)
        r_pose.append(PJ[tac][:4] if PJ is not None else [P["rotation_euler"][2], P["sample_x"] * 1e3, P["sample_y"] * 1e3, P["sample_z"] * 1e3])
        r_obj.append(tac.split("/")[-5])
    n_true = len(r_obj)
    if args.extra_root:
        for tac, kind, _ in json.load(open(osp.join(args.extra_root, "pseudo_manifest.json")))["manifest"]:
            if kind != "mesh":
                continue
            P = json.load(open(osp.join(osp.dirname(osp.dirname(tac)).replace(REAL, args.extra_root), "raw_data", osp.basename(tac)[:4] + "_pose.json")))
            r_img.append(np.array(Image.open(tac).convert("RGB"), np.float32).transpose(2, 0, 1) / 127.5 - 1)
            r_pose.append([P["rotation_euler"][2], P["sample_x"] * 1e3, P["sample_y"] * 1e3, P["sample_z"] * 1e3])
            r_obj.append(tac.split("/")[-5])
        print(f"+ {len(r_obj) - n_true} aligner-labeled frames from {args.extra_root}", flush=True)
    r_img = torch.from_numpy(np.stack(r_img)).to(dev); r_pose = torch.tensor(r_pose, device=dev)
    objs = sorted(set(r_obj))
    for o in objs:
        B.get_zmap(o)
    rng = {}                                                    # per-object pose range of the labeled frames (+ margin)
    for o in objs:
        p = r_pose[[i for i, n in enumerate(r_obj[:n_true]) if n == o]]
        lo, hi = p.min(0).values, p.max(0).values
        m = torch.tensor([math.radians(10), 1.0, 1.0, 0.05], device=dev)
        rng[o] = (lo - m, hi + m)
    print(f"{n_true} labeled real frames, {len(objs)} objects", flush=True)
    G = UNetG(in_ch=3).to(dev).eval(); G.load_state_dict(torch.load(osp.join(args.gan, "G_final.pt"), map_location=dev))
    A = Aligner(pretrained=not args.no_pretrain, arch=args.arch).to(dev)
    ck = osp.join(args.out, "aligner.pt")
    if args.eval_only:
        A.load_state_dict(torch.load(ck, map_location=dev))
    else:
        opt = torch.optim.AdamW(A.parameters(), lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
        A.train()
        for step in range(args.steps):
            # real: true pose perturbed
            ri = torch.randint(0, len(r_obj), (args.bs_real,), device=dev)
            ro = [r_obj[i] for i in ri.tolist()]; rpz = r_pose[ri]; rim = r_img[ri]
            rim = rim * (1 + 0.1 * (torch.rand(args.bs_real, 1, 1, 1, device=dev) - 0.5)) + 0.05 * (torch.rand(args.bs_real, 3, 1, 1, device=dev) - 0.5)
            # synthetic: random pose in the object's labeled range, rendered by the K-shot renderer
            so = [objs[i] for i in np.random.randint(0, len(objs), args.bs_sim)]
            if args.bs_sim:
                lo = torch.stack([rng[o][0] for o in so]); hi = torch.stack([rng[o][1] for o in so])
                spz = lo + torch.rand(args.bs_sim, 4, device=dev) * (hi - lo)
            else:
                spz = torch.zeros(0, 4, device=dev)
            sim = render(G, depth_batch(so, spz, dev), dev) if args.bs_sim else torch.zeros(0, 3, 224, 224, device=dev)
            ob = ro + so; pz = torch.cat([rpz, spz]); im = torch.cat([rim, sim])
            d = perturb(len(ob), dev)
            pz2 = pz + d; pz2[:, 3] = pz2[:, 3].clamp(0.3, 1.6); d = pz2 - pz
            pred = A(im, depth_batch(ob, pz2, dev))
            loss_all = nn.functional.smooth_l1_loss(pred, -d / SCALE.to(dev), beta=0.1, reduction="none").mean(1)
            loss = loss_all.mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            if step % 500 == 0 or step == args.steps - 1:
                print(f"step {step:5d}  loss {loss.item():.4f}  real {loss_all[:args.bs_real].mean().item():.4f}  sim {loss_all[args.bs_real:].mean().item():.4f}", flush=True)
        torch.save(A.state_dict(), ck)
    A.eval()
    if not args.eval_root:
        return
    # ---- evaluation only: refine the teacher pose of unlabeled frames, score the mesh label against the withheld GT ----
    man = json.load(open(osp.join(args.eval_root, "pseudo_manifest.json")))["manifest"]
    frames = [t for t, k, _ in man if k == "pseudo"][::args.eval_step]
    TP = json.load(open(args.eval_poses))
    rows = {}
    def add(k, v): rows.setdefault(k, []).append(v)
    for i0 in range(0, len(frames), 64):
        fr = frames[i0:i0 + 64]; ob = [t.split("/")[-5] for t in fr]
        img = torch.from_numpy(np.stack([np.array(Image.open(t).convert("RGB"), np.float32).transpose(2, 0, 1) / 127.5 - 1 for t in fr])).to(dev)
        Dt = [np.load(osp.join(osp.dirname(osp.dirname(t)).replace(REAL, args.eval_root), "raw_data", osp.basename(t)[:4] + "_gt.npy")) * 1000 for t in fr]
        p0 = torch.tensor([[TP[t][0], TP[t][1] * 1e3, TP[t][2] * 1e3, float(np.clip(d.max(), 0.3, 1.6))] for t, d in zip(fr, Dt)], device=dev)
        outs = {"teacher pose": p0}
        for it in (1, 3, 6):
            outs[f"aligner x{it}"] = refine(A, ob, img, p0.clone(), dev, iters=it)
        for j, t in enumerate(fr):
            gu = osp.dirname(osp.dirname(t)); idx = osp.basename(t)[:4]
            gt = np.load(osp.join(gu, "raw_data", idx + "_gt.npy")) * 1000; P = json.load(open(osp.join(gu, "raw_data", idx + "_pose.json")))
            add("teacher depth", quality(Dt[j], gt) + (float("nan"), float("nan")))
            for k, pz in outs.items():
                th, x, y, p = pz[j].tolist()
                rot = abs((math.degrees(th - P["rotation_euler"][2]) + 180) % 360 - 180)
                add("mesh @ " + k, quality(B.gt_depth_mm(ob[j], th, x, y, p), gt) + (rot, abs(p - P["sample_z"] * 1e3)))
    print(f"n={len(frames)}")
    print(f"{'labels':24s} {'IoU':>6s} {'cMAE':>6s} {'rot':>6s} {'rot_med':>7s} {'|dp|':>6s}")
    for k, v in rows.items():
        v = np.array(v); print(f"{k:24s} {np.nanmean(v[:, 0]):6.3f} {np.nanmean(v[:, 1]):6.3f} {np.nanmean(v[:, 2]):6.2f} {np.nanmedian(v[:, 2]):7.2f} {np.nanmean(v[:, 3]):6.3f}")
    v = np.array(rows["mesh @ aligner x6"]); print("IoU quantiles (aligner x6):", np.quantile(v[:, 0], [.1, .25, .5, .75, .9]).round(3))


if __name__ == "__main__":
    main()
