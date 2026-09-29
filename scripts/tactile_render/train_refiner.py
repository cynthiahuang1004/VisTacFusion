"""Contact-preserving unpaired refiner: Blender render -> real-looking tactile image, ZERO labeled pairs.

Motivation (paper findings): pose transfers through the contact geometry, dense depth needs the
appearance inside the contact, and the frozen encoder is robust to the illumination field.  So:
  * geometry is frozen by construction: output = m * G(x) + (1 - m) * real session background, where
    m is the (feathered) contact mask from the simulated GT depth -> depth/normal/pose labels stay valid;
  * only the contact shading is learned, from UNLABELED real frames (no depth, no pose), with a
    PatchGAN + PatchNCE (CUT, Park et al. ECCV'20) content loss so the ridge structure of the Blender
    render is kept while its shading is translated; optional SimGAN-style self-regularization.
Real budget: unlabeled frames of the allowed objects + one no-contact frame per session.  Leakage rule
as in the paper: --exclude-objects removes the held-out objects' frames entirely; --per-object K keeps
the same K evenly spaced train frames as the downstream model (val frames never seen).

usage:
  python scripts/tactile_render/train_refiner.py --out outputs/tactile_refiner_loo \
      --exclude-objects pattern_04_3_lines_angle_2 pattern_31_rod pattern_33 --device cuda:0
"""
import argparse
import glob
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from train_tactile_gan import BG_ROOT, PHYS_ROOT, REAL_ROOT, SIM_ROOT, VAL_EVERY, UNetG, down  # noqa: E402

OBJECTS = sorted(d for d in os.listdir(REAL_ROOT) if osp.isdir(osp.join(REAL_ROOT, d)))
_bg_cache = {}


def load_img(p):
    return torch.from_numpy((np.array(Image.open(p).convert("RGB"), np.float32) / 127.5 - 1).transpose(2, 0, 1))


def session_bg(obj, bg_tag="real"):
    key = f"{obj}__session_000"
    if key not in _bg_cache:
        _bg_cache[key] = load_img(osp.join(BG_ROOT, bg_tag, key + ".png"))
    return _bg_cache[key]


def contact_mask(depth, dilate=4, feather=2):
    """depth [1,H,W] metres -> soft contact mask [1,H,W] in [0,1]: dilate by `dilate` px, feather edges."""
    m = (depth > 0).float()[None]
    if dilate > 0:
        m = F.max_pool2d(m, 2 * dilate + 1, 1, dilate)
    if feather > 0:
        m = F.avg_pool2d(m, 2 * feather + 1, 1, feather)
    return m[0]


class SimRenders(Dataset):
    """Blender renders + GT depth of the 12 pattern objects (the meshes are available for every object)."""

    def __init__(self, root=SIM_ROOT, objects=OBJECTS, max_items=None, seed=0):
        items = []
        for obj in objects:
            for tac in glob.glob(f"{root}/{obj}/session_*/sensor_0000/samples/*.png"):
                dep = osp.join(osp.dirname(osp.dirname(tac)), "raw_data", osp.basename(tac).replace(".png", "_gt.npy"))
                if osp.exists(dep):
                    items.append((obj, dep, tac))
        items.sort()
        random.Random(seed).shuffle(items)
        self.items = items[:max_items] if max_items else items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        obj, dep, tac = self.items[i]
        d = torch.from_numpy(np.load(dep).astype(np.float32))[None]
        return load_img(tac), session_bg(obj), contact_mask(d)


class RealUnlabeled(Dataset):
    """Unlabeled real frames (images only).  Same split / K-shot rule as RealPairs, no depth needed."""

    def __init__(self, root=REAL_ROOT, exclude_objects=(), per_object=None, split="train"):
        per_unit = {}
        for tac in sorted(glob.glob(f"{root}/*/session_*/sensor_0000/samples/*.png")):
            idx = int(osp.splitext(osp.basename(tac))[0])
            if (split == "train") == (idx % VAL_EVERY == 0):
                continue
            unit = osp.dirname(osp.dirname(tac))
            obj = osp.basename(osp.dirname(osp.dirname(unit)))
            if obj in exclude_objects:
                continue
            per_unit.setdefault(unit, []).append((idx, tac))
        self.items = []
        for unit, lst in per_unit.items():
            lst.sort()
            if per_object is not None and len(lst) > per_object:
                pos = np.linspace(0, len(lst) - 1, per_object).round().astype(int)
                lst = [lst[i] for i in sorted(set(pos.tolist()))]
            self.items += [tac for _, tac in lst]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        tac = self.items[i]
        obj = tac.split("/")[-5]
        return load_img(tac), session_bg(obj)


class PhysVal(Dataset):
    """Real val frames with the Blender render at the same GT pose (real_filtered_phys): fidelity proxy of the
    refiner = contact-region L1 between refine(phys render) and the real frame (same metric as the paper)."""

    def __init__(self, objects=OBJECTS):
        self.items = []
        for obj in objects:
            for tac in sorted(glob.glob(f"{REAL_ROOT}/{obj}/session_*/sensor_0000/samples/*.png")):
                idx = int(osp.splitext(osp.basename(tac))[0])
                if idx % VAL_EVERY:
                    continue
                unit = osp.dirname(osp.dirname(tac))
                dep = osp.join(unit, "raw_data", f"{idx:04d}_gt.npy")
                phys = tac.replace(REAL_ROOT, PHYS_ROOT)
                if osp.exists(dep) and osp.exists(phys):
                    self.items.append((obj, dep, phys, tac))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        obj, dep, phys, tac = self.items[i]
        d = torch.from_numpy(np.load(dep).astype(np.float32))[None]
        return load_img(phys), session_bg(obj), contact_mask(d), load_img(tac), (d > 0).float()


class RefinerG(UNetG):
    """UNetG on [blender(3), bg(3), mask(1)]; exposes encoder taps for PatchNCE."""

    def __init__(self):
        super().__init__(in_ch=7)

    def encode(self, x):
        e1 = self.d1(x); e2 = self.d2(e1); e3 = self.d3(e2)
        return [e1, e2, e3]


def refine(G, blender, bg, m):
    x = torch.cat([blender, bg, m], 1)
    return m * G(x) + (1 - m) * bg, x


class ImageD(nn.Module):
    """70x70 PatchGAN, spectral norm, image only (unpaired)."""

    def __init__(self):
        super().__init__()
        sn = spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv2d(3, 64, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(256, 512, 4, 1, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(512, 1, 4, 1, 1)))

    def forward(self, img):
        return self.net(img)


class PatchNCE(nn.Module):
    """CUT PatchNCE over the generator's encoder taps; patches sampled inside the contact mask
    with higher probability (the background is pasted, only the contact region is translated)."""

    def __init__(self, chans=(64, 128, 256), dim=256, n_patches=256, tau=0.07):
        super().__init__()
        self.mlps = nn.ModuleList(nn.Sequential(nn.Linear(c, dim), nn.ReLU(True), nn.Linear(dim, dim)) for c in chans)
        self.n, self.tau = n_patches, tau

    def forward(self, feats_src, feats_out, m):
        loss = 0.0
        for mlp, fs, fo in zip(self.mlps, feats_src, feats_out):
            B, C, H, W = fs.shape
            w = F.adaptive_avg_pool2d(m, (H, W)).flatten(1) + 0.15          # [B, HW] mask-biased sampling
            ids = torch.multinomial(w, min(self.n, H * W), replacement=False)  # [B, n]
            fs_ = torch.gather(fs.flatten(2), 2, ids[:, None].expand(-1, C, -1)).transpose(1, 2)  # [B,n,C]
            fo_ = torch.gather(fo.flatten(2), 2, ids[:, None].expand(-1, C, -1)).transpose(1, 2)
            q, k = F.normalize(mlp(fo_), dim=-1), F.normalize(mlp(fs_), dim=-1)
            logits = torch.bmm(q, k.transpose(1, 2)) / self.tau                 # [B,n,n], positives on the diagonal
            tgt = torch.arange(q.shape[1], device=q.device)[None].expand(B, -1)
            loss = loss + F.cross_entropy(logits.flatten(0, 1), tgt.flatten())
        return loss / len(self.mlps)


def grad_l1_masked(a, b, m):
    gy = ((a[..., 1:, :] - a[..., :-1, :]) - (b[..., 1:, :] - b[..., :-1, :])).abs() * m[..., 1:, :]
    gx = ((a[..., :, 1:] - a[..., :, :-1]) - (b[..., :, 1:] - b[..., :, :-1])).abs() * m[..., :, 1:]
    return (gy.sum() + gx.sum()) / (m.sum() * 3 + 1e-6)


@torch.no_grad()
def fidelity(G, dl, dev):
    """Returns (full L1, contact L1) of refine(phys render) vs the real frame, in [0,1] image units."""
    G.eval()
    l1s, cl1s = [], []
    for phys, bg, m, real, hard in dl:
        phys, bg, m, real, hard = (t.to(dev) for t in (phys, bg, m, real, hard))
        y, _ = refine(G, phys, bg, m)
        e = (y - real).abs().mean(1, keepdim=True) / 2
        l1s += e.mean((1, 2, 3)).tolist()
        cl1s += [(e[i][hard[i] > 0].mean().item() if hard[i].any() else float("nan")) for i in range(len(e))]
    G.train()
    return float(np.mean(l1s)), float(np.nanmean(cl1s))


@torch.no_grad()
def save_grid(G, sim, real, phys, path, dev, n=5):
    G.eval()
    rows = []
    for i in np.linspace(0, len(sim) - 1, n).astype(int):
        b, bg, m = (t[None].to(dev) for t in sim[int(i)])
        y, _ = refine(G, b, bg, m)
        r = real[int(i) % len(real)][0][None].to(dev)
        row = [b, bg, m.expand(-1, 3, -1, -1) * 2 - 1, y, r]
        if len(phys):
            p, pbg, pm, pr, _ = (t[None].to(dev) for t in phys[int(i) % len(phys)])
            py, _ = refine(G, p, pbg, pm)
            row += [p, py, pr]
        rows.append(torch.cat(row, 3)[0])
    g = ((torch.cat(rows, 1).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(g).save(path)
    G.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tactile_refiner")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sim-root", default=SIM_ROOT, help="Blender renders to refine (renders_v3 or renders_v3_dr)")
    ap.add_argument("--exclude-objects", nargs="*", default=[], help="objects whose real frames are never used")
    ap.add_argument("--per-object", type=int, default=None, help="K unlabeled train frames per object")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--nce", type=float, default=1.0, help="PatchNCE weight (CUT)")
    ap.add_argument("--reg", type=float, default=0.0, help="SimGAN self-regularization: masked gradient L1 to the Blender render")
    ap.add_argument("--diff-d", action="store_true",
                    help="discriminator sees difference images (image - session background): judges the contact response only")
    ap.add_argument("--dilate", type=int, default=4)
    ap.add_argument("--feather", type=int, default=2)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    global contact_mask
    _cm = contact_mask
    contact_mask = lambda d: _cm(d, args.dilate, args.feather)  # noqa: E731

    sim = SimRenders(root=args.sim_root)
    real = RealUnlabeled(exclude_objects=set(args.exclude_objects), per_object=args.per_object)
    phys = PhysVal()
    print(f"sim renders={len(sim)}  unlabeled real={len(real)}  phys-val pairs={len(phys)}", flush=True)
    G, D, nce = RefinerG().to(dev), ImageD().to(dev), PatchNCE().to(dev)
    if args.eval_only:
        G.load_state_dict(torch.load(osp.join(args.out, "G_final.pt"), map_location=dev))
        l1, cl1 = fidelity(G, DataLoader(phys, 32, num_workers=4), dev)
        print(f"fidelity: L1={l1:.4f} contactL1={cl1:.4f}", flush=True)
        return
    optG = torch.optim.Adam(list(G.parameters()) + list(nce.parameters()), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    dl_sim = DataLoader(sim, args.batch, shuffle=True, num_workers=6, drop_last=True, pin_memory=True)
    dl_real = DataLoader(real, args.batch, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    dl_phys = DataLoader(phys, 32, num_workers=4)
    it_sim, it_real = iter(dl_sim), iter(dl_real)
    log = open(osp.join(args.out, "train.log"), "a")
    acc = dict(g=0.0, d=0.0, nce=0.0, reg=0.0); n = 0
    for step in range(1, args.steps + 1):
        try:
            blender, bg, m = next(it_sim)
        except StopIteration:
            it_sim = iter(dl_sim); blender, bg, m = next(it_sim)
        try:
            r, rbg = next(it_real)
        except StopIteration:
            it_real = iter(dl_real); r, rbg = next(it_real)
        blender, bg, m, r, rbg = blender.to(dev), bg.to(dev), m.to(dev), r.to(dev), rbg.to(dev)
        dfake = (lambda img: (img - bg).clamp(-1, 1)) if args.diff_d else (lambda img: img)
        dreal = (lambda img: (img - rbg).clamp(-1, 1)) if args.diff_d else (lambda img: img)
        scale = 1.0 if step < args.steps // 2 else 1.0 - (step - args.steps // 2) / (args.steps - args.steps // 2)
        for opt in (optG, optD):
            for pg in opt.param_groups:
                pg["lr"] = args.lr * max(scale, 0.02)
        y, x = refine(G, blender, bg, m)
        # D
        optD.zero_grad()
        pr, pf = D(dreal(r)), D(dfake(y.detach()))
        lossD = 0.5 * (bce(pr, torch.ones_like(pr)) + bce(pf, torch.zeros_like(pf)))
        lossD.backward(); optD.step()
        # G
        optG.zero_grad()
        pf = D(dfake(y))
        loss_adv = bce(pf, torch.ones_like(pf))
        loss_nce = nce(G.encode(x), G.encode(torch.cat([y, bg, m], 1)), m) if args.nce > 0 else torch.zeros((), device=dev)
        loss_reg = grad_l1_masked(y, blender, m) if args.reg > 0 else torch.zeros((), device=dev)
        lossG = loss_adv + args.nce * loss_nce + args.reg * loss_reg
        lossG.backward(); optG.step()
        acc["g"] += loss_adv.item(); acc["d"] += lossD.item(); acc["nce"] += loss_nce.item(); acc["reg"] += loss_reg.item(); n += 1
        if step % 200 == 0:
            msg = f"[step {step:5d}] adv={acc['g']/n:.3f} D={acc['d']/n:.3f} nce={acc['nce']/n:.3f} reg={acc['reg']/n:.4f}"
            if step % 1000 == 0 or step == args.steps:
                l1, cl1 = fidelity(G, dl_phys, dev)
                msg += f"  fidelity L1={l1:.4f} contactL1={cl1:.4f}"
                save_grid(G, sim, real, phys, osp.join(args.out, f"grid_{step:05d}.png"), dev)
                torch.save(G.state_dict(), osp.join(args.out, "G_latest.pt"))
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            acc = dict(g=0.0, d=0.0, nce=0.0, reg=0.0); n = 0
    torch.save(G.state_dict(), osp.join(args.out, "G_final.pt"))
    torch.save(D.state_dict(), osp.join(args.out, "D_final.pt"))
    l1, cl1 = fidelity(G, dl_phys, dev)
    print(f"final fidelity: L1={l1:.4f} contactL1={cl1:.4f}", flush=True)
    log.write(f"final fidelity: L1={l1:.4f} contactL1={cl1:.4f}\n"); log.close()


if __name__ == "__main__":
    main()
