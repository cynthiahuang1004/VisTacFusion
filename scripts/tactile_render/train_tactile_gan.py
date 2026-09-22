"""Train a depth->tactile renderer (pix2pix) on real_filtered's exact pairs.

Learns the REAL sensor's optical response G: depth map -> tactile image from
the (depth GT, tactile) pairs that every real press provides (pixel-aligned,
same press). Applied later to sim depth GT to re-render sim tactile images
in the real sensor's style (see generate_tactile.py).

Recipe follows Church et al. CoRL 2021: U-Net generator + PatchGAN
discriminator with spectral norm, L1(x100) + adversarial loss.
Train split only (idx % 10 != 0) — val real images never touch G.

Usage:
    python scripts/tactile_render/train_tactile_gan.py \
        --out outputs/tactile_gan --device cuda:1
"""
import argparse
import glob
import os
import os.path as osp
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset

REAL_ROOT = "/media/hdd2/ihsuan/gs_blender/real_filtered"
BG_ROOT = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), "outputs", "session_bg")
SIM_ROOT = "/media/hdd2/ihsuan/gs_blender/renders_v3"
DEPTH_SCALE = 0.0012  # max press depth (m); normalizes depth to ~[0, 1]
VAL_EVERY = 10


def with_coords(d):
    """Concat x,y coordinate channels (CoordConv): the illumination field is a
    function of position; without coords G must guess it from zero background."""
    _, H, W = d.shape
    ys = torch.linspace(-1, 1, H).view(H, 1).expand(H, W)
    xs = torch.linspace(-1, 1, W).view(1, W).expand(H, W)
    return torch.cat([d, xs[None], ys[None]], 0)


class RealPairs(Dataset):
    diff = False      # True: target = tactile - session background (difference image)
    bg_tag = "real"
    _bg_cache = {}

    def _bg(self, tac):
        """Per-session no-contact background in [-1, 1] (compute_session_bg.py)."""
        unit = osp.dirname(osp.dirname(tac))
        key = f"{unit.split('/')[-3]}__{unit.split('/')[-2]}"
        if key not in self._bg_cache:
            im = Image.open(osp.join(BG_ROOT, self.bg_tag, key + ".png")).convert("RGB")
            self._bg_cache[key] = np.array(im, np.float32) / 127.5 - 1
        return self._bg_cache[key]

    def __init__(self, root=REAL_ROOT, split="train", exclude_objects=(), per_object=None):
        """exclude_objects: leave-object-out (held-out objects never seen by G).
        per_object: keep K evenly spaced train pairs per session — the SAME linspace
        rule as SimVisuoTactileDataset.train_samples_per_session, so G and the
        downstream model see the identical K-shot real subset."""
        self.items = []
        per_unit = {}
        for tac in sorted(glob.glob(f"{root}/*/session_*/sensor_0000/samples/*.png")):
            idx = int(osp.splitext(osp.basename(tac))[0])
            is_val = idx % VAL_EVERY == 0
            if (split == "train") == is_val:
                continue
            unit = osp.dirname(osp.dirname(tac))
            obj = osp.basename(osp.dirname(osp.dirname(unit)))
            if obj in exclude_objects:
                continue
            per_unit.setdefault(unit, []).append((idx, tac))
        for unit, lst in per_unit.items():
            lst.sort()
            if split == "train" and per_object is not None and len(lst) > per_object:
                pos = np.linspace(0, len(lst) - 1, per_object).round().astype(int)
                lst = [lst[i] for i in sorted(set(pos.tolist()))]
            for idx, tac in lst:
                dep = osp.join(unit, "raw_data", f"{idx:04d}_gt.npy")
                if not osp.exists(dep):
                    dep = osp.join(unit, "raw_data", f"{idx:04d}.npy")
                if osp.exists(dep):
                    self.items.append((dep, tac))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        dep, tac = self.items[i]
        d = np.load(dep).astype(np.float32) / DEPTH_SCALE * 2 - 1
        t = np.array(Image.open(tac).convert("RGB"), np.float32) / 127.5 - 1
        if self.diff:
            t = np.clip(t - self._bg(tac), -1, 1)
        x = with_coords(torch.from_numpy(d)[None])
        return x, torch.from_numpy(t.transpose(2, 0, 1))


class SimPairs(RealPairs):
    """(Blender GT depth -> Blender-rendered tactile) pairs from the calibrated simulator.
    Used to PRETRAIN G/D on the physics-based renderer (no real data needed); the real
    K-shot pairs then only have to adapt appearance (hybrid renderer)."""

    def __init__(self, root=SIM_ROOT, max_pairs=30000, seed=0):
        items = []
        for tac in glob.glob(f"{root}/pattern_*/session_*/sensor_0000/samples/*.png"):
            unit = osp.dirname(osp.dirname(tac))
            dep = osp.join(unit, "raw_data", osp.basename(tac).replace(".png", "_gt.npy"))
            if osp.exists(dep):
                items.append((dep, tac))
        items.sort()
        random.Random(seed).shuffle(items)
        self.items = items[:max_pairs]


def down(cin, cout, norm=True):
    layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(cout))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up(cin, cout, drop=False):
    # Upsample+Conv instead of ConvTranspose: avoids checkerboard/patch artifacts
    layers = [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
              nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
              nn.InstanceNorm2d(cout), nn.ReLU(inplace=True)]
    if drop:
        layers.append(nn.Dropout(0.5))  # pix2pix noise source
    return nn.Sequential(*layers)


class UNetG(nn.Module):
    """224 -> 7 bottleneck, 5 levels."""

    def __init__(self):
        super().__init__()
        self.d1 = down(3, 64, norm=False)   # 112  (depth + xy coords)
        self.d2 = down(64, 128)             # 56
        self.d3 = down(128, 256)            # 28
        self.d4 = down(256, 512)            # 14
        self.d5 = down(512, 512)            # 7
        self.u1 = up(512, 512, drop=True)           # 14
        self.u2 = up(1024, 256, drop=True)          # 28
        self.u3 = up(512, 128)                      # 56
        self.u4 = up(256, 64)                       # 112
        self.out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 3, 3, 1, 1), nn.Tanh())  # 224

    def forward(self, x):
        e1 = self.d1(x); e2 = self.d2(e1); e3 = self.d3(e2)
        e4 = self.d4(e3); e5 = self.d5(e4)
        y = self.u1(e5)
        y = self.u2(torch.cat([y, e4], 1))
        y = self.u3(torch.cat([y, e3], 1))
        y = self.u4(torch.cat([y, e2], 1))
        return self.out(torch.cat([y, e1], 1))


class PatchD(nn.Module):
    """70x70 PatchGAN with spectral norm (Church's tweak), input = depth+image."""

    def __init__(self):
        super().__init__()
        def sn(c): return spectral_norm(c)
        self.net = nn.Sequential(
            sn(nn.Conv2d(6, 64, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(256, 512, 4, 1, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(512, 1, 4, 1, 1)),
        )

    def forward(self, d, img):
        return self.net(torch.cat([d, img], 1))


def save_grid(G, ds, path, device, n=6):
    G.eval()
    idxs = np.linspace(0, len(ds) - 1, n).astype(int)
    rows = []
    with torch.no_grad():
        for i in idxs:
            d, t = ds[int(i)]
            fake = G(d[None].to(device))[0].cpu()
            dep_vis = (d[:1].repeat(3, 1, 1) + 1) / 2  # depth channel only
            rows.append(torch.cat([dep_vis, (fake + 1) / 2, (t + 1) / 2], dim=2))
    grid = torch.cat(rows, dim=1).clamp(0, 1).numpy().transpose(1, 2, 0)
    Image.fromarray((grid * 255).astype(np.uint8)).save(path)
    G.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tactile_gan")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--l1", type=float, default=100.0)
    ap.add_argument("--exclude-objects", nargs="*", default=[],
                    help="leave-object-out: real objects G never sees")
    ap.add_argument("--per-object", type=int, default=None,
                    help="K-shot: evenly spaced K train pairs per object")
    ap.add_argument("--sim-pretrain", action="store_true",
                    help="train on Blender (depth -> Blender tactile) pairs instead of real")
    ap.add_argument("--init", default=None,
                    help="dir with G_final.pt/D_final.pt to fine-tune from (hybrid renderer)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--diff", action="store_true",
                    help="learn depth -> (tactile - session background) difference images")
    ap.add_argument("--min-steps", type=int, default=0,
                    help="raise epochs so total G updates >= this (small K-shot sets)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    torch.manual_seed(0); random.seed(0); np.random.seed(0)

    RealPairs.diff = args.diff
    if args.sim_pretrain:
        tr = SimPairs()
    else:
        tr = RealPairs(split="train", exclude_objects=set(args.exclude_objects),
                       per_object=args.per_object)
    va = RealPairs(split="val", exclude_objects=set(args.exclude_objects))
    print(f"pairs: train={len(tr)} val={len(va)}", flush=True)
    steps_per_epoch = max(1, len(tr) // args.batch)
    if args.min_steps and args.epochs * steps_per_epoch < args.min_steps:
        args.epochs = -(-args.min_steps // steps_per_epoch)
        print(f"epochs raised to {args.epochs} (min_steps={args.min_steps})", flush=True)
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                    num_workers=8, pin_memory=True, drop_last=True)

    G, D = UNetG().to(dev), PatchD().to(dev)
    if args.init:
        G.load_state_dict(torch.load(osp.join(args.init, "G_final.pt"), map_location=dev))
        D.load_state_dict(torch.load(osp.join(args.init, "D_final.pt"), map_location=dev))
        print(f"initialized G/D from {args.init}", flush=True)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    blur = torch.nn.AvgPool2d(16, 8, 4)

    def lowfreq_l1(a, b):
        return l1(blur(a), blur(b))

    def grad_l1(a, b):
        return (l1(a[..., 1:, :] - a[..., :-1, :], b[..., 1:, :] - b[..., :-1, :])
                + l1(a[..., 1:] - a[..., :-1], b[..., 1:] - b[..., :-1]))

    for ep in range(args.epochs):
        # linear lr decay over the second half (pix2pix schedule)
        scale = 1.0 if ep < args.epochs // 2 else \
            1.0 - (ep - args.epochs // 2) / max(1, args.epochs - args.epochs // 2)
        for opt in (optG, optD):
            for pg in opt.param_groups:
                pg["lr"] = args.lr * max(scale, 0.02)
        gl = dl_ = l1l = 0.0
        for d, t in dl:
            d, t = d.to(dev), t.to(dev)
            fake = G(d)
            # --- D step
            optD.zero_grad()
            pr = D(d, t); pf = D(d, fake.detach())
            lossD = 0.5 * (bce(pr, torch.ones_like(pr)) + bce(pf, torch.zeros_like(pf)))
            lossD.backward(); optD.step()
            # --- G step
            optG.zero_grad()
            pf = D(d, fake)
            loss_l1 = l1(fake, t)
            lossG = (bce(pf, torch.ones_like(pf)) + args.l1 * loss_l1
                     + args.l1 * grad_l1(fake, t)
                     + args.l1 * lowfreq_l1(fake, t))
            lossG.backward(); optG.step()
            gl += lossG.item(); dl_ += lossD.item(); l1l += loss_l1.item()
        n = len(dl)
        print(f"[ep {ep:03d}] G={gl/n:.3f} D={dl_/n:.3f} L1={l1l/n:.4f}", flush=True)
        if ep % 10 == 0 or ep == args.epochs - 1:
            save_grid(G, va, osp.join(args.out, f"val_ep{ep:03d}.png"), dev)
            torch.save(G.state_dict(), osp.join(args.out, "G_latest.pt"))
    torch.save(G.state_dict(), osp.join(args.out, "G_final.pt"))
    torch.save(D.state_dict(), osp.join(args.out, "D_final.pt"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
