"""Compute CORAL stats for shared DINOv3 encoder (baseline model config).

Same logic as compute_coral_stats.py but uses DINOv3 for BOTH tactile and RGB
(matching configs/model.yaml's share_encoder_weights: true).

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/compute_coral_stats_dino.py \
        --out pretrained_encoders/coral_stats_dino.pt --device cuda:0
"""
import argparse, glob, json, math, os, os.path as osp, random, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from vistacfusion.models.encoders import DINOv3Encoder

SIM_ROOT = "/media/hdd2/ihsuan/gs_blender/renders_v3"
REAL_ROOT = "/media/hdd2/ihsuan/gs_blender/real_filtered"
WINDOWS = "ablation/simqty_filtered/real_rotation_windows.json"
CROP = 1.0 / math.sqrt(2.0)
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
N_PER_OBJ = 400
REAL_VAL_EVERY = 10
EPS_REL = 0.05


def preprocess(path):
    img = np.array(Image.open(path).convert("RGB"))
    H, W = img.shape[:2]
    s = int(min(H, W) * CROP)
    oy, ox = (H - s) // 2, (W - s) // 2
    img = img[oy:oy + s, ox:ox + s]
    img = np.array(Image.fromarray(img).resize((224, 224), Image.BILINEAR), np.float32)
    return torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1))


def in_window(base_deg, lo, hi):
    center = (lo + hi) / 2.0
    halfwidth = (hi - lo) / 2.0
    d = (base_deg - center) % 360.0
    return min(d, 360.0 - d) <= halfwidth


def collect_paths(windows):
    rng = random.Random(0)
    out = {}
    for obj in sorted(os.listdir(SIM_ROOT)):
        if not osp.isdir(osp.join(SIM_ROOT, obj)):
            continue
        entry = {"sim": None, "real": None}
        idxs = []
        for sdir in sorted(glob.glob(f"{SIM_ROOT}/{obj}/session_*")):
            sj = json.load(open(osp.join(sdir, "session.json")))
            base_deg = math.degrees(sj["base_rotation"][2])
            if obj in windows and not in_window(base_deg, *windows[obj]):
                continue
            for tac in glob.glob(f"{sdir}/sensor_*/samples/*.png"):
                rgb = tac.replace("/samples/", "/rgb/")
                if osp.exists(rgb):
                    idxs.append((tac, rgb))
        if idxs:
            picks = rng.sample(idxs, min(N_PER_OBJ, len(idxs)))
            entry["sim"] = {"samples": [p[0] for p in picks],
                            "rgb": [p[1] for p in picks]}
        idxs = []
        for sdir in sorted(glob.glob(f"{REAL_ROOT}/{obj}/session_*/sensor_*")):
            pngs = sorted(glob.glob(f"{sdir}/samples/*.png"))
            for i, tac in enumerate(pngs):
                if i % REAL_VAL_EVERY == 0:
                    continue
                rgb = tac.replace("/samples/", "/rgb/")
                if osp.exists(rgb):
                    idxs.append((tac, rgb))
        if idxs:
            picks = rng.sample(idxs, min(N_PER_OBJ, len(idxs)))
            entry["real"] = {"samples": [p[0] for p in picks],
                             "rgb": [p[1] for p in picks]}
        out[obj] = entry
    return out


class StatAcc:
    def __init__(self, dim):
        self.n = 0
        self.s = torch.zeros(dim, dtype=torch.float64)
        self.ss = torch.zeros(dim, dim, dtype=torch.float64)

    def add(self, X):
        X = X.double().cpu()
        self.n += X.shape[0]
        self.s += X.sum(0)
        self.ss += X.T @ X

    def mean(self):
        return self.s / self.n

    def cov(self):
        mu = self.mean()
        return self.ss / self.n - torch.outer(mu, mu)


def mat_pow(C, p, eps_rel=EPS_REL):
    d = C.shape[0]
    eps = eps_rel * torch.diagonal(C).mean().clamp(min=1e-8)
    vals, vecs = torch.linalg.eigh(C + eps * torch.eye(d, dtype=C.dtype))
    vals = vals.clamp(min=1e-10)
    return vecs @ torch.diag(vals ** p) @ vecs.T


@torch.no_grad()
def branch_stats(enc, paths_by_obj, subdir, device, batch=32):
    D = enc.embed_dim
    acc = {("global", dom, kind): StatAcc(D)
           for dom in ("sim", "real") for kind in ("patch", "cls")}
    obj_mean = {}

    for obj, entry in paths_by_obj.items():
        for dom in ("sim", "real"):
            if entry[dom] is None:
                continue
            paths = entry[dom][subdir]
            for i in range(0, len(paths), batch):
                x = torch.stack([preprocess(p) for p in paths[i:i + batch]]).to(device)
                patch, cls = enc(x)
                pt = patch.reshape(-1, D).float()
                ct = cls.squeeze(1).float()
                acc[("global", dom, "patch")].add(pt)
                acc[("global", dom, "cls")].add(ct)
                for kind, t in (("patch", pt), ("cls", ct)):
                    key = (obj, dom, kind)
                    if key not in obj_mean:
                        obj_mean[key] = [torch.zeros(D, dtype=torch.float64), 0]
                    obj_mean[key][0] += t.double().cpu().sum(0)
                    obj_mean[key][1] += t.shape[0]
        print(f"    {obj}: sim={'y' if entry['sim'] else '-'} "
              f"real={'y' if entry['real'] else '-'}", flush=True)

    out = {}
    for kind in ("patch", "cls"):
        mu_s = acc[("global", "sim", kind)].mean()
        mu_r = acc[("global", "real", kind)].mean()
        A = mat_pow(acc[("global", "sim", kind)].cov(), -0.5) @ \
            mat_pow(acc[("global", "real", kind)].cov(), 0.5)
        out[f"{kind}_mu_s"] = mu_s.float()
        out[f"{kind}_mu_r"] = mu_r.float()
        out[f"{kind}_A"] = A.float()

    objects = sorted(paths_by_obj.keys())
    n_obj = len(objects)
    for kind in ("patch", "cls"):
        oms = torch.zeros(n_obj, D)
        omr = torch.zeros(n_obj, D)
        for i, obj in enumerate(objects):
            ks, kr = (obj, "sim", kind), (obj, "real", kind)
            oms[i] = (obj_mean[ks][0] / obj_mean[ks][1]).float() \
                if ks in obj_mean else out[f"{kind}_mu_s"]
            omr[i] = (obj_mean[kr][0] / obj_mean[kr][1]).float() \
                if kr in obj_mean else out[f"{kind}_mu_r"]
        out[f"obj_{kind}_mu_s"] = oms
        out[f"obj_{kind}_mu_r"] = omr
    out["obj_has_real"] = torch.tensor(
        [paths_by_obj[o]["real"] is not None for o in objects])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pretrained_encoders/coral_stats_dino.pt")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    windows = json.load(open(WINDOWS))
    paths = collect_paths(windows)
    objects = sorted(paths.keys())
    n_sim = sum(len(e["sim"]["rgb"]) for e in paths.values() if e["sim"])
    n_real = sum(len(e["real"]["rgb"]) for e in paths.values() if e["real"])
    print(f"objects={len(objects)}  sim imgs={n_sim}  real imgs={n_real}", flush=True)

    stats = {"objects": objects}

    print("Loading DINOv3 ViT-L/16...", flush=True)
    enc = DINOv3Encoder(model_name="dinov3_vitl16",
                        weights="weights/dinov3_vitl16_pretrain_lvd1689m.pth")
    enc = enc.to(args.device).eval()

    print("  [tactile / DINOv3]", flush=True)
    stats["tactile"] = branch_stats(enc, paths, "samples", args.device)

    print("  [rgb / DINOv3]", flush=True)
    stats["rgb"] = branch_stats(enc, paths, "rgb", args.device)

    del enc
    torch.cuda.empty_cache()

    torch.save(stats, args.out)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
