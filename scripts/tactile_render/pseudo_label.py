"""Self-training the renderer: pseudo-label unlabeled real frames with a K-shot downstream model.

The K-shot downstream model (trained on K labeled frames per object + GAN-K data) predicts depth on
every UNLABELED real training frame; the (pseudo depth, real image) pairs then extend the renderer's
training set (train_tactile_gan.py --real-root <out-root>).  The real appearance is exact, only the
depth is approximate.  Frames are kept when the predicted contact agrees with the image's own ridge
mask (black-hat, no labels needed) and the indentation is plausible.  The K labeled frames keep their
true labels.  Only train-split frames (idx % 10 != 0) are written: validation frames never touch the
renderer, as in every other protocol of the paper.

usage: python scripts/tactile_render/pseudo_label.py --run B_k25_gank --k 25 \
           --out-root /media/hdd2/ihsuan/gs_blender/real_pseudo_k25 --device cuda:0
"""
import argparse
import contextlib
import io
import json
import math
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, osp.join(ROOT, "scripts")); sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from calibrate_new_real_poses import blackhat_mask  # noqa: E402
from train_tactile_gan import REAL_ROOT, RealPairs  # noqa: E402
from vistacfusion.data.dataset import SimVisuoTactileDataset, build_datasets  # noqa: E402
from vistacfusion.engine.train import load_checkpoint  # noqa: E402
from vistacfusion.models.model import build_model  # noqa: E402
from vistacfusion.utils.config import merge_configs  # noqa: E402

OUT = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"
MODEL = "ablation/encoder/tac_sitr_single.yaml"
TRAIN = "ablation/pilot_robosoft/train_pilot_e50.yaml"


def uncrop(pred_mm, crop, H=224, W=224):
    """Model output lives in the fixed center crop; paste it back into the full frame (zeros outside)."""
    side = int(math.floor(min(H, W) * crop))
    oy, ox = (H - side) // 2, (W - side) // 2
    full = np.zeros((H, W), np.float32)
    full[oy:oy + side, ox:ox + side] = cv2.resize(pred_mm, (side, side), interpolation=cv2.INTER_LINEAR)
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="downstream run used as labeler, e.g. B_k25_gank")
    ap.add_argument("--prefix", default="pilot150_")
    ap.add_argument("--k", type=int, required=True, help="K labeled frames per object (kept with true labels)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iou-thr", type=float, default=0.2, help="min IoU(pred contact, black-hat ridge mask)")
    ap.add_argument("--min-depth", type=float, default=0.2, help="mm")
    ap.add_argument("--max-depth", type=float, default=1.6, help="mm")
    ap.add_argument("--contact-thr", type=float, default=0.05, help="mm; pred below this is zeroed")
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many frames")
    args = ap.parse_args()
    os.chdir(ROOT)
    dev = args.device
    cfg = merge_configs(MODEL, TRAIN, f"ablation/pilot_robosoft/data_{args.run}.yaml")
    with contextlib.redirect_stdout(io.StringIO()):
        _, val_ds = build_datasets(cfg)
        model = build_model(cfg).to(dev).eval()
        load_checkpoint(osp.join(OUT, args.prefix + args.run, "best_depth.pt"), model, device=dev)
        cfg.real.train_samples_per_session = 10 ** 6            # all train-split frames, not only the K
        ds = SimVisuoTactileDataset(cfg, cfg.image_size, augment=False, split="train", data_section="real",
                                    shared_obj_map=val_ds._obj_to_id)
    crop = float(cfg.real.get("fixed_crop", cfg.get("fixed_crop", 0.816)))
    rp = RealPairs(split="train", per_object=args.k)
    labeled = {tac for _, tac in rp.items}
    print(f"labeler={args.prefix}{args.run}  frames={len(ds)}  labeled(K={args.k})={len(labeled)}  crop={crop}", flush=True)

    n = len(ds) if not args.limit else min(args.limit, len(ds))
    idxs = list(range(n))
    dl = DataLoader(torch.utils.data.Subset(ds, idxs), batch_size=32, shuffle=False, num_workers=6)
    stats = dict(kept=0, dropped_iou=0, dropped_depth=0, labeled=0, iou=[])
    manifest = []
    i0 = 0
    # The downstream model sees a `crop` center window; the renderer needs full-frame labels.  Predict on
    # five windows (center + four corners) and average them into the full frame.
    side = int(math.floor(224 * crop)); mx_off = 224 - side
    offsets = [(mx_off // 2, mx_off // 2), (0, 0), (0, mx_off), (mx_off, 0), (mx_off, mx_off)]
    with torch.no_grad():
        for b in dl:
            B = b["tactile"].shape[0]
            wins, ims = [], []
            for j in range(B):
                unit, sidx = ds.samples[idxs[i0 + j]]
                im = np.array(Image.open(osp.join(unit, "samples", f"{sidx:04d}.png")).convert("RGB"), np.float32)
                ims.append(im)
                for oy, ox in offsets:
                    wins.append(ds.img_xform(cv2.resize(im[oy:oy + side, ox:ox + side], (224, 224), interpolation=cv2.INTER_LINEAR)))
            tac = torch.stack(wins).to(dev)
            rgb = b["rgb"].to(dev).repeat_interleave(len(offsets), 0)
            obj = b["object"].to(dev).repeat_interleave(len(offsets), 0) if torch.is_tensor(b.get("object")) else None
            out = model(rgb, tac, config="tactile", object_ids=obj)["depth"][:, 0].float().clamp(min=0).cpu().numpy()
            pred = np.zeros((B, 224, 224), np.float32)
            for j in range(B):
                acc, cnt = np.zeros((224, 224), np.float32), np.zeros((224, 224), np.float32)
                for w, (oy, ox) in enumerate(offsets):
                    acc[oy:oy + side, ox:ox + side] += cv2.resize(out[j * len(offsets) + w], (side, side), interpolation=cv2.INTER_LINEAR)
                    cnt[oy:oy + side, ox:ox + side] += 1
                pred[j] = acc / np.maximum(cnt, 1)
            pred[pred < args.contact_thr] = 0
            for j in range(len(pred)):
                unit, sidx = ds.samples[idxs[i0 + j]]
                obj, sess = unit.split("/")[-3], unit.split("/")[-2]
                tac = osp.join(unit, "samples", f"{sidx:04d}.png")
                out_unit = osp.join(args.out_root, obj, sess, "sensor_0000")
                os.makedirs(osp.join(out_unit, "samples"), exist_ok=True); os.makedirs(osp.join(out_unit, "raw_data"), exist_ok=True)
                dst_img = osp.join(out_unit, "samples", f"{sidx:04d}.png")
                dst_dep = osp.join(out_unit, "raw_data", f"{sidx:04d}_gt.npy")
                if tac in labeled:                                  # true label for the K budget frames
                    src = osp.join(unit, "raw_data", f"{sidx:04d}_gt.npy")
                    if not osp.exists(src):
                        src = osp.join(unit, "raw_data", f"{sidx:04d}.npy")
                    np.save(dst_dep, np.load(src).astype(np.float32))
                    if not osp.lexists(dst_img):
                        os.symlink(tac, dst_img)
                    stats["labeled"] += 1; manifest.append((tac, "labeled", 1.0)); continue
                full = pred[j]                                       # mm, full frame (stitched)
                mx = float(full.max())
                if not (args.min_depth <= mx <= args.max_depth):
                    stats["dropped_depth"] += 1; manifest.append((tac, "drop_depth", mx)); continue
                bh = blackhat_mask(ims[j].astype(np.uint8))
                pm = full > args.contact_thr
                iou = float((pm & bh).sum() / max(1, (pm | bh).sum()))
                stats["iou"].append(iou)
                if iou < args.iou_thr:
                    stats["dropped_iou"] += 1; manifest.append((tac, "drop_iou", iou)); continue
                np.save(dst_dep, (full / 1000.0).astype(np.float32))  # metres, like the real GT
                if not osp.lexists(dst_img):
                    os.symlink(tac, dst_img)
                stats["kept"] += 1; manifest.append((tac, "pseudo", iou))
            i0 += len(pred)
            if (i0 // 32) % 20 == 0:
                print(f"  {i0}/{n}", flush=True)
    ious = np.array(stats.pop("iou"))
    stats["iou_mean"] = float(ious.mean()) if len(ious) else None
    stats["iou_median"] = float(np.median(ious)) if len(ious) else None
    print(json.dumps(stats, indent=1), flush=True)
    json.dump({"stats": stats, "manifest": manifest, "args": vars(args)}, open(osp.join(args.out_root, "pseudo_manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
