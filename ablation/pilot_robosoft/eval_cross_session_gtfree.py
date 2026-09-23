"""GT-free cross-session evaluation on ALL usable new-session frames.

The dense GT of the new session is unreliable (pose calibration), so this script scores what can be
measured without it:
  * contact IoU_img : IoU between the predicted contact region (depth > 0.05 mm) and the ridge
                      mask visible in the tactile image (black-hat), i.e. "does the model put the
                      contact where the image shows it";
  * rotation error  : vs the robot yaw + per-object theta offset (v2 joint calibration; rotation is
                      well constrained even for line patterns, unlike x/y);
  * the same IoU_img on the old-session real val split for reference.

Frames: press >= 0.5 mm (robot), pose/image lag <= 0.5 s, every 3rd frame within a press,
visible ridges (mask area >= 1%), objects listed in --exclude dropped.

usage: python ablation/pilot_robosoft/eval_cross_session_gtfree.py run:prefix ... [--device cuda:0]
"""
import argparse
import contextlib
import io
import json
import math
import os
import os.path as osp
import re
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "scripts")
from build_real_filtered_new import OBJ_MAP, T, robot_to_gt  # noqa: E402
from calibrate_new_real_poses import blackhat_mask  # noqa: E402
from vistacfusion.data.dataset import SimVisuoTactileDataset  # noqa: E402
from vistacfusion.data.transforms import ToTensorResize, fixed_center_crop  # noqa: E402
from vistacfusion.engine.train import load_checkpoint  # noqa: E402
from vistacfusion.models.model import build_model  # noqa: E402
from vistacfusion.utils.config import merge_configs  # noqa: E402

OUT = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"
MODEL = "ablation/encoder/tac_sitr_single.yaml"
TRAIN = "ablation/pilot_robosoft/train_pilot_e50.yaml"
V2_LOG = f"{T}/validation/calibration_v2.log"


def v2_dtheta():
    out = {}
    for line in open(V2_LOG):
        m = re.match(r"(\S+)\s+IoU .* dth=\s*([-+\d.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def new_session_frames(exclude, stride=3, min_press=0.5):
    dth = v2_dtheta(); frames = []
    for real_name in sorted(OBJ_MAP):
        sim_name, th_off, bx, by = OBJ_MAP[real_name]
        if sim_name in exclude or not osp.isdir(f"{T}/{real_name}"):
            continue
        for run in sorted((d for d in os.listdir(f"{T}/{real_name}") if d.isdigit()), key=int):
            p = f"{T}/{real_name}/{run}/tactile_images/poses.json"
            if not osp.exists(p):
                continue
            m = json.load(open(p)); cz = m["contact_z_m"]; k = 0; prev = False
            for f in m["frames"]:
                if abs(f["pose_ros_timestamp_sec"] - f["image_ros_timestamp_sec"]) > 0.5:
                    continue
                ee = f["robot_pose_xyz_xyzw"]
                th, x, y, press = robot_to_gt(ee[3:], ee[:2], th_off, bx, by, cz, ee[2])
                if press < min_press:
                    prev = False; continue
                k = k + 1 if prev else 0; prev = True
                if k % stride:
                    continue
                frames.append(dict(obj=sim_name, run=run, frame=f["frame_id"],
                                   path=f"{T}/{real_name}/{run}/tactile_images/{f['filename']}",
                                   theta=th + math.radians(dth.get(real_name, 0.0)), press=press))
    return frames


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--exclude", nargs="*", default=["pattern_04_3_lines_angle_2"])
    ap.add_argument("--json", default="ablation/pilot_robosoft/cross_session_gtfree.json")
    args = ap.parse_args()
    dev = args.device
    frames = new_session_frames(set(args.exclude))
    # image ridge masks (cropped like the model input) computed once
    masks, tacs = [], []
    for fr in frames:
        im = np.asarray(Image.open(fr["path"]).convert("RGB"))
        im_c = fixed_center_crop(im, crop=0.816)
        masks.append(blackhat_mask(np.asarray(Image.fromarray(im_c).resize((224, 224)))))
        tacs.append(im)
    keep = [i for i, m in enumerate(masks) if m.mean() >= 0.01]
    frames = [frames[i] for i in keep]; masks = np.stack([masks[i] for i in keep]); tacs = [tacs[i] for i in keep]
    print(f"{len(frames)} new-session frames with visible contact ({len(set(f['obj'] for f in frames))} objects)", flush=True)
    results = json.load(open(args.json)) if osp.exists(args.json) else {}
    print(f"{'run':24s} {'prefix':10s} {'IoU_img new':>12s} {'IoU_img old':>12s} {'rot new':>8s} {'rot old':>8s}", flush=True)
    for spec in args.runs:
        run, prefix = spec.split(":")
        ckpt = osp.join(OUT, prefix + run, "best_depth.pt")
        if not osp.exists(ckpt):
            print(run, "no ckpt"); continue
        base = re.sub(r'_seed\d+$', '', run)
        cfg = merge_configs(MODEL, TRAIN, f"ablation/pilot_robosoft/data_{base}.yaml")
        with contextlib.redirect_stdout(io.StringIO()):
            old = SimVisuoTactileDataset(cfg, cfg.image_size, augment=False, split="val", data_section="real")
            model = build_model(cfg).to(dev).eval(); load_checkpoint(ckpt, model, device=dev)
        omap = old._obj_to_id; xf = ToTensorResize((224, 224), cfg.norm.imagenet_mean, cfg.norm.imagenet_std)
        # --- new session ---
        ious, rots = [], []
        for i in range(0, len(frames), 32):
            chunk = frames[i:i + 32]
            x = torch.stack([xf(fixed_center_crop(tacs[i + j], crop=0.816)) for j in range(len(chunk))]).to(dev)
            oid = torch.tensor([omap.get(c["obj"], 0) for c in chunk], device=dev)
            out = model(torch.zeros_like(x), x, config="tactile", object_ids=oid)
            pm = (out["depth"][:, 0] > 0.05).cpu().numpy(); gm = masks[i:i + 32]
            ious += list((pm & gm).sum((1, 2)) / np.maximum(1, (pm | gm).sum((1, 2))))
            se2 = out["se2"].cpu().numpy()
            for c, s in zip(chunk, se2):
                d = math.cos(c["theta"]) * s[0] + math.sin(c["theta"]) * s[1]
                rots.append(math.degrees(math.acos(max(-1, min(1, d)))))
        # --- old session val (reference) ---
        from torch.utils.data import DataLoader
        oi, orr = [], []
        for b in DataLoader(old, batch_size=32, num_workers=4):
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b["rgb"], b["tactile"], config="tactile", object_ids=b["object"])
            t = b["tactile"].cpu().numpy(); mean = np.array(cfg.norm.imagenet_mean)[:, None, None]; std = np.array(cfg.norm.imagenet_std)[:, None, None]
            imgs = np.clip(t * std + mean, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
            gm = np.stack([blackhat_mask(im) for im in imgs]); pm = (out["depth"][:, 0] > 0.05).cpu().numpy()
            ok = gm.mean((1, 2)) >= 0.01
            oi += list(((pm & gm).sum((1, 2)) / np.maximum(1, (pm | gm).sum((1, 2))))[ok])
            se2 = out["se2"]; gp = b["pose"]
            orr += list((torch.acos((se2[:, 0] * gp[:, 0] + se2[:, 1] * gp[:, 1]).clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / math.pi).cpu().numpy())
        r = dict(iou_img_new=float(np.mean(ious)), iou_img_old=float(np.mean(oi)), rot_new=float(np.mean(rots)),
                 rot_new_median=float(np.median(rots)), rot_old=float(np.mean(orr)), n_new=len(ious), n_old=len(oi))
        per_obj = {}
        for c, v, rr in zip(frames, ious, rots):
            a = per_obj.setdefault(c["obj"], [0.0, 0.0, 0]); a[0] += v; a[1] += rr; a[2] += 1
        r["per_object"] = {o: dict(iou_img=a[0] / a[2], rot=a[1] / a[2], n=a[2]) for o, a in per_obj.items()}
        results[f"{prefix}{run}"] = r
        print(f"{run:24s} {prefix:10s} {r['iou_img_new']:12.3f} {r['iou_img_old']:12.3f} {r['rot_new']:8.2f} {r['rot_old']:8.2f}", flush=True)
        json.dump(results, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
