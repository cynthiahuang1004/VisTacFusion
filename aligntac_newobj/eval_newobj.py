"""Per-object evaluation of the 16-object K=25 runs on the new objects' held-out test frames (real_filtered_testP4),
plus the 12 session-1 objects (real_filtered_testA) as a check. Writes per-frame predictions for Fig. 6.

Metrics per object: contact MAE / IoU (best_depth.pt, as in eval_cross_session.py), rotation error and translation
error in mm (best_pose.pt). Translation is the mean coordinate error (trans_l1 / 2 * 41 mm, as in build_tables.py).
Object IDs come from build_datasets (sim objects first); a real-only map would shift the new objects' IDs.

usage (cwd = VisTacFusion): python aligntac_newobj/eval_newobj.py --device cuda:0 P4_k25_realonly P4_k25_semipm P4_k25_rf
writes aligntac_newobj/eval_newobj.json and outputs/robosoft_qual/qual_P4.npz
"""
import argparse, contextlib, copy, io, json, math, os, os.path as osp, re
import numpy as np, torch
from torch.utils.data import DataLoader
from vistacfusion.data.dataset import SimVisuoTactileDataset, build_datasets
from vistacfusion.engine.train import load_checkpoint
from vistacfusion.models.model import build_model
from vistacfusion.utils.config import merge_configs

OUT = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"; P = "ablation/pilot_robosoft"; G = "/media/hdd2/ihsuan/gs_blender"
MODEL = "ablation/encoder/tac_sitr_single.yaml"; TRAIN = f"{P}/train_pilot_e50.yaml"
ROOTS = {"new": f"{G}/real_filtered_testP4", "old": f"{G}/real_filtered_testA"}
HALF_MM = 41.0; IOU_THR = 0.05


@torch.no_grad()
def predict(run, root, ckpt, dev):
    base = re.sub(r"_seed\d+$", "", run)                      # seed runs share the data config
    cfg = merge_configs(MODEL, TRAIN, f"{P}/data_{base}.yaml")
    with contextlib.redirect_stdout(io.StringIO()):
        omap = build_datasets(cfg)[1]._obj_to_id   # training map: sim objects first, then real-only objects (IDs 12-15)
        c2 = copy.deepcopy(cfg); c2["real"]["root"] = root; c2["real"]["val_every"] = 1; c2["real"].pop("exclude_frames", None)
        ds = SimVisuoTactileDataset(c2, cfg.image_size, augment=False, split="all", data_section="real", shared_obj_map=omap)
        model = build_model(cfg).to(dev).eval(); load_checkpoint(osp.join(OUT, f"pilot150_{run}", ckpt), model, device=dev)
    id2obj = {v: k for k, v in omap.items()}
    R = dict(key=[], obj=[], depth=[], gt=[], img=[], se2=[], pose=[], normal=[], gt_normal=[])
    for b in DataLoader(ds, batch_size=64, num_workers=4):
        o = model(b["rgb"].to(dev), b["tactile"].to(dev), config="tactile", object_ids=b["object"].to(dev))
        R["depth"].append(o["depth"].float().cpu()[:, 0]); R["gt"].append(b["depth"].float()[:, 0]); R["img"].append(b["tactile"])
        R["se2"].append(o["se2"].float().cpu()); R["pose"].append(b["pose"].float()); R["obj"] += [id2obj[int(i)] for i in b["object"]]
        R["normal"].append(torch.nn.functional.normalize(o["normal"].float().cpu(), dim=1)); R["gt_normal"].append(torch.nn.functional.normalize(b["normal"].float(), dim=1))
    R["key"] = ["/".join(str(s[0]).split("/")[-4:-1]) + f":{s[1]}" for s in ds.samples]
    for k in ("depth", "gt", "img", "se2", "pose", "normal", "gt_normal"):
        R[k] = torch.cat(R[k])
    return R, cfg


def frame_metrics(R):
    pred, gt = R["depth"], R["gt"]; m = gt > 0
    cmae = ((pred - gt).abs() * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp(min=1)
    pm = pred > IOU_THR
    iou = (pm & m).flatten(1).sum(1).float() / (pm | m).flatten(1).sum(1).float().clamp(min=1)
    s, g = R["se2"], R["pose"]
    rot = torch.rad2deg(torch.acos((s[:, 0] * g[:, 0] + s[:, 1] * g[:, 1]).clamp(-1 + 1e-6, 1 - 1e-6)))
    trans = (s[:, 2:] - g[:, 2:]).abs().mean(1) * HALF_MM
    return dict(cmae=cmae.numpy(), iou=iou.numpy(), rot=rot.numpy(), trans=trans.numpy(),
                c_sum=((pred - gt).abs() * m).flatten(1).sum(1).numpy(), c_n=m.flatten(1).sum(1).numpy())


def summarize(fm, objs, sel):
    idx = [i for i, o in enumerate(objs) if sel(o)]
    if not idx:
        return None
    return dict(c_mae=float(fm["c_sum"][idx].sum() / fm["c_n"][idx].sum()), iou=float(fm["iou"][idx].mean()),
                rot=float(fm["rot"][idx].mean()), rot_med=float(np.median(fm["rot"][idx])),
                trans=float(fm["trans"][idx].mean()), n=len(idx))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json", default="aligntac_newobj/eval_newobj.json"); ap.add_argument("--no-old", action="store_true")
    a = ap.parse_args()
    res = json.load(open(a.json)) if osp.exists(a.json) else {}
    qual = {}
    for run in a.runs:
        for which, root in ROOTS.items():
            if which == "old" and a.no_old:
                continue
            Rd, cfg = predict(run, root, "best_depth.pt", a.device); Rp, _ = predict(run, root, "best_pose.pt", a.device)
            assert Rd["key"] == Rp["key"]
            fd, fp = frame_metrics(Rd), frame_metrics(Rp)
            fm = dict(fd, rot=fp["rot"], trans=fp["trans"])
            objs = Rd["obj"]
            out = {o: summarize(fm, objs, lambda x, o=o: x == o) for o in sorted(set(objs))}
            out["all"] = summarize(fm, objs, lambda x: True)
            res[f"{run}/{which}"] = out
            print(f"== {run} [{which}]", flush=True)
            for o, r in out.items():
                print(f"  {o:30s} cMAE {r['c_mae']:.3f}  IoU {r['iou']:.3f}  rot {r['rot']:6.2f} (med {r['rot_med']:5.2f})  trans {r['trans']:.2f} mm  n {r['n']}", flush=True)
            if which == "new":
                if "keys" not in qual:
                    mean = torch.tensor(cfg.norm.imagenet_mean).view(1, 3, 1, 1); std = torch.tensor(cfg.norm.imagenet_std).view(1, 3, 1, 1)
                    qual.update(keys=np.array(Rd["key"]), obj=np.array(objs), gt=Rd["gt"].numpy().astype(np.float16),
                                img=(Rd["img"] * std + mean).clamp(0, 255).permute(0, 2, 3, 1).numpy().astype(np.uint8))
                assert list(qual["keys"]) == Rd["key"]
                qual[f"pred_{run}"] = Rd["depth"].numpy().astype(np.float16); qual[f"cmae_{run}"] = fd["cmae"]
                m = Rd["gt"] > 0; ang = torch.rad2deg(torch.acos((Rd["normal"] * Rd["gt_normal"]).sum(1).clamp(-1 + 1e-6, 1 - 1e-6)))
                qual["gt_normal"] = Rd["gt_normal"].permute(0, 2, 3, 1).numpy().astype(np.float16)
                qual[f"normal_{run}"] = Rd["normal"].permute(0, 2, 3, 1).numpy().astype(np.float16)
                qual[f"nerr_{run}"] = ((ang * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp(min=1)).numpy()
        json.dump(res, open(a.json, "w"), indent=1)
    if qual:
        os.makedirs("outputs/robosoft_qual", exist_ok=True)
        path = "outputs/robosoft_qual/qual_P4.npz"
        if osp.exists(path):                                  # keep predictions of runs evaluated earlier
            old = dict(np.load(path))
            if list(old["keys"]) == list(qual["keys"]):
                qual = {**old, **qual}
        np.savez_compressed(path, **qual)


if __name__ == "__main__":
    main()
