"""Predictions for the qualitative figure (fig_qual): real-only vs SnapTac at K=25 on held-out frames of both sensors.
Writes outputs/robosoft_qual/qual_{A,B}.npz (frame, reference depth, predicted depth per run, per-frame contact MAE).
usage: python paper_RoboSoft/scripts/dump_qual_preds.py [--device cuda:0]
"""
import argparse, contextlib, copy, io, os.path as osp
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from vistacfusion.data.dataset import SimVisuoTactileDataset, build_datasets
from vistacfusion.engine.train import load_checkpoint
from vistacfusion.models.model import build_model
from vistacfusion.utils.config import merge_configs

OUT = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"; P = "ablation/pilot_robosoft"
MODEL = "ablation/encoder/tac_sitr_single.yaml"; TRAIN = f"{P}/train_pilot_e50.yaml"
SETS = {"A": dict(runs=[("pilot150_", "B_k25_realonly"), ("pilot150_", "B_k25_al2")], root=None, step=1),
        "B": dict(runs=[("pilot_", "S_k25_realonly_m"), ("pilot_", "S_k25_mapl_m")], root="/media/hdd2/ihsuan/gs_blender/real_gelslim40_test_mesh", step=4)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args(); dev = a.device
    for name, S in SETS.items():
        out = {}
        for prefix, run in S["runs"]:
            cfg = merge_configs(MODEL, TRAIN, f"{P}/data_{run}.yaml")
            with contextlib.redirect_stdout(io.StringIO()):
                if S["root"] is None:
                    _, ds = build_datasets(cfg)
                else:
                    old = SimVisuoTactileDataset(cfg, cfg.image_size, augment=False, split="val", data_section="real")
                    c2 = copy.deepcopy(cfg); c2["real"]["root"] = S["root"]; c2["real"]["val_every"] = 1; c2["fixed_crop"] = 1.0
                    ds = SimVisuoTactileDataset(c2, cfg.image_size, augment=False, split="all", data_section="real", shared_obj_map=old._obj_to_id)
                model = build_model(cfg).to(dev).eval(); load_checkpoint(osp.join(OUT, prefix + run, "best_depth.pt"), model, device=dev)
            idx = list(range(0, len(ds), S["step"]))
            keys = ["/" + "/".join(str(ds.samples[i][0]).split("/")[-3:]) + f"/:{ds.samples[i][1]}" for i in idx]
            pred, gt, img = [], [], []
            for b in DataLoader(Subset(ds, idx), batch_size=32, num_workers=4):
                o = model(b["rgb"].to(dev), b["tactile"].to(dev), config="tactile", object_ids=b["object"].to(dev))
                pred.append(o["depth"].float().cpu()[:, 0]); gt.append(b["depth"].float()[:, 0]); img.append(b["tactile"])
            pred, gt, img = torch.cat(pred), torch.cat(gt), torch.cat(img)
            if "keys" not in out:
                mean = torch.tensor(cfg.norm.imagenet_mean).view(1, 3, 1, 1); std = torch.tensor(cfg.norm.imagenet_std).view(1, 3, 1, 1)
                out.update(keys=np.array(keys), gt=gt.numpy().astype(np.float16), img=(img * std + mean).clamp(0, 255).permute(0, 2, 3, 1).numpy().astype(np.uint8))
            assert list(out["keys"]) == keys
            m = gt > 0
            out[f"pred_{run}"] = pred.numpy().astype(np.float16)
            out[f"cmae_{run}"] = ((pred - gt).abs() * m).flatten(1).sum(1).div(m.flatten(1).sum(1).clamp(min=1)).numpy()
            print(name, run, len(keys), "contact MAE", float(out[f"cmae_{run}"].mean()), flush=True)
        np.savez_compressed(f"outputs/robosoft_qual/qual_{name}.npz", **out)


if __name__ == "__main__":
    main()
