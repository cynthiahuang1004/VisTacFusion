"""Batch evaluation for v1 ratio runs (r05, r25, r50) — same format as ratio_t3_mae_r100:
  eval_metrics.txt + sim_val_vis/ + eval_results.json + real_test_vis/ + real_train_vis/
"""
import json
import os
import shutil
import sys
import tempfile

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vistacfusion.data.dataset import build_datasets
from vistacfusion.engine.eval import evaluate
from vistacfusion.engine.inference import load_model, run_eval_sim, run_real_data_tree
from vistacfusion.utils.config import merge_configs

RUNS = ["ratio_t3_mae_r25", "ratio_t3_mae_r05", "ratio_t3_mae_r50"]
TEST_OBJECTS = ["pattern_01_2_lines_angle_2", "pattern_04_3_lines_angle_1", "pattern_32"]
REAL_ROOT = "/media/hdd2/ihsuan/gs_blender/real_data"

cfg_sim = merge_configs("ablation/encoder/tac_t3_rgb_mae.yaml", "configs/train.yaml",
                        "configs/data.yaml")
cfg_real = merge_configs("ablation/encoder/tac_t3_rgb_mae.yaml", "configs/train_bs32.yaml",
                         "ablation/ratio/data_ratio25.yaml")
device = torch.device("cuda:0")

# Real val/test loaders (shared across runs)
train_ds, real_val = build_datasets(cfg_real)
real_test = train_ds.real_test
val_loader = DataLoader(real_val, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(real_test, batch_size=32, shuffle=False, num_workers=4)

train_objects = [o for o in sorted(os.listdir(REAL_ROOT))
                 if os.path.isdir(os.path.join(REAL_ROOT, o)) and o not in TEST_OBJECTS]

for run in RUNS:
    train_dir = f"outputs/{run}"
    out_base = f"eval_results/{run}"
    os.makedirs(out_base, exist_ok=True)
    print(f"\n{'#'*70}\n# {run}\n{'#'*70}")

    depth_model = load_model(cfg_sim, f"{train_dir}/best_depth.pt", device)
    pose_model = load_model(cfg_sim, f"{train_dir}/best_pose.pt", device)
    depth_ep = torch.load(f"{train_dir}/best_depth.pt", map_location="cpu",
                          weights_only=False)["epoch"]
    pose_ep = torch.load(f"{train_dir}/best_pose.pt", map_location="cpu",
                         weights_only=False)["epoch"]

    # 1. Sim val eval + vis
    print(f"\n--- SIM VAL ({run}) ---")
    sim_out = os.path.join(out_base, "sim_val_vis")
    os.makedirs(sim_out, exist_ok=True)
    run_eval_sim(depth_model, cfg_sim, device, sim_out, pose_model=pose_model,
                 ckpt_info=f"depth=best_depth(ep{depth_ep}), pose=best_pose(ep{pose_ep})")

    # 2. Real quantitative eval (all 3 ckpts)
    print(f"\n--- REAL QUANT ({run}) ---")
    results = {}
    eval_model = load_model(cfg_sim, f"{train_dir}/latest.pt", device)
    for tag in ["best_depth", "best_pose", "latest"]:
        ckpt = torch.load(f"{train_dir}/{tag}.pt", map_location=device, weights_only=False)
        eval_model.load_state_dict(ckpt["model"])
        val_rep = evaluate(eval_model, val_loader, cfg_real, device)
        test_rep = evaluate(eval_model, test_loader, cfg_real, device)
        results[tag] = {"epoch": ckpt["epoch"], "real_val_7obj": val_rep,
                        "real_test_3obj": test_rep}
        v, t = val_rep["both"], test_rep["both"]
        print(f"{tag} (ep{ckpt['epoch']}): "
              f"val depth={v['depth_mse']:.4f} rot={v['pose_rot_deg']:.2f} | "
              f"test depth={t['depth_mse']:.4f} rot={t['pose_rot_deg']:.2f}")
    with open(os.path.join(out_base, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # 3. Real test vis (3 held-out objects, 20 samples each)
    print(f"\n--- REAL TEST VIS ({run}) ---")
    tmpdir = tempfile.mkdtemp(prefix="real_test_")
    for obj in TEST_OBJECTS:
        os.symlink(os.path.join(REAL_ROOT, obj), os.path.join(tmpdir, obj))
    real_test_out = os.path.join(out_base, "real_test_vis")
    if os.path.exists(real_test_out):
        shutil.rmtree(real_test_out)
    os.makedirs(real_test_out, exist_ok=True)
    run_real_data_tree(depth_model, tmpdir, cfg_sim, device, real_test_out,
                       max_samples_per_sensor=20, pose_model=pose_model, rgb_subdir="rgb")
    shutil.rmtree(tmpdir)

    # 4. Real train vis (7 objects, 10 samples each)
    print(f"\n--- REAL TRAIN VIS ({run}) ---")
    tmpdir2 = tempfile.mkdtemp(prefix="real_train_")
    for obj in train_objects:
        os.symlink(os.path.join(REAL_ROOT, obj), os.path.join(tmpdir2, obj))
    real_train_out = os.path.join(out_base, "real_train_vis")
    if os.path.exists(real_train_out):
        shutil.rmtree(real_train_out)
    os.makedirs(real_train_out, exist_ok=True)
    run_real_data_tree(depth_model, tmpdir2, cfg_sim, device, real_train_out,
                       max_samples_per_sensor=10, pose_model=pose_model, rgb_subdir="rgb")
    shutil.rmtree(tmpdir2)

    del depth_model, pose_model, eval_model
    torch.cuda.empty_cache()

print("\nAll runs done!")
