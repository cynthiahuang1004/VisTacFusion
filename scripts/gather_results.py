#!/usr/bin/env python3
"""Gather best-epoch results from all ratio runs and print tables."""
import json, os, math, glob

BASE = "/media/hdd2/ihsuan/VisTacFusion/outputs"

def load_best(history_path):
    """Load history.json, return best metrics per mode (best by rot_deg)."""
    with open(history_path) as f:
        hist = json.load(f)

    # Detect modes
    ep0_val = hist[0].get("val", {})
    modes = [k for k in ep0_val if isinstance(ep0_val[k], dict)]
    if not modes:
        modes = ["flat"]

    results = {}
    for mode in modes:
        best_rot_deg = 999
        best_rec = None
        for rec in hist:
            val = rec.get("val", {})
            v = val if mode == "flat" else val.get(mode, {})
            rd = v.get("pose_rot_deg", v.get("rot_deg", None))
            if rd is None:
                rl = v.get("pose_rot_l1", v.get("rot_l1", None))
                if rl is not None:
                    rd = rl * 180 / math.pi
            if rd is not None and rd < best_rot_deg:
                best_rot_deg = rd
                best_rec = (rec["epoch"], v)

        if best_rec:
            ep, v = best_rec
            results[mode] = {
                "epoch": ep,
                "depth_mse": v.get("depth_mse", None),
                "normal_mse": v.get("normal_mse", None),
                "rot_l1": v.get("pose_rot_l1", v.get("rot_l1", None)),
                "rot_deg": v.get("pose_rot_deg", v.get("rot_deg", None)),
                "trans_l1": v.get("pose_trans", v.get("trans_l1", None)),
            }
    return results

# Collect all runs
patterns = {
    "bl3s": "ratio_bl3s_sim*",
    "g3s": "ratio_g3s_sim*",
    "realonly": "ratio_realonly*",
    "bl3s_simonly": "bl3s_simonly*",
    "g3s_simonly": "g3s_simonly*",
}

all_runs = {}
for tag, pat in patterns.items():
    dirs = sorted(glob.glob(os.path.join(BASE, pat)))
    for d in dirs:
        hf = os.path.join(d, "history.json")
        if os.path.isfile(hf):
            name = os.path.basename(d)
            try:
                all_runs[name] = load_best(hf)
            except Exception as e:
                print(f"ERROR loading {name}: {e}")

# Sort and group
bl3s_runs = {k: v for k, v in sorted(all_runs.items()) if k.startswith("ratio_bl3s")}
g3s_runs = {k: v for k, v in sorted(all_runs.items()) if k.startswith("ratio_g3s")}
realonly_runs = {k: v for k, v in sorted(all_runs.items()) if "realonly" in k}
bl3s_simonly = {k: v for k, v in sorted(all_runs.items()) if k.startswith("bl3s_simonly")}
g3s_simonly = {k: v for k, v in sorted(all_runs.items()) if k.startswith("g3s_simonly")}

def fmt(val, decimals=4):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"

def print_table(title, runs_dict):
    print(f"\n{'='*120}")
    print(f"  {title}")
    print(f"{'='*120}")

    modes = ["both", "tactile", "rgb"]

    # Header
    header = f"{'Run':<35} {'Mode':<9} {'Ep':>4} {'Depth MSE':>11} {'Normal MSE':>11} {'Rot L1':>10} {'Rot°':>8} {'Trans L1':>10}"
    print(header)
    print("-" * 120)

    for run_name, run_data in runs_dict.items():
        first = True
        for mode in modes:
            if mode in run_data:
                m = run_data[mode]
                label = run_name if first else ""
                print(f"{label:<35} {mode:<9} {m['epoch']:>4} {fmt(m['depth_mse'], 6):>11} {fmt(m['normal_mse'], 6):>11} {fmt(m['rot_l1'], 6):>10} {fmt(m['rot_deg'], 3):>8} {fmt(m['trans_l1'], 6):>10}")
                first = False
        if "flat" in run_data:
            m = run_data["flat"]
            print(f"{run_name:<35} {'flat':<9} {m['epoch']:>4} {fmt(m['depth_mse'], 6):>11} {fmt(m['normal_mse'], 6):>11} {fmt(m['rot_l1'], 6):>10} {fmt(m['rot_deg'], 3):>8} {fmt(m['trans_l1'], 6):>10}")
        print()

import sys
out_path = "/media/hdd2/ihsuan/VisTacFusion/.tmp_table_output.txt"
with open(out_path, "w") as out_f:
    old_stdout = sys.stdout
    sys.stdout = out_f
    print_table("Blender 3-Session (bl3s) Ratio Ladder", {**bl3s_runs, **realonly_runs, **bl3s_simonly})
    print_table("GAN 3-Session (g3s) Ratio Ladder", {**g3s_runs, **realonly_runs, **g3s_simonly})
    sys.stdout = old_stdout
