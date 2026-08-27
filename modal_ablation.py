"""Modal deployment for VisTacFusion encoder ablation (6 tactile × 4 RGB).

Usage (run from the VisTacFusion repo root on your server):

  # 0. Authenticate (one time)
  modal token set

  # 1. Upload data + weights to Modal volume (~60 GB, takes ~30-60 min)
  python modal_ablation.py upload

  # 2. Run a single combo
  modal run modal_ablation.py::train --tac t3 --rgb clip

  # 3. Run all 21 remaining combos (up to 10 GPUs in parallel)
  modal run modal_ablation.py::train_remaining

  # 4. Download results back to local outputs/
  modal run modal_ablation.py::download_results
"""
from __future__ import annotations

import itertools
import os
import sys

import modal

# ============================================================
# Config
# ============================================================

VOLUME_NAME = "vistacfusion-data"
RESULTS_VOLUME = "vistacfusion-results"
GPU_TYPE = "A100"          # A100-40GB: ~$3/hr, A10G: ~$1.10/hr
TIMEOUT_HOURS = 24
REPO_URL = "https://github.com/cynthiahuang1004/VisTacFusion.git"
REPO_BRANCH = "VisTacFusion-v2"

TACTILE_KEYS = ["t3", "dinov3", "sparshv2", "sparshmae", "sitr", "dav2"]
RGB_KEYS = ["mae", "dinov3", "clip", "siglip"]

DONE_LOCALLY = {
    ("t3", "mae"),
    ("t3", "clip"),
    ("t3", "dinov3"),
    ("t3", "siglip"),
    ("dinov3", "mae"),
    ("dinov3", "dinov3"),
    ("dinov3", "clip"),
    ("dinov3", "siglip"),
    ("sitr", "mae"),
    ("sitr", "clip"),
    ("dav2", "clip"),
    ("dav2", "dinov3"),
}

app = modal.App("vistacfusion-ablation")
data_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers", "safetensors", "tensorboard",
        "opencv-python-headless", "matplotlib", "pyyaml",
        "tqdm", "timm", "scipy", "Pillow", "trimesh",
    )
)

# Add repo code into the image (baked in, no git clone needed)
image = image.add_local_dir(
    "/media/hdd2/ihsuan/VisTacFusion",
    remote_path="/workspace",
    ignore=[
        "outputs/", ".git/", "weights/", "pretrained_encoders/",
        "tmp_*", ".tmp_*", ".claude_tmp*", ".matplotlib*", ".mpl_tmp/",
        "eval_results/", "*.pyc", "__pycache__/",
    ],
)

# ============================================================
# Helpers
# ============================================================

def model_config(tac: str, rgb: str) -> str:
    return f"ablation/encoder/tac_{tac}_rgb_{rgb}.yaml"

def output_name(tac: str, rgb: str) -> str:
    return f"ablation_g3s_sim315_{tac}_{rgb}"


# ============================================================
# Upload (run locally on the server, NOT on Modal)
# ============================================================

def do_upload():
    """Upload sim/real data + encoder weights to Modal volume."""
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    SIM_ROOT = "/media/hdd2/ihsuan/gs_blender/renders_v3"
    REAL_ROOT = "/media/hdd2/ihsuan/gs_blender/real_filtered"
    MESH_DIR = "/media/hdd2/ihsuan/gs_blender/meshes"
    OBJECTS = [
        "pattern_01_2_lines_angle_1_2", "pattern_01_2_lines_angle_2",
        "pattern_01_2_lines_angle_3", "pattern_04_3_lines_angle_1",
        "pattern_04_3_lines_angle_2", "pattern_06_5_lines_angle_1",
        "pattern_31_rod", "pattern_32", "pattern_33",
        "pattern_35", "pattern_36", "pattern_37",
    ]
    NEEDED_SUBDIRS = {"samples_g", "rgb", "raw_data", "norms", "calibration"}

    with vol.batch_upload(force=True) as batch:
        # --- Sim data (selective subdirs only) ---
        print("Uploading sim data (12 objects × needed subdirs)...")
        for obj in OBJECTS:
            obj_path = os.path.join(SIM_ROOT, obj)
            if not os.path.isdir(obj_path):
                print(f"  SKIP {obj}")
                continue
            for sess in sorted(os.listdir(obj_path)):
                sess_path = os.path.join(obj_path, sess)
                if not os.path.isdir(sess_path):
                    continue
                sj = os.path.join(sess_path, "session.json")
                if os.path.exists(sj):
                    batch.put_file(sj, f"/sim/{obj}/{sess}/session.json")
                for sensor in sorted(os.listdir(sess_path)):
                    if not sensor.startswith("sensor_"):
                        continue
                    sensor_path = os.path.join(sess_path, sensor)
                    for sub in NEEDED_SUBDIRS:
                        sub_path = os.path.join(sensor_path, sub)
                        if os.path.isdir(sub_path):
                            batch.put_directory(sub_path,
                                                f"/sim/{obj}/{sess}/{sensor}/{sub}")
            print(f"  {obj}")

        # --- Real data (entire tree, ~2.2 GB) ---
        print("Uploading real data...")
        batch.put_directory(REAL_ROOT, "/real")

        # --- Meshes ---
        print("Uploading meshes...")
        batch.put_directory(MESH_DIR, "/meshes")

        # --- Encoder weights ---
        print("Uploading encoder weights...")
        WEIGHTS = [
            ("/media/hdd2/ihsuan/VisTacFusion/pretrained_encoders/t3_large/encoder_mini.pth",
             "/weights/t3_large/encoder_mini.pth"),
            ("/media/hdd2/ihsuan/VisTacFusion/pretrained_encoders/t3_large/trunk.pth",
             "/weights/t3_large/trunk.pth"),
            ("/media/hdd2/ihsuan/VisTacFusion/weights/dinov3_vitl16_pretrain_lvd1689m.pth",
             "/weights/dinov3_vitl16.pth"),
            ("/media/hdd2/ihsuan/VisTacFusion/pretrained_encoders/mae_vitl16.pth",
             "/weights/mae_vitl16.pth"),
            ("/media/hdd2/ihsuan/VisTacFusion/pretrained_encoders/dav2_vitl14.pth",
             "/weights/dav2_vitl14.pth"),
            ("/media/hdd2/ihsuan/sparsh/weights/sparsh/dinov2_vitbase.ckpt",
             "/weights/sparsh_dinov2_base.ckpt"),
            ("/media/hdd2/ihsuan/sparsh/weights/sparsh/mae_vitbase.ckpt",
             "/weights/sparsh_mae_base.ckpt"),
            ("/media/hdd2/ihsuan/gsrl/datasets/checkpoints/SITR_B18.pth",
             "/weights/SITR_B18.pth"),
        ]
        for local, remote in WEIGHTS:
            if os.path.exists(local):
                mb = os.path.getsize(local) / 1e6
                print(f"  {os.path.basename(local)} ({mb:.0f} MB)")
                batch.put_file(local, remote)
            else:
                print(f"  MISSING: {local}")

        # --- Rotation windows ---
        rw = "/media/hdd2/ihsuan/VisTacFusion/ablation/simqty_filtered/real_rotation_windows.json"
        if os.path.exists(rw):
            batch.put_file(rw, "/configs/real_rotation_windows.json")

    print("\nUpload complete.")


# ============================================================
# Training (runs on Modal GPU)
# ============================================================

# Checkpoint path remapping: local paths -> Modal volume paths
CKPT_REMAP = {
    "pretrained_encoders/t3_large": "/data/weights/t3_large",
    "weights/dinov3_vitl16_pretrain_lvd1689m.pth": "/data/weights/dinov3_vitl16.pth",
    "pretrained_encoders/mae_vitl16.pth": "/data/weights/mae_vitl16.pth",
    "pretrained_encoders/dav2_vitl14.pth": "/data/weights/dav2_vitl14.pth",
    "pretrained_encoders/sparsh_dinov2_base.ckpt": "/data/weights/sparsh_dinov2_base.ckpt",
    "pretrained_encoders/sparsh_dinov2_base.safetensors": "/data/weights/sparsh_dinov2_base.safetensors",
    "pretrained_encoders/sparsh_mae_base.safetensors": "/data/weights/sparsh_mae_base.safetensors",
    "/media/hdd2/ihsuan/sparsh/weights/sparsh/mae_vitbase.ckpt": "/data/weights/sparsh_mae_base.safetensors",
    "/media/hdd2/ihsuan/gsrl/datasets/checkpoints/SITR_B18.pth": "/data/weights/SITR_B18.pth",
}

SITR_CAL_DIR_MODAL = "/data/sim/pattern_01_2_lines_angle_1_2/session_000/sensor_0000/calibration"


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_vol, "/results": results_vol},
    timeout=3600 * TIMEOUT_HOURS,
    memory=32768,
)
def train_one(tac: str, rgb: str, transfilt: bool = False):
    """Train one encoder combo on a Modal GPU."""
    import shutil, subprocess, yaml

    run_name = output_name(tac, rgb)
    if transfilt:
        run_name = run_name.replace("ablation_g3s_sim315_", "ablation_g3s_sim315_tf_")
    print(f"\n{'='*60}")
    print(f"  {run_name}  (tac={tac}, rgb={rgb})")
    print(f"{'='*60}\n")

    os.chdir("/workspace")

    # --- Build data config with Modal paths ---
    data_cfg = {
        "image_size": 224,
        "dataset": "sim+real",
        "synthetic": {"num_samples": 256, "num_objects": 8},
        "sim": {
            "tactile_subdir": "samples_g",
            "include_objects": [
                "pattern_01_2_lines_angle_1_2", "pattern_01_2_lines_angle_2",
                "pattern_01_2_lines_angle_3", "pattern_04_3_lines_angle_1",
                "pattern_04_3_lines_angle_2", "pattern_06_5_lines_angle_1",
                "pattern_31_rod", "pattern_32", "pattern_33",
                "pattern_35", "pattern_36", "pattern_37",
            ],
            "root": "/data/sim",
            "mesh_dir": "/data/meshes",
            "rgb_subdir": "rgb",
            "use_gt_depth": True,
            "use_rendered_normals": True,
            "gel_view_m": 0.017502,
            "rot_augment": True,
            "rot_augment_max_deg": 15.0,
            "val_every": 20,
            "align_real_rotation": True,
            "rotation_windows": "/data/configs/real_rotation_windows.json",
            **({"translation_bounds": "/data/configs/real_translation_bounds.json",
                "translation_margin": 1.5,
                "train_samples_per_session": 459} if transfilt else
               {"train_samples_per_session": 315}),
        },
        "real": {
            "root": "/data/real",
            "mesh_dir": "/data/meshes",
            "rgb_subdir": "rgb",
            "use_rendered_normals": False,
            "val_every": 10,
            "augment": False,
            "oversample": 1,
        },
        "loader": {
            "num_workers": 4,
            "pin_memory": True,
            "prefetch_factor": 4,
            "persistent_workers": True,
        },
        "norm": {
            "imagenet_mean": [123.675, 116.28, 103.53],
            "imagenet_std": [58.395, 57.12, 57.375],
        },
    }
    data_path = "/tmp/data_modal.yaml"
    with open(data_path, "w") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)

    # --- Patch model config: remap checkpoint paths ---
    src_model_cfg = model_config(tac, rgb)
    with open(src_model_cfg) as f:
        mcfg = yaml.safe_load(f)

    for section in ["encoder", "rgb_encoder"]:
        enc = mcfg.get(section)
        if enc is None:
            continue
        ckpt = enc.get("checkpoint", "")
        if ckpt in CKPT_REMAP:
            enc["checkpoint"] = CKPT_REMAP[ckpt]
        cal = enc.get("calibration_dir", "")
        if cal:
            enc["calibration_dir"] = SITR_CAL_DIR_MODAL

    model_path = "/tmp/model_modal.yaml"
    with open(model_path, "w") as f:
        yaml.dump(mcfg, f, default_flow_style=False, sort_keys=False)

    output_dir = f"/workspace/outputs/{run_name}"

    # --- Train ---
    cmd = [
        "python", "-u", "-m", "vistacfusion.engine.train",
        "--model", model_path,
        "--train", "configs/train_bs32.yaml",
        "--data", data_path,
        "--output-dir", output_dir,
    ]
    print(f"CMD: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd)

    # --- Save results to results volume ---
    results_dir = f"/results/{run_name}"
    if os.path.isdir(output_dir):
        shutil.copytree(output_dir, results_dir, dirs_exist_ok=True)
        results_vol.commit()
        print(f"\nResults saved to {RESULTS_VOLUME}:/{run_name}")

    return {"run_name": run_name, "returncode": proc.returncode}


# ============================================================
# Ratio ladder (T3+MAE, transfilt + sim rgb_zoom 1.15 + fixed_crop 0.816)
# ============================================================

# name -> (train_samples_per_session, sim_oversample); sim count = name*36, real = 4197
RATIO_POINTS = {
    "sim34": (85, None), "sim68": (163, None), "sim120": (293, None), "sim148": (363, None),
    "sim190": (459, None), "sim213": (520, None), "sim229": (556, None), "sim250": (606, None),
    "sim279": (677, None), "sim315": (None, None), "sim348": (None, 1.11),
    "sim380": (None, 1.214), "sim570": (None, 1.82), "sim760": (None, 2.427),
}

def ratio_run_name(name: str) -> str:
    return f"ratio_g3s_{name}_transfilt_zoom115_crop816"


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_vol, "/results": results_vol},
    timeout=3600 * TIMEOUT_HOURS,
    memory=32768,
)
def train_ratio_one(name: str, tac: str = "t3", rgb: str = "mae"):
    """One ratio-ladder point (crop 0.816 world). Mirrors
    ablation/simqty_gtac/data_ratio_g3s_<name>_transfilt_zoom115_crop816.yaml with Modal paths."""
    import shutil, subprocess, yaml

    spp, oversample = RATIO_POINTS[name]
    run_name = ratio_run_name(name)
    print(f"\n{'='*60}\n  {run_name}  (spp={spp}, oversample={oversample})\n{'='*60}\n")
    os.chdir("/workspace")

    sim_cfg = {
        "tactile_subdir": "samples_g",
        "include_objects": [
            "pattern_01_2_lines_angle_1_2", "pattern_01_2_lines_angle_2",
            "pattern_01_2_lines_angle_3", "pattern_04_3_lines_angle_1",
            "pattern_04_3_lines_angle_2", "pattern_06_5_lines_angle_1",
            "pattern_31_rod", "pattern_32", "pattern_33",
            "pattern_35", "pattern_36", "pattern_37",
        ],
        "root": "/data/sim",
        "mesh_dir": "/data/meshes",
        "rgb_subdir": "rgb",
        "rgb_zoom": 1.15,
        "use_gt_depth": True,
        "use_rendered_normals": True,
        "gel_view_m": 0.017502,
        "rot_augment": True,
        "rot_augment_max_deg": 15.0,
        "val_every": 20,
        "align_real_rotation": True,
        "rotation_windows": "/data/configs/real_rotation_windows.json",
        "translation_bounds": "/data/configs/real_translation_bounds.json",
        "translation_margin": 1.5,
    }
    if spp is not None:
        sim_cfg["train_samples_per_session"] = spp
    if oversample is not None:
        sim_cfg["sim_oversample"] = oversample

    data_cfg = {
        "image_size": 224,
        "fixed_crop": 0.816,
        "dataset": "sim+real",
        "synthetic": {"num_samples": 256, "num_objects": 8},
        "sim": sim_cfg,
        "real": {
            "root": "/data/real", "mesh_dir": "/data/meshes", "rgb_subdir": "rgb",
            "use_rendered_normals": False, "val_every": 10, "augment": False, "oversample": 1,
        },
        "loader": {"num_workers": 4, "pin_memory": True, "prefetch_factor": 4,
                   "persistent_workers": True},
        "norm": {"imagenet_mean": [123.675, 116.28, 103.53],
                 "imagenet_std": [58.395, 57.12, 57.375]},
    }
    data_path = "/tmp/data_modal.yaml"
    with open(data_path, "w") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)

    with open(model_config(tac, rgb)) as f:
        mcfg = yaml.safe_load(f)
    for section in ["encoder", "rgb_encoder"]:
        enc = mcfg.get(section)
        if enc is None:
            continue
        if enc.get("checkpoint", "") in CKPT_REMAP:
            enc["checkpoint"] = CKPT_REMAP[enc["checkpoint"]]
        if enc.get("calibration_dir", ""):
            enc["calibration_dir"] = SITR_CAL_DIR_MODAL
            enc["fixed_crop"] = 0.816
    model_path = "/tmp/model_modal.yaml"
    with open(model_path, "w") as f:
        yaml.dump(mcfg, f, default_flow_style=False, sort_keys=False)

    output_dir = f"/workspace/outputs/{run_name}"
    cmd = ["python", "-u", "-m", "vistacfusion.engine.train",
           "--model", model_path, "--train", "configs/train_bs32.yaml",
           "--data", data_path, "--output-dir", output_dir]
    print(f"CMD: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd)

    results_dir = f"/results/{run_name}"
    if os.path.isdir(output_dir):
        shutil.copytree(output_dir, results_dir, dirs_exist_ok=True)
        results_vol.commit()
        print(f"\nResults saved to {RESULTS_VOLUME}:/{run_name}")
    return {"run_name": run_name, "returncode": proc.returncode}


@app.local_entrypoint()
def train_ratio_ladder(names: str):
    """Run ratio-ladder points in parallel.
    modal run modal_ablation.py::train_ratio_ladder --names sim34,sim68,sim120"""
    pts = [n.strip() for n in names.split(",") if n.strip()]
    bad = [n for n in pts if n not in RATIO_POINTS]
    if bad:
        raise SystemExit(f"unknown ratio points: {bad}; known: {list(RATIO_POINTS)}")
    print(f"Launching {len(pts)} ratio runs: {pts}", flush=True)
    handles = {n: train_ratio_one.spawn(n) for n in pts}
    for n, h in handles.items():
        try:
            r = h.get()
            print(f"  {r['run_name']}: {'OK' if r['returncode'] == 0 else 'FAIL'}", flush=True)
        except Exception as e:
            print(f"  {ratio_run_name(n)}: ERROR - {e}", flush=True)


@app.local_entrypoint()
def download_ratio(names: str = "", history_only: bool = False, dest: str = "outputs"):
    """Download ratio-ladder results (history.json [+ checkpoints]) from the results volume."""
    import time
    vol = modal.Volume.from_name(RESULTS_VOLUME)
    pts = [n.strip() for n in names.split(",") if n.strip()] or list(RATIO_POINTS)
    for n in pts:
        run = ratio_run_name(n)
        local_dir = os.path.join(dest, run)
        try:
            files = list(vol.listdir(f"/{run}"))
        except Exception as e:
            print(f"  SKIP {run}: {e}")
            continue
        os.makedirs(local_dir, exist_ok=True)
        for f in files:
            fname = os.path.basename(f.path)
            if history_only and fname != "history.json":
                continue
            if not fname.endswith((".pt", ".json", ".yaml")):
                continue
            local_path = os.path.join(local_dir, fname)
            if fname.endswith(".pt") and os.path.exists(local_path):
                continue
            time.sleep(1)
            with open(local_path, "wb") as fout:
                for chunk in vol.read_file(f.path):
                    fout.write(chunk)
            print(f"  {run}/{fname}", flush=True)
    print("Done.")


# ============================================================
# Entrypoints
# ============================================================

@app.local_entrypoint()
def train(tac: str, rgb: str):
    """Run one training combo.  modal run modal_ablation.py::train --tac t3 --rgb clip"""
    result = train_one.remote(tac, rgb)
    status = "OK" if result["returncode"] == 0 else f"FAIL (exit {result['returncode']})"
    print(f"{result['run_name']}: {status}")


@app.local_entrypoint()
def train_tf(tac: str, rgb: str):
    """Run one transfilt combo.  modal run modal_ablation.py::train_tf --tac t3 --rgb mae"""
    result = train_one.remote(tac, rgb, transfilt=True)
    status = "OK" if result["returncode"] == 0 else f"FAIL (exit {result['returncode']})"
    print(f"{result['run_name']}: {status}")


@app.local_entrypoint()
def train_all():
    """Run all 24 combos."""
    combos = list(itertools.product(TACTILE_KEYS, RGB_KEYS))
    print(f"Launching {len(combos)} runs...")
    handles = [train_one.spawn(t, r) for t, r in combos]
    for h, (t, r) in zip(handles, combos):
        try:
            result = h.get()
            s = "OK" if result["returncode"] == 0 else "FAIL"
            print(f"  {result['run_name']}: {s}")
        except Exception as e:
            print(f"  {output_name(t, r)}: ERROR - {e}")


@app.local_entrypoint()
def train_remaining():
    """Run only combos not done locally."""
    combos = [(t, r) for t, r in itertools.product(TACTILE_KEYS, RGB_KEYS)
              if (t, r) not in DONE_LOCALLY]
    print(f"Launching {len(combos)} runs (skipping {len(DONE_LOCALLY)} done locally)...")
    handles = [train_one.spawn(t, r) for t, r in combos]
    for h, (t, r) in zip(handles, combos):
        try:
            result = h.get()
            s = "OK" if result["returncode"] == 0 else "FAIL"
            print(f"  {result['run_name']}: {s}")
        except Exception as e:
            print(f"  {output_name(t, r)}: ERROR - {e}")


@app.local_entrypoint()
def train_all_transfilt():
    """Run all 24 combos with translation filtering."""
    combos = list(itertools.product(TACTILE_KEYS, RGB_KEYS))
    print(f"Launching {len(combos)} transfilt runs...")
    handles = [train_one.spawn(t, r, transfilt=True) for t, r in combos]
    for h, (t, r) in zip(handles, combos):
        try:
            result = h.get()
            s = "OK" if result["returncode"] == 0 else "FAIL"
            print(f"  {result['run_name']}: {s}")
        except Exception as e:
            print(f"  ablation_g3s_sim315_tf_{t}_{r}: ERROR - {e}")


@app.local_entrypoint()
def download_results():
    """Download checkpoints + history from Modal to local outputs/."""
    import re
    vol = modal.Volume.from_name(RESULTS_VOLUME)
    for entry in vol.listdir("/"):
        remote_name = entry.path.strip("/")
        # Rename: ablation_{tac}_{rgb}_g3s_sim315 -> ablation_g3s_sim315_{tac}_{rgb}
        m = re.match(r"ablation_(.+)_g3s_sim315$", remote_name)
        if m:
            local_name = f"ablation_g3s_sim315_{m.group(1)}"
        else:
            local_name = remote_name
        local_dir = os.path.join("outputs", local_name)
        os.makedirs(local_dir, exist_ok=True)
        for f in vol.listdir(f"/{remote_name}"):
            fname = os.path.basename(f.path)
            if fname.endswith((".pt", ".json", ".yaml")):
                local_path = os.path.join(local_dir, fname)
                with open(local_path, "wb") as fout:
                    for chunk in vol.read_file(f.path):
                        fout.write(chunk)
                print(f"  {local_name}/{fname}")
    print("Download complete.")


# ============================================================
# CLI: python modal_ablation.py upload
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "upload":
        do_upload()
    else:
        print(__doc__)
