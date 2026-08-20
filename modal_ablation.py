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
TIMEOUT_HOURS = 12
REPO_URL = "https://github.com/cynthiahuang1004/VisTacFusion.git"
REPO_BRANCH = "VisTacFusion-v2"

TACTILE_KEYS = ["t3", "dinov3", "sparshv2", "sparshmae", "sitr", "dav2"]
RGB_KEYS = ["mae", "dinov3", "clip", "siglip"]

DONE_LOCALLY = {
    ("t3", "mae"),
    ("dinov3", "mae"),
    ("sitr", "mae"),
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
        "tqdm", "timm", "scipy", "Pillow",
    )
    .apt_install("git")
)

# ============================================================
# Helpers
# ============================================================

def model_config(tac: str, rgb: str) -> str:
    return f"ablation/encoder/tac_{tac}_rgb_{rgb}.yaml"

def output_name(tac: str, rgb: str) -> str:
    return f"ablation_{tac}_{rgb}_g3s_sim315"


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
    "/media/hdd2/ihsuan/sparsh/weights/sparsh/mae_vitbase.ckpt": "/data/weights/sparsh_mae_base.ckpt",
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
def train_one(tac: str, rgb: str):
    """Train one encoder combo on a Modal GPU."""
    import shutil, subprocess, yaml

    run_name = output_name(tac, rgb)
    print(f"\n{'='*60}")
    print(f"  {run_name}  (tac={tac}, rgb={rgb})")
    print(f"{'='*60}\n")

    # Clone repo
    subprocess.run(
        ["git", "clone", "--branch", REPO_BRANCH, "--depth", "1",
         REPO_URL, "/workspace"],
        check=True, capture_output=True,
    )
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
            "train_samples_per_session": 315,
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
# Entrypoints
# ============================================================

@app.local_entrypoint()
def train(tac: str, rgb: str):
    """Run one training combo.  modal run modal_ablation.py::train --tac t3 --rgb clip"""
    result = train_one.remote(tac, rgb)
    status = "OK" if result["returncode"] == 0 else f"FAIL (exit {result['returncode']})"
    print(f"{result['run_name']}: {status}")


@app.local_entrypoint()
def train_all():
    """Run all 24 combos."""
    combos = list(itertools.product(TACTILE_KEYS, RGB_KEYS))
    print(f"Launching {len(combos)} runs...")
    for r in train_one.starmap(combos):
        s = "OK" if r["returncode"] == 0 else "FAIL"
        print(f"  {r['run_name']}: {s}")


@app.local_entrypoint()
def train_remaining():
    """Run only combos not done locally."""
    combos = [(t, r) for t, r in itertools.product(TACTILE_KEYS, RGB_KEYS)
              if (t, r) not in DONE_LOCALLY]
    print(f"Launching {len(combos)} runs (skipping {len(DONE_LOCALLY)} done locally)...")
    for r in train_one.starmap(combos):
        s = "OK" if r["returncode"] == 0 else "FAIL"
        print(f"  {r['run_name']}: {s}")


@app.local_entrypoint()
def download_results():
    """Download checkpoints + history from Modal to local outputs/."""
    vol = modal.Volume.from_name(RESULTS_VOLUME)
    for entry in vol.listdir("/"):
        run = entry.path.strip("/")
        local_dir = os.path.join("outputs", run)
        os.makedirs(local_dir, exist_ok=True)
        for f in vol.listdir(f"/{run}"):
            fname = os.path.basename(f.path)
            if fname.endswith((".pt", ".json", ".yaml")):
                local_path = os.path.join(local_dir, fname)
                with open(local_path, "wb") as fout:
                    for chunk in vol.read_file(f.path):
                        fout.write(chunk)
                print(f"  {run}/{fname}")
    print("Download complete.")


# ============================================================
# CLI: python modal_ablation.py upload
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "upload":
        do_upload()
    else:
        print(__doc__)
