"""Download transfilt (tf_) MAE ablation results from Modal volume.

Usage:
  python scripts/modal_download_tf.py --history-only   # just history.json (fast)
  python scripts/modal_download_tf.py                  # history + checkpoints
"""
import modal, os, sys, time

vol = modal.Volume.from_name("vistacfusion-results")
HISTORY_ONLY = "--history-only" in sys.argv
TACS = ["t3", "dinov3", "sparshv2", "sparshmae", "sitr", "dav2"]

for tac in TACS:
    run = f"ablation_g3s_sim315_tf_{tac}_mae"
    local_dir = os.path.join("outputs", run)
    os.makedirs(local_dir, exist_ok=True)
    time.sleep(2)
    try:
        files = list(vol.listdir(f"/{run}"))
    except Exception as ex:
        print(f"  SKIP {run}: {ex}")
        continue
    if not files:
        print(f"  SKIP {run} (empty)")
        continue
    for f in files:
        fname = os.path.basename(f.path)
        if HISTORY_ONLY and fname != "history.json":
            continue
        if not fname.endswith((".pt", ".json", ".yaml")):
            continue
        local_path = os.path.join(local_dir, fname)
        if os.path.exists(local_path) and fname.endswith(".pt"):
            print(f"  exists {run}/{fname}")
            continue
        time.sleep(1)
        try:
            with open(local_path, "wb") as fout:
                for chunk in vol.read_file(f.path):
                    fout.write(chunk)
            print(f"  {run}/{fname}")
        except Exception as ex:
            print(f"  FAIL {run}/{fname}: {ex}")

print("Done.")
