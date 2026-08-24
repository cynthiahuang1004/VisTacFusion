"""Run 24 transfilt ablation on Modal: 6 MAE first, then 18 others."""
import subprocess, sys

TACS = ["t3", "dinov3", "sparshv2", "sparshmae", "sitr", "dav2"]
RGBS = ["mae", "dinov3", "clip", "siglip"]

# Phase 1: 6 MAE combos (priority)
mae_combos = [(t, "mae") for t in TACS]

# Phase 2: 18 non-MAE combos
other_combos = [(t, r) for t in TACS for r in RGBS if r != "mae"]

print(f"=== Phase 1: {len(mae_combos)} MAE combos ===")
procs = []
for tac, rgb in mae_combos:
    log = f"outputs/modal_tf_{tac}_{rgb}.log"
    print(f"  Spawning {tac}+{rgb}")
    p = subprocess.Popen(
        ["modal", "run", "modal_ablation.py::train_tf", "--tac", tac, "--rgb", rgb],
        stdout=open(log, "w"), stderr=subprocess.STDOUT
    )
    procs.append((tac, rgb, p, log))

# Wait for all 6 MAE to finish
print("\nWaiting for 6 MAE combos to finish...")
for tac, rgb, p, log in procs:
    rc = p.wait()
    status = "OK" if rc == 0 else f"FAIL (exit {rc})"
    print(f"  {tac}+{rgb}: {status}")

print(f"\n=== Phase 2: {len(other_combos)} non-MAE combos ===")
procs2 = []
for tac, rgb in other_combos:
    log = f"outputs/modal_tf_{tac}_{rgb}.log"
    print(f"  Spawning {tac}+{rgb}")
    p = subprocess.Popen(
        ["modal", "run", "modal_ablation.py::train_tf", "--tac", tac, "--rgb", rgb],
        stdout=open(log, "w"), stderr=subprocess.STDOUT
    )
    procs2.append((tac, rgb, p, log))

print("\nWaiting for 18 non-MAE combos to finish...")
for tac, rgb, p, log in procs2:
    rc = p.wait()
    status = "OK" if rc == 0 else f"FAIL (exit {rc})"
    print(f"  {tac}+{rgb}: {status}")

print("\nAll 24 transfilt runs complete.")
