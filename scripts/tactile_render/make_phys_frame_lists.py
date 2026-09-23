"""Frame lists for the physics renders at real poses (real_filtered_phys).

Per object: the union of the K-shot train subsets for K in --ks (same linspace rule as
RealPairs.per_object / SimVisuoTactileDataset.train_samples_per_session) plus every val frame
(idx % VAL_EVERY == 0), so GAN-K (any K), GAN-LOO(per-object 100) and eval_renderer all have
their physics condition. Writes <out>/<obj>.json (list of ints) and prints the counts.
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from train_tactile_gan import REAL_ROOT, VAL_EVERY  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ks", nargs="*", type=int, default=[10, 25, 50, 100])
ap.add_argument("--out", default="outputs/phys_frames")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
total = 0
for od in sorted(glob.glob(f"{REAL_ROOT}/pattern_*")):
    obj = osp.basename(od)
    idxs = sorted(int(osp.splitext(osp.basename(p))[0]) for p in glob.glob(f"{od}/session_000/sensor_0000/samples/*.png"))
    train = [i for i in idxs if i % VAL_EVERY != 0]
    val = [i for i in idxs if i % VAL_EVERY == 0]
    # priority order: small-K subsets first (K=10, 25), then every 3rd val frame, then K=50, K=100, remaining val
    keep, seen = [], set()
    def add(lst):
        for i in lst:
            if i not in seen:
                seen.add(i); keep.append(i)
    def subset(k):
        if len(train) > k:
            pos = np.linspace(0, len(train) - 1, k).round().astype(int)
            return [train[i] for i in sorted(set(pos.tolist()))]
        return train
    add(subset(10)); add(subset(25)); add(val[::3]); add(subset(50)); add(subset(100)); add(val)
    json.dump(keep, open(f"{args.out}/{obj}.json", "w"))
    print(f"{obj:32s} frames={len(idxs):4d} val={len(val):3d} keep={len(keep)}")
    total += len(keep)
print("total", total)
