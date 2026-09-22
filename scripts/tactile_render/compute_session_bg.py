"""Per-session tactile background (no-contact appearance) as the per-pixel median over a
session's frames — contacts move around and cover ~10% of pixels, so the median is the
no-contact image. Needs no labels (a deployed sensor would simply grab a no-contact frame).

Writes outputs/session_bg/<domain>/<object>__<session>.png and, for real, ref.png
(mean of the real session backgrounds = the canonical appearance everything is mapped to).
usage: python scripts/tactile_render/compute_session_bg.py real | sim:<tactile_subdir>
"""
import glob
import os
import os.path as osp
import sys

import numpy as np
from PIL import Image

ROOTS = {"real": "/media/hdd2/ihsuan/gs_blender/real_filtered",
         "sim": "/media/hdd2/ihsuan/gs_blender/renders_v3"}
OUT = "outputs/session_bg"


def main(spec):
    dom, _, sub = spec.partition(":")
    sub = sub or "samples"
    tag = "real" if dom == "real" else f"sim_{sub}"
    os.makedirs(osp.join(OUT, tag), exist_ok=True)
    bgs = []
    for unit in sorted(glob.glob(f"{ROOTS[dom]}/pattern_*/session_*/sensor_0000")):
        fs = sorted(glob.glob(osp.join(unit, sub, "*.png")))
        if not fs:
            continue
        fs = fs[:: max(1, len(fs) // 60)]
        stack = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in fs])
        bg = np.median(stack, 0).astype(np.uint8)
        obj, sess = unit.split("/")[-3], unit.split("/")[-2]
        Image.fromarray(bg).save(osp.join(OUT, tag, f"{obj}__{sess}.png"))
        bgs.append(bg.astype(np.float32))
    if dom == "real":
        Image.fromarray(np.mean(bgs, 0).round().astype(np.uint8)).save(osp.join(OUT, "ref.png"))
    print(tag, len(bgs), "sessions")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(s)
