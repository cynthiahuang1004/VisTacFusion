"""Merge locally rendered physics tactile images (phys_upload/) into real_filtered_phys/ and verify
every frame in phys_frames/<obj>.json exists, is a valid 224x224 PNG, and nothing extra is present.
usage: python scripts/tactile_render/merge_phys_upload.py [--src /media/hdd2/ihsuan/gs_blender/phys_upload]
"""
import argparse, glob, json, os, os.path as osp, shutil
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/media/hdd2/ihsuan/gs_blender/phys_upload")
ap.add_argument("--dst", default="/media/hdd2/ihsuan/gs_blender/real_filtered_phys")
ap.add_argument("--lists", default="/media/hdd2/ihsuan/gs_blender/phys_frames")
a = ap.parse_args()
# tolerate one extra nesting level (scp -r into an existing dir)
src = a.src
if not glob.glob(f"{src}/pattern_*") and glob.glob(f"{src}/*/pattern_*"):
    src = glob.glob(f"{src}/*")[0]
moved = 0
for p in sorted(glob.glob(f"{src}/pattern_*/session_000/sensor_0000/samples/*.png")):
    rel = osp.relpath(p, src); d = osp.join(a.dst, rel); os.makedirs(osp.dirname(d), exist_ok=True)
    if not osp.exists(d) or osp.getsize(d) < 1000:
        shutil.copy2(p, d); moved += 1
print(f"copied {moved} new files from {src}")
ok = True; total = 0
for lst in sorted(glob.glob(f"{a.lists}/*.json")):
    obj = osp.basename(lst)[:-5]; want = set(json.load(open(lst)))
    have = {int(osp.basename(f)[:-4]) for f in glob.glob(f"{a.dst}/{obj}/session_000/sensor_0000/samples/*.png")}
    bad = []
    for i in want & have:
        try:
            im = Image.open(f"{a.dst}/{obj}/session_000/sensor_0000/samples/{i:04d}.png"); im.verify()
            if im.size != (224, 224): bad.append(i)
        except Exception: bad.append(i)
    miss = sorted(want - have); extra = sorted(have - want); total += len(want)
    print(f"{obj:32s} want {len(want):3d} have {len(have & want):3d} missing {len(miss):3d} extra {len(extra):3d} bad {len(bad)}" + (f"  missing e.g. {miss[:5]}" if miss else ""))
    ok &= not miss and not bad
print("ALL COMPLETE" if ok else "INCOMPLETE", "| total frames", total)
